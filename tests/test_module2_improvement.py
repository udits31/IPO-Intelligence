import numpy as np
import pandas as pd
import gzip,hashlib,sqlite3
import ml.recover_module2_daily_news as recovery
from ml.improve_module2_next_session import positions,realised,evergreen
from ml.recover_module2_daily_news import parse_feed


def test_policy_is_causal_and_does_not_leak_position_between_issuers():
    f=pd.DataFrame({'ipo_id':[1,1,2], 'decision_date':pd.to_datetime(['2025-01-01','2025-01-02','2025-01-01']),
                    'rule_position':[1,1,1]})
    p=positions(f,np.array([.5,.1,.1]),('return_hysteresis',.35,-.15))
    assert p.tolist()==[1,1,0]
    assert positions(f.iloc[:1],np.array([.5]),('return_hysteresis',.35,-.15)).iloc[0]==p.iloc[0]


def test_execution_sells_at_following_open_and_carries_overnight():
    f=pd.DataFrame({'ipo_id':[1,1],'company_name':['A','A'],'listing_year':[2025,2025],
        'outcome_date':pd.to_datetime(['2025-01-02','2025-01-03']),
        'next_open':[100.,120.],'next_close':[110.,90.]})
    r=realised(f,[1,0],35)[0]
    assert np.isclose(r['net_return_pct'],100*(1.2*(1-.00175)**2-1))
    assert r['trades'][1]['price']==120.
    assert r['buys']==r['sells']==1


def test_feed_preserves_dates_and_rejects_other_issuer_and_outside_window():
    raw=b'''<rss><channel>
    <item><title>CMS Info Systems wins new contract - Publisher</title><link>https://example.com/a</link><pubDate>Tue, 08 Feb 2022 08:00:00 GMT</pubDate></item>
    <item><title>Reliance Industries wins new contract</title><link>https://example.com/b</link><pubDate>Tue, 08 Feb 2022 08:00:00 GMT</pubDate></item>
    <item><title>CMS Info Systems earnings rise</title><link>https://example.com/c</link><pubDate>Thu, 08 Feb 2024 08:00:00 GMT</pubDate></item>
    </channel></rss>'''
    rows,rejected,n=parse_feed(raw,'CMS Info Systems Ltd.',pd.Timestamp('2022-02-01',tz='Asia/Kolkata'),pd.Timestamp('2022-03-01',tz='Asia/Kolkata'))
    assert n==3 and len(rows)==1
    assert 'order' in rows[0]['event_tags']
    assert pd.Timestamp(rows[0]['published_at'])==pd.Timestamp('2022-02-08T08:00Z')
    assert rejected['outside_requested_dates']==rejected['identity_or_directory']==1
    assert evergreen('Technical Analysis of CMS Info Systems Ltd. (NSE:CMSINFO)')


def test_news_database_appends_source_versions_without_rewriting_history(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    path=tmp_path/'source.xml.gz';database=tmp_path/'news.db'
    def record(raw):
        path.write_bytes(gzip.compress(raw))
        return dict(status='verified_rss_response',ipo_id=1,source_url='https://example.com/rss',
            raw_sha256=hashlib.sha256(raw).hexdigest(),retrieved_at_utc='2026-10-29T01:00:00+00:00',archive_path='source.xml.gz')
    first=record(b'<rss>first</rss>')
    assert recovery.persist_sources([first],database)==1
    assert recovery.persist_sources([first],database)==1
    second=record(b'<rss>new response</rss>')
    assert recovery.persist_sources([second],database)==2
    with sqlite3.connect(database) as c:
        rows=c.execute('SELECT raw_sha256,raw_rss_gzip FROM historical_news_sources_v1').fetchall()
    assert any(sha==first['raw_sha256'] and gzip.decompress(raw)==b'<rss>first</rss>' for sha,raw in rows)
