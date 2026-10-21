"""Historical public tables must remain distinct from originally issued forecasts."""

import gzip
import hashlib
import json
import sqlite3
from datetime import date

import pytest
from services import historical_bid_service as history
from services import snapshot_service as snapshots


ET = 'https://economictimes.indiatimes.com/bonfiglioli-transmissions-ltd/ipos/companyid-2450233.cms'
MC = 'https://www.moneycontrol.com/ipo/bonfiglioli-transmissions-btl04-ipodetail'
ITEM = {'company_name': 'Bonfiglioli Transmissions', 'date_range': '15 - 19 Oct',
        'url': 'https://www.chittorgarh.com/ipo/bonfiglioli-transmissions-ipo/2861/',
        'category': 'CLOSED'}
# Category order differs between these publishers. Values match the retrieved
# 16 October rows, while final-day totals must never enter this reconstruction.
ET_HTML = '''<h1>Bonfiglioli Transmissions Ltd. IPO</h1><table>
<tr><th>Date</th><th>QIBs (ex-anchors)</th><th>NIIs</th><th>RIIs</th></tr>
<tr><td>Day 1 (15 Oct 2026)</td><td>0.08x</td><td>0.78x</td><td>0.57x</td></tr>
<tr><td>Day 2 (16 Oct 2026)</td><td>0.16x</td><td>1.71x</td><td>1.16x</td></tr>
<tr><td>Day 3 (19 Oct 2026)</td><td>120.78x</td><td>40.40x</td><td>5.91x</td></tr>
</table>'''
MC_HTML = '''<h1>Bonfiglioli Transmissions Mainline</h1><table>
<tr><th>Date</th><th>QIB</th><th>RII</th><th>NII</th></tr>
<tr><td>Day 2 16-10-2026</td><td>0.16x</td><td>1.16x</td><td>1.71x</td></tr>
</table>'''
TIMED = 'https://theipolist.in/ipo-details/bonfiglioli-transmissions'
TIMED_HTML = '''<h1>Bonfiglioli Transmissions</h1>
<div><h3>Issue open date</h3><p>15 Oct 2026</p></div>
<div><h3>Issue close date</h3><p>19 Oct 2026</p></div>
<section><h2>IPO Subscription Status</h2><table>
<tr><th>Category</th><th>Times</th><th>Allotment Chances</th></tr>
<tr><td>bHNIs</td><td>90.00</td><td>0.01</td></tr>
<tr><td>HNIs (Total)</td><td>40.36</td><td>0.01</td></tr>
<tr><td>QIBs</td><td>120.78</td><td>0.01</td></tr>
<tr><td>Retail</td><td>5.78</td><td>0.01</td></tr></table>
<p>Last updated: 19 October'26 04:44PM</p></section>
<section><h2>GMP</h2><p>Last updated: 21 October'26 10:45AM</p></section>'''


def test_date_and_category_order_and_closing_day_exclusion():
    args = ('Bonfiglioli Transmissions', date(2026, 10, 15), date(2026, 10, 19))
    first = history.parse_daywise(ET_HTML, *args)
    second = history.parse_daywise(MC_HTML, *args)
    assert first[-1] == second[-1] == {'observed_on': '2026-10-16', 'qib': .16, 'hni': 1.71, 'retail': 1.16}
    assert len(first) == 2
    assert history.parse_daywise(ET_HTML, 'Another Company', *args[1:]) == []
    assert history.parse_daywise(MC_HTML.replace('0.16x', 'N.A.'), *args) == []


def test_recovered_score_uses_prior_day_inputs_and_preserves_raw_pages(tmp_path, monkeypatch):
    db = str(tmp_path / 'snapshots.db')
    snapshots.init_db(db)
    monkeypatch.setattr(history, 'discover_sources', lambda _: [ET, MC])
    monkeypatch.setattr(history, '_fetch', lambda url: (url, ET_HTML if url == ET else MC_HTML))
    result = history.recover_historical_bid_forecast(ITEM, db_path=db)
    expected = snapshots._scheduled_predictor().score_interim_bid_snapshot(qib_sub=.16, hni_sub=1.71, retail_sub=1.16)
    assert result['positive_listing_probability'] == expected['positive_listing_probability']
    assert result['observed_on'] == '2026-10-16'
    assert result['eligible_for_prospective_accuracy'] is False
    assert len(result['sources']) == 2
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM automatic_decision_v1').fetchone()[0] == 0
        pages = conn.execute('SELECT raw_sha256,raw_html_gzip FROM historical_bid_page_v1').fetchall()
    assert len(pages) == 2
    assert all(hashlib.sha256(gzip.decompress(blob)).hexdigest() == fingerprint for fingerprint, blob in pages)
    monkeypatch.setattr(history, '_fetch', lambda _: (_ for _ in ()).throw(AssertionError('Should use saved evidence')))
    assert history.recover_historical_bid_forecast(ITEM, db_path=db)['computed_at_ist'] == result['computed_at_ist']


def test_materially_disagreeing_tables_are_not_blended(tmp_path, monkeypatch):
    db = str(tmp_path / 'snapshots.db')
    monkeypatch.setattr(history, 'discover_sources', lambda _: [ET, MC])
    # Both sources now contain just one date; the QIB discrepancy blocks it.
    primary = MC_HTML
    other = MC_HTML.replace('0.16x', '9.50x')
    monkeypatch.setattr(history, '_fetch', lambda url: (url, primary if url == ET else other))
    assert history.recover_historical_bid_forecast(ITEM, db_path=db) is None


def test_timestamp_scope_categories_issuer_and_offer_dates():
    args = ('Bonfiglioli Transmissions', date(2026, 10, 15), date(2026, 10, 19), TIMED)
    rows = history.parse_timestamped_snapshot(TIMED_HTML, *args)
    assert rows[0]['source_reported_at_ist'] == '2026-10-19T16:44:00+05:30'
    assert (rows[0]['qib'], rows[0]['hni'], rows[0]['retail']) == (120.78, 40.36, 5.78)
    assert history.parse_timestamped_snapshot(TIMED_HTML, 'Unrelated Company', *args[1:]) == []
    assert history.parse_timestamped_snapshot(TIMED_HTML.replace('19 Oct 2026', '20 Oct 2026'), *args) == []
    assert history.parse_timestamped_snapshot(TIMED_HTML.replace('120.78', 'Unavailable'), *args) == []
    assert history.parse_timestamped_snapshot(TIMED_HTML.replace('120.78', '0.00'), *args)[0]['qib'] == 0


@pytest.mark.parametrize(('stamp', 'accepted'), [('04:59PM', True), ('05:00PM', False), ('05:34PM', False)])
def test_timestamp_must_be_strictly_before_own_cutoff(stamp, accepted):
    rows = history.parse_timestamped_snapshot(TIMED_HTML.replace('04:44PM', stamp),
        'Bonfiglioli Transmissions', date(2026, 10, 15), date(2026, 10, 19), TIMED)
    assert bool(rows) is accepted


def test_niftytrader_scoped_bid_time_not_later_gmp_time():
    html = '''<h1>Bonfiglioli Transmissions IPO</h1>
    <h2>Bonfiglioli Transmissions IPO Subscriptions Status</h2>
    <p>Subscribed by October 19, 2026 2:34:58 PM (Day 3)</p>
    <table><tr><td>QIB (Ex Anchor)</td><td>20.00</td></tr>
    <tr><td>NII</td><td>28.22</td></tr><tr><td>Retail</td><td>4.35</td></tr></table>
    <h2>GMP</h2><p>by October 20, 2026 6:00:00 PM</p>'''
    row = history.parse_timestamped_snapshot(html, 'Bonfiglioli Transmissions',
        date(2026, 10, 15), date(2026, 10, 19), 'https://niftytrader.in/ipo/bonfiglioli-transmissions-ipo')[0]
    assert row['source_reported_at_ist'] == '2026-10-19T14:34:58+05:30'
    assert (row['qib'], row['hni'], row['retail']) == (20, 28.22, 4.35)


def test_new_timed_evidence_upgrades_old_cache_and_sorts_by_observation(tmp_path, monkeypatch):
    db = str(tmp_path / 'snapshots.db')
    snapshots.init_db(db)
    monkeypatch.setattr(history, 'discover_sources', lambda _: [ET])
    monkeypatch.setattr(history, '_fetch', lambda u: (u, ET_HTML))
    old = history.recover_historical_bid_forecast(ITEM, db_path=db)
    # Simulate a cache produced by the earlier date-only implementation.
    with sqlite3.connect(db) as conn:
        data = json.loads(conn.execute('SELECT input_json FROM historical_bid_reconstruction_v1 WHERE id=?', (old['id'],)).fetchone()[0])
        data.pop('recovery_protocol')
        conn.execute('UPDATE historical_bid_reconstruction_v1 SET input_json=? WHERE id=?', (json.dumps(data), old['id']))
    monkeypatch.setattr(history, 'discover_sources', lambda _: [TIMED])
    monkeypatch.setattr(history, '_fetch', lambda u: (u, TIMED_HTML))
    latest = history.recover_historical_bid_forecast(ITEM, db_path=db)
    assert latest['observed_on'] == '2026-10-19'
    assert latest['source_reported_at_ist'].endswith('16:44:00+05:30')
    assert latest['inputs']['qib_subscription'] == 120.78
    assert latest['eligible_for_prospective_accuracy'] is False
    assert len(latest['history']) == 3
    # Older same-day observation fetched later must not displace 16:44.
    monkeypatch.setattr(history, '_fetch', lambda u: (u, TIMED_HTML.replace('04:44PM', '02:30PM').replace('120.78', '20.00')))
    result = history.recover_historical_bid_forecast(ITEM, db_path=db, force=True)
    assert result['inputs']['qib_subscription'] == 120.78
    assert len([r for r in result['snapshot_history'] if r['observed_on'] == '2026-10-19']) == 2
    assert result['source_conflicts'] == []
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM automatic_decision_v1').fetchone()[0] == 0
        for fingerprint, blob in conn.execute('SELECT raw_sha256,raw_html_gzip FROM historical_bid_page_v1'):
            assert hashlib.sha256(gzip.decompress(blob)).hexdigest() == fingerprint


def test_discovery_includes_timestamped_calendar_links(monkeypatch):
    monkeypatch.setattr(history, '_index_cache', (0, {}))
    monkeypatch.setattr(history, '_timed_index_cache', (0, {}))
    monkeypatch.setattr(history, '_fetch', lambda u: (u,
        '<a href="/ipo-details/new-issuer">New Issuer Mainboard</a>' if 'theipolist' in u else ''))
    assert 'https://theipolist.in/ipo-details/new-issuer' in history.discover_sources('New Issuer')


CHOICE_HTML = '''<h1>Runwal Enterprises Limited IPO Subscription Update</h1>
<script type="application/ld+json">{"@type":"NewsArticle",
"datePublished":"2026-09-29T08:56:00Z","dateModified":"2026-09-29T08:56:00Z"}</script>
<main><p>Final day till 29-Sep-2026 14:24 Pm.</p><table>
<tr><th>SR. NO.</th><th>CATEGORY</th><th>NO OF TIMES</th></tr>
<tr><td>1</td><td>Qualified Institutional Buyers (QIBs)</td><td>2.20</td></tr>
<tr><td>2</td><td>Non-Institutional Investors</td><td>2.85</td></tr>
<tr><td>3</td><td>Retail Individual Investors (RIIs)</td><td>0.82</td></tr></table></main>'''


def test_article_bid_clock_is_separate_from_publication_and_url_slug():
    args = ('Runwal Enterprises', date(2026, 9, 25), date(2026, 9, 29),
            'https://choiceindia.com/news/lumino-industries-ipo-subscription-day-3')
    row = history.parse_article_snapshot(CHOICE_HTML, *args)[0]
    assert row['source_reported_at_ist'] == '2026-09-29T14:24:00+05:30'
    assert row['publication_times_ist']['datePublished'] == '2026-09-29T14:26:00+05:30'
    assert row['evidence_warnings'] == []
    assert (row['qib'], row['hni'], row['retail']) == (2.2, 2.85, .82)
    assert history.parse_article_snapshot(CHOICE_HTML, 'Lumino Industries', *args[1:]) == []
    assert history.parse_article_snapshot(CHOICE_HTML.replace('14:24', '17:00'), *args) == []
    conflict = history.parse_article_snapshot(CHOICE_HTML.replace('14:24', '15:24'), *args)[0]
    assert history.evidence_severity(conflict) == 2


def test_official_issue_overlay_fills_missing_dates_and_checks_price(tmp_path):
    path = tmp_path / 'nse.json'
    path.write_text(json.dumps([{'company':'Nityas Gems and Jewellery Limited',
        'ipoStartDate':'30-SEP-2026','ipoEndDate':'05-OCT-2026','listingDate':'08-OCT-2026',
        'symbol':'NITYAS','issuePrice':'75'}]))
    item = {'company_name':'Nityas Gems & Jewellery','issue_price':75,'date_range':''}
    result = history.resolve_official_issue(item,path=path)
    assert result['date_range'] == '2026-09-30 to 2026-10-05'
    assert result['nse_symbol'] == 'NITYAS'
    assert result['official_offer_evidence']['raw_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert history.resolve_official_issue({**item,'issue_price':99},path=path)['official_identity_error']


def test_clean_same_day_reading_beats_later_conflicted_article(tmp_path,monkeypatch):
    db = str(tmp_path/'history.db')
    item = {'company_name':'Runwal Enterprises','date_range':'25 - 29 Sep'}
    url = 'https://choiceindia.com/news/lumino-industries-ipo-subscription-day-3'
    monkeypatch.setattr(history,'discover_sources',lambda _: [url])
    monkeypatch.setattr(history,'_fetch',lambda u:(u,CHOICE_HTML))
    first = history.recover_historical_bid_forecast(item,db_path=db)
    monkeypatch.setattr(history,'_fetch',lambda u:(u,CHOICE_HTML.replace('14:24','16:00').replace('2.20','9.99')))
    second = history.recover_historical_bid_forecast(item,db_path=db,force=True)
    assert second['source_reported_at_ist'] == first['source_reported_at_ist']
    assert second['inputs']['qib_subscription'] == 2.2
    assert len(second['snapshot_history']) == 2
