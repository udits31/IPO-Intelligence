import hashlib
import json
import sqlite3
from datetime import datetime,timedelta,timezone

import pytest

from services.module2_intraday import IST,analyse,parse_candles,capture,read


def bars(n=17):
    first=datetime(2026,10,26,9,15,tzinfo=IST)
    return [dict(bar_start_utc=(first+timedelta(minutes=i)).astimezone(timezone.utc).isoformat(),
        session_date='2026-10-26',open=100.,high=101.,low=99.,close=100.,volume=100.) for i in range(n)]


def payload(rows,symbol='TEST.NS'):
    return {'chart':{'result':[{'meta':{'symbol':symbol,'currency':'INR'},
        'timestamp':[int(datetime.fromisoformat(r['bar_start_utc']).timestamp()) for r in rows],
        'indicators':{'quote':[{k:[r[k] for r in rows] for k in ['open','high','low','close','volume']}]}}]}}


def test_incomplete_candle_is_never_used_and_wrong_security_rejected():
    rows=bars(3);now=datetime.fromisoformat(rows[1]['bar_start_utc'])+timedelta(seconds=30)
    parsed,_=parse_candles(payload(rows),'TEST.NS',now)
    assert len(parsed)==1
    with pytest.raises(ValueError,match='Instrument'):
        parse_candles(payload(rows),'OTHER.NS',now)


def test_two_completed_bars_confirm_opening_range_break():
    rows=bars()
    for r in rows[-2:]:r.update(low=90.,close=91.)
    now=datetime.fromisoformat(rows[-1]['bar_start_utc'])+timedelta(minutes=1)
    r=analyse(rows,now)
    assert r['signal']=='REVIEW_EXIT' and r['alert_eligible']
    assert r['opening_range_low']==99.
    # One weak candle alone is not a confirmed opening-range exit signal.
    rows[-2].update(low=99.,close=100.)
    assert analyse(rows,now)['signal']!='REVIEW_EXIT'


def test_no_active_alert_from_old_or_previous_day_prices():
    rows=bars()
    for r in rows[-2:]:r.update(low=90.,close=91.)
    now=datetime.fromisoformat(rows[-1]['bar_start_utc'])+timedelta(minutes=8)
    r=analyse(rows,now)
    assert r['status']=='DELAYED' and not r['alert_eligible']
    assert analyse(rows,now+timedelta(days=1))['status']=='UNAVAILABLE'


def test_session_end_and_gaps_disable_active_exit_confirmation():
    rows=bars()
    for r in rows[-2:]:r.update(low=90.,close=91.)
    now=datetime.fromisoformat(rows[-1]['bar_start_utc'])+timedelta(minutes=1)
    # Missing prefix is not quietly interpreted as the full opening range.
    assert not analyse(rows[2:],now)['opening_range_complete']
    # A missing minute cannot stand in for two consecutive minutes.
    rows.pop(-2)
    assert analyse(rows,now)['signal']!='REVIEW_EXIT'
    assert analyse(rows,datetime(2026,10,26,16,tzinfo=IST))['status']=='MARKET_CLOSED'


def test_zero_volume_does_not_create_vwap_or_alerts():
    rows=bars()
    for r in rows:r['volume']=0.
    now=datetime.fromisoformat(rows[-1]['bar_start_utc'])+timedelta(minutes=1)
    r=analyse(rows,now)
    assert r['bar_based_vwap'] is None and not r['alert_eligible']


def test_saturday_keeps_friday_candles_without_active_alerts(tmp_path,monkeypatch):
    import services.module2_intraday as service
    monkeypatch.setattr(service,'ARCHIVES',tmp_path/'archive')
    rows=bars()
    for row in rows:
        row['bar_start_utc']=datetime.fromisoformat(row['bar_start_utc']).replace(day=30).isoformat()
        row['session_date']='2026-10-30'
    class Response:
        content=json.dumps(payload(rows)).encode();url='https://query1.finance.yahoo.com/test'
        def raise_for_status(self):pass
        def json(self):return json.loads(self.content)
    path=tmp_path/'intraday.db'
    friday=datetime(2026,10,30,9,33,tzinfo=IST)
    capture(dict(ipo_id=1,company_name='Test',symbol='TEST.NS'),now=friday,db_path=path,fetch=lambda *a,**k:Response())
    result=read(1,now=datetime(2026,10,31,11,tzinfo=IST),db_path=path)
    assert result['status']=='MARKET_CLOSED' and result['chart_session_date']=='2026-10-30'
    assert result['chart_is_historical'] and len(result['candles'])==len(rows)
    assert not result['alert_eligible']


def test_revision_history_retained_alert_dedup_and_failed_refresh_preserves_data(tmp_path,monkeypatch):
    import services.module2_intraday as service
    monkeypatch.setattr(service,'ARCHIVES',tmp_path/'archive')
    rows=bars()
    for r in rows[-2:]:r.update(low=90.,close=91.)
    now=datetime.fromisoformat(rows[-1]['bar_start_utc'])+timedelta(minutes=1)
    class Response:
        url='https://example.test/saved-minute-test'
        def __init__(self):self.content=json.dumps(payload(rows)).encode()
        def json(self):return json.loads(self.content)
        def raise_for_status(self):pass
    def fetch(*a,**kw):return Response()
    issuer=dict(ipo_id=1,company_name='Test',symbol='TEST.NS')
    db=tmp_path/'observations.db'
    capture(issuer,now=now,db_path=db,fetch=fetch)
    capture(issuer,now=now+timedelta(seconds=1),db_path=db,fetch=fetch)
    rows[-1]['volume']+=1
    capture(issuer,now=now+timedelta(seconds=2),db_path=db,fetch=fetch)
    with sqlite3.connect(db) as con:
        assert con.execute('select count(*) from candles_v1').fetchone()[0]==18
        assert con.execute('select count(*) from alerts_v1').fetchone()[0]==1
        original=con.execute('select payload_json from observations_v1 where id=1').fetchone()[0]
    def failed(*a,**kw):raise TimeoutError('offline')
    capture(issuer,now=now+timedelta(seconds=3),db_path=db,fetch=failed)
    report=read(1,now=now+timedelta(seconds=3),db_path=db)
    assert report['status']=='SOURCE_UNAVAILABLE' and not report['alert_eligible']
    assert report['last_successful_observation']['latest_price']==91.
    assert len(report['candles'])==17
    assert report['candles'][-1]['volume']==101.
    with sqlite3.connect(db) as con:
        assert con.execute('select payload_json from observations_v1 where id=1').fetchone()[0]==original
    # Old success cannot remain marked fresh just because its HTTP request succeeded.
    assert not read(1,now=now+timedelta(minutes=10),db_path=db)['alert_eligible']


def test_listing_day_opening_range_excludes_preauction_entries():
    rows=bars(17)
    for r in rows:
        r['bar_start_utc']=(datetime.fromisoformat(r['bar_start_utc'])+timedelta(minutes=45)).isoformat()
    pre=dict(rows[0],bar_start_utc=datetime(2026,10,26,9,22,tzinfo=IST).astimezone(timezone.utc).isoformat(),
             open=80.,high=80.,low=80.,close=80.,volume=0.)
    now=datetime.fromisoformat(rows[-1]['bar_start_utc'])+timedelta(minutes=1)
    report=analyse([pre]+rows,now,listing_date='2026-10-26')
    assert report['opening_range_complete']
    assert report['session_low']==99. and report['session_open_price']==100.
    assert not analyse([pre]+rows[1:],now,listing_date='2026-10-26')['opening_range_complete']


def test_bad_ohlc_and_duplicate_minutes_are_rejected():
    rows=bars(2);rows[0]['high']=80.
    now=datetime(2026,10,26,10,tzinfo=IST)
    with pytest.raises(ValueError,match='Invalid candle'):
        parse_candles(payload(rows),'TEST.NS',now)
    rows=bars(2);rows[1]['bar_start_utc']=rows[0]['bar_start_utc']
    with pytest.raises(ValueError,match='duplicate'):
        parse_candles(payload(rows),'TEST.NS',now)


def test_unbound_calendar_ticker_requires_matching_provider_company(tmp_path,monkeypatch):
    import services.module2_intraday as service
    monkeypatch.setattr(service,'ARCHIVES',tmp_path/'archive')
    rows=bars()
    class Response:
        url='https://example.test/identity'
        content=json.dumps(payload(rows)).encode()
        def json(self):return json.loads(self.content)
        def raise_for_status(self):pass
    issuer=dict(ipo_id=9,company_name='A completely different issuer',symbol='TEST.NS',
        require_provider_name_match=True)
    now=datetime(2026,10,26,10,tzinfo=IST)
    report=capture(issuer,now=now,db_path=tmp_path/'ledger.db',fetch=lambda *a,**k:Response())
    assert report['status']=='SOURCE_UNAVAILABLE' and not report['alert_eligible']
    with sqlite3.connect(tmp_path/'ledger.db') as con:
        assert con.execute('select count(*) from candles_v1').fetchone()[0]==0
