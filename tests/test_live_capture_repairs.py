import gzip,hashlib,json,sqlite3
from datetime import datetime,date,timedelta
from zoneinfo import ZoneInfo
from unittest.mock import Mock
import pytest
from services import snapshot_service as snapshots,nse_subscription_service as nse
from scrapers import live_ipo_scraper as live
from services.subscription_training import training_captures
from services import live_valuation_recovery as valuations

IST=ZoneInfo('Asia/Kolkata')


def add(c,at,qib,*,status='verified',bad_hash=False):
    raw=json.dumps(dict(updateTime='Updated as on 27-Oct-2026 '+at,symbol='EXAMPLE')).encode()
    stamp='2026-10-27T'+at+'+05:30'
    c.execute('''INSERT INTO subscription_capture_v2
        (ipo_id,company_name,ipo_url,attempted_at_ist,fetched_at_ist,slot_key,closing_date,
         snapshot_stage,status,qib_x,nii_x,retail_x,raw_sha256,raw_html_gzip,source_family,capture_phase)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        ('test','Example','url',stamp,stamp,'2026-10-27_1600','2026-10-29','HOURLY',status,qib,2.,0.,
         '0'*64 if bad_hash else hashlib.sha256(raw).hexdigest(),gzip.compress(raw),'nse_exchange','pre_deadline'))


def test_training_dedup_is_causal_hash_checked_and_keeps_raw(tmp_path):
    db=tmp_path/'captures.db';snapshots.init_db(str(db))
    with sqlite3.connect(db) as c:
        add(c,'16:00:00',1.);add(c,'16:04:00',2.);add(c,'16:04:30',3.,bad_hash=True)
    old=training_captures('2026-10-27',asof='2026-10-27T16:02:00+05:30',db_path=db)
    latest=training_captures('2026-10-27',db_path=db)
    assert len(old['records'])==len(latest['records'])==1
    assert old['records'][0]['qib_x']==1. and latest['records'][0]['qib_x']==2.
    assert latest['records'][0]['retail_x']==0. # Observed zero is valid.
    with sqlite3.connect(db) as c:assert c.execute('SELECT count(*) FROM subscription_capture_v2').fetchone()[0]==3
    assert len(latest['rejected'])==1


def test_ampersand_name_normalisation_does_not_fuzz_other_issuers():
    assert nse._name('Bombay Coated & Special Steels')==nse._name('Bombay Coated and Special Steels Limited')
    assert nse._name('Bombay Coated & Special Steels')!=nse._name('Bombay Coated Engineering Limited')


def test_nse_match_accepts_equivalent_name_but_still_requires_exact_closing_date(monkeypatch):
    listing=[dict(companyName='Bombay Coated and Special Steels Limited',issueEndDate='02-Nov-2026',symbol='BCSSL',series='EQ')]
    payload=dict(symbol='BCSSL',updateTime=datetime.now(IST).strftime('Updated as on %d-%b-%Y %H:%M:%S'),dataList=[
        dict(srNo='1',category='Qualified Institutional Buyers',noOfTotalMeant='1.2'),
        dict(srNo='2',category='Non Institutional Investors',noOfTotalMeant='0.7'),
        dict(srNo='3',category='Retail Individual Investors',noOfTotalMeant='1.1')])
    def response(data):
        r=Mock(status_code=200,text=json.dumps(data));r.json.return_value=data;return r
    client=Mock();client.get.side_effect=lambda url:response(listing if url.endswith('/ipo-current-issue') else payload)
    context=Mock();context.__enter__=Mock(return_value=client);context.__exit__=Mock(return_value=False)
    monkeypatch.setattr(nse.httpx,'Client',lambda **kw:context)
    good=nse.fetch_nse_subscription_evidence('Bombay Coated & Special Steels',date(2026,11,2))
    wrong=nse.fetch_nse_subscription_evidence('Bombay Coated & Special Steels',date(2026,11,3))
    assert good['status']=='verified' and good['nse_symbol']=='BCSSL'
    assert wrong['status']=='identity_unavailable'


def test_partial_slot_restart_only_fetches_unfinished_source(tmp_path,monkeypatch):
    db=str(tmp_path/'captures.db');snapshots.init_db(db)
    clock=datetime(2026,10,27,16,0,tzinfo=IST)
    raw=b'<p>published source</p>';values=dict(qib_subscription=1.,hni_subscription=2.,retail_subscription=3.)
    evidence=dict(status='verified',raw_html=raw.decode(),raw_sha256=hashlib.sha256(raw).hexdigest(),values=values,
                  fetched_at_ist=clock.isoformat(),source_displayed_at_ist=clock.isoformat())
    agg=Mock(return_value=evidence);exchange=Mock(return_value=evidence)
    monkeypatch.setattr(live,'fetch_subscription_evidence',agg);monkeypatch.setattr(nse,'fetch_nse_subscription_evidence',exchange)
    monkeypatch.setattr(live,'save_scraped_cache',lambda x:None)
    monkeypatch.setattr(snapshots,'_record_capture_decision',lambda **kw:None)
    cache={'https://www.chittorgarh.com/ipo/example/123/':dict(company_name='Example',category='OPEN',source_observed=True,date_range='27 - 29 Oct')}
    assert snapshots.capture_active_subscriptions(now_ist=clock,cache=cache,db_path=db,capture_exchange=False)==1
    assert snapshots.capture_active_subscriptions(now_ist=clock,cache=cache,db_path=db,capture_exchange=True)==1
    assert snapshots.capture_active_subscriptions(now_ist=clock,cache=cache,db_path=db,capture_exchange=True)==0
    assert agg.call_count==exchange.call_count==1
    with sqlite3.connect(db) as c:assert c.execute('SELECT count(*) FROM subscription_capture_v2').fetchone()[0]==2


HTML='''<h1>SFC Environmental Technologies Limited</h1><table>
 <tr><td>IPO Date</td><td>October 27, 2026 to October 29, 2026</td></tr>
 <tr><td>Price Band</td><td>Rs. 313 to Rs. 330 per share</td></tr></table>
 <table><tr><th>Key Performance Indicator</th><th>Units</th><th>FY 2026</th><th>FY 2025</th></tr>
 <tr><td>Profit After Tax (PAT)</td><td>Cr.</td><td>160.21</td><td>152.01</td></tr>
 <tr><td>Total Income</td><td>Cr.</td><td>821.27</td><td>726.96</td></tr>
 <tr><td>Diluted Earnings per Share</td><td>Rs.</td><td>16.52</td><td>15.42</td></tr>
 <tr><td>Net Asset Value per Share</td><td>Rs.</td><td>86.09</td><td>69.44</td></tr></table>'''
BASE=dict(annual_period='2026-03-31',annual_profit_after_tax=160.21,annual_revenue=821.27)
ITEM=dict(company_name='SFC Environmental Technologies',issue_price=330.)


def test_valuation_requires_matching_issuer_period_cap_dates_and_accounts():
    f=valuations.extract(HTML,ITEM,BASE,date(2026,10,27),date(2026,10,29))
    assert f['ipo_pre_offer_eps']==16.52 and f['ipo_pre_offer_nav']==86.09
    for html,item,base,closed in [(HTML,dict(ITEM,issue_price=331),BASE,date(2026,10,29)),
        (HTML.replace('FY 2026','Quarter 2026'),ITEM,BASE,date(2026,10,29)),
        (HTML,ITEM,dict(BASE,annual_profit_after_tax=999),date(2026,10,29)),
        (HTML,ITEM,BASE,date(2026,10,30))]:
        with pytest.raises(ValueError):valuations.extract(html,item,base,date(2026,10,27),closed)


def test_fetched_valuation_is_not_available_before_its_capture(tmp_path,monkeypatch):
    fixed=datetime(2026,10,29,19,0,tzinfo=IST)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return fixed
    monkeypatch.setattr(valuations,'datetime',Clock)
    response=Mock(content=HTML.encode(),text=HTML);response.raise_for_status.return_value=None
    get=Mock(return_value=response);monkeypatch.setattr(valuations.requests,'get',get)
    import services.sqlite_supabase_sync as sync
    monkeypatch.setattr(sync,'request_sync',lambda reason:None)
    db=tmp_path/'valuation.db'
    args=(ITEM,BASE,date(2026,10,27),date(2026,10,29))
    assert valuations.recover(*args,fixed-timedelta(seconds=1),db_path=db) is None
    f=valuations.recover(*args,fixed+timedelta(seconds=1),db_path=db)
    assert f['ipo_pre_offer_eps']==16.52
    assert valuations.recover(*args,fixed-timedelta(hours=1),db_path=db) is None
    assert get.call_count==1


def test_failed_valuation_source_is_not_retried_every_analysis(tmp_path,monkeypatch):
    fixed=datetime(2026,10,29,19,0,tzinfo=IST)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return fixed
    monkeypatch.setattr(valuations,'datetime',Clock)
    import requests
    get=Mock(side_effect=requests.HTTPError('503 unavailable'));monkeypatch.setattr(valuations.requests,'get',get)
    import services.sqlite_supabase_sync as sync
    monkeypatch.setattr(sync,'request_sync',lambda reason:None)
    args=(ITEM,BASE,date(2026,10,27),date(2026,10,29),fixed)
    for _ in range(3):assert valuations.recover(*args,db_path=tmp_path/'failed.db') is None
    assert get.call_count==1
