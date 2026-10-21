import json
import sqlite3
import hashlib
import gzip
from datetime import datetime, timedelta

from services.data_maintenance import DataMaintenance, claim, maintain_offer
from services.point_in_time import IST
from services.preclose_context import attach_offer_context, archive_offer_context


URL='https://www.chittorgarh.com/ipo/automation-example/987650/'
NOW=datetime(2026,10,28,11,tzinfo=IST)
ITEM=dict(company_name='Automation Example',issue_price=100,category='OPEN',source_observed=True)
HTML='''<title>Automation Example IPO</title><table>
<tr><td>Issue Price</td><td>₹100</td></tr><tr><td>Lot Size</td><td>150 Shares</td></tr>
<tr><td>Total Issue Size</td><td>100 crore</td></tr><tr><td>Fresh Issue</td><td>100 crore</td></tr>
<tr><td>Retail Portion</td><td>35%</td></tr></table>
<table id="ObjectiveIssue"><tr><th>#</th><th>Issue Objects</th><th>Amt (Cr)</th></tr>
<tr><td>1</td><td>Repayment of borrowings</td><td>90</td></tr>
<tr><td>2</td><td>Issue Expenses</td><td>10</td></tr><tr><td></td><td>Total</td><td>100</td></tr></table>'''


class Response:
    def __init__(self,html=HTML):self.content=html.encode()
    def raise_for_status(self):pass


def test_new_offer_is_fetched_archived_and_reused_after_worker_restart(tmp_path):
    db=tmp_path/'test.db';calls=[]
    def fetch(*args,**kwargs):calls.append(args[0]);return Response()
    first=DataMaintenance(db).run_once(cache={URL:ITEM},now=NOW,fetch=fetch)
    assert first['records'][0]['state']=='ready'
    data=attach_offer_context(dict(ITEM,url=URL),db)
    assert data['promoter_ofs_ratio']==0 and data['retail_quota_pct']==35
    assert data['original_lot_size']==150 and data['objects_capex_ratio']==0
    assert DataMaintenance(db).run_once(cache={URL:ITEM},now=NOW+timedelta(minutes=1),fetch=fetch)['offers_checked']==0
    assert calls==[URL]
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(*) FROM offer_context_archive_v1').fetchone()[0]==1
        assert c.execute('SELECT COUNT(*) FROM data_maintenance_events_v1').fetchone()[0]==1


def test_parser_upgrade_uses_preserved_source_without_fetching_or_changing_raw_archive(tmp_path):
    db=tmp_path/'test.db';item=dict(ITEM,url=URL)
    original=archive_offer_context(item,HTML,URL,db)
    old=dict(original,objects_capex_ratio=None,parser_version='old-parser')
    with sqlite3.connect(db) as c:
        c.execute('UPDATE offer_context_archive_v1 SET facts_json=?',(json.dumps(old),))
        c.execute('DELETE FROM offer_context_parse_v1')
        raw=c.execute('SELECT raw_html_gzip FROM offer_context_archive_v1').fetchone()[0]
    result=maintain_offer(URL,item,now=NOW,fetch=lambda *a,**k:(_ for _ in ()).throw(AssertionError('already have source')),db_path=db)
    assert result['state']=='ready'
    restored=attach_offer_context(item,db)
    assert restored['objects_capex_ratio']==0
    assert restored['offer_facts']['fetch_timestamp']==original['fetch_timestamp']
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT raw_html_gzip FROM offer_context_archive_v1').fetchone()[0]==raw
        assert c.execute('SELECT COUNT(*) FROM offer_context_archive_v1').fetchone()[0]==1


def test_precise_saved_offer_amounts_survive_newer_rounded_source(tmp_path):
    db=tmp_path/'test.db';item=dict(ITEM,url=URL,issue_size_cr=100.)
    raw=HTML.replace('Repayment of borrowings','Capital expenditure for new machinery')
    r=archive_offer_context(item,raw,URL,db)
    old_raw=b'{"retained_precise_offer":100.25}'
    old=dict(r,sha256=hashlib.sha256(old_raw).hexdigest(),objects_capex_ratio=None,objects_capex_cr=None,
             offer=dict(ofs_ratio=0.,fresh_issue_cr=100.25,total_issue_cr=100.25),source_url='https://www.ipopremium.in/saved-example')
    with sqlite3.connect(db) as c:
        c.execute('INSERT INTO offer_context_archive_v1(company_name,source_url,fetched_at_ist,source_sha256,raw_html_gzip,facts_json) VALUES(?,?,?,?,?,?)',
                  (ITEM['company_name'],old['source_url'],old['fetch_timestamp'],old['sha256'],gzip.compress(old_raw),json.dumps(old)))
    result=attach_offer_context(item,db)
    assert result['fresh_issue_cr']==100.25
    assert result['objects_capex_ratio']==90/100.25


def test_outage_has_durable_backoff_and_does_not_turn_missing_values_into_zero(tmp_path):
    db=tmp_path/'test.db';calls=[]
    def fail(*a,**k):calls.append(1);raise TimeoutError('upstream offline')
    assert maintain_offer(URL,ITEM,now=NOW,fetch=fail,db_path=db)['state']=='retrying'
    assert maintain_offer(URL,ITEM,now=NOW+timedelta(minutes=1),fetch=fail,db_path=db) is None
    assert maintain_offer(URL,ITEM,now=NOW+timedelta(minutes=5),fetch=fail,db_path=db)['state']=='retrying'
    with sqlite3.connect(db) as c:
        attempts,due=c.execute('SELECT attempts,next_attempt_at_ist FROM data_maintenance_jobs_v1').fetchone()
    assert attempts==2 and due==(NOW+timedelta(minutes=15)).isoformat() and len(calls)==2
    assert attach_offer_context(dict(ITEM,url=URL),db).get('retail_quota_pct') is None


def test_one_failed_source_does_not_block_another_issuer(tmp_path):
    db=tmp_path/'test.db';bad=URL.replace('987650','987651')
    def fetch(url,**kw):
        if url==bad:raise ConnectionError('offline')
        return Response()
    result=DataMaintenance(db).run_once(cache={bad:{**ITEM,'company_name':'Broken Example'},URL:ITEM},now=NOW,fetch=fetch)
    assert sorted(r['state'] for r in result['records'])==['ready','retrying']


def test_lease_survives_restart_and_expires_after_interrupted_work(tmp_path):
    db=tmp_path/'test.db'
    assert claim('job','offer_facts','Example',NOW,db)
    assert not claim('job','offer_facts','Example',NOW+timedelta(minutes=1),db)
    assert claim('job','offer_facts','Example',NOW+timedelta(minutes=5),db)


def test_missing_offer_terms_wait_without_fetch_or_fabricated_defaults(tmp_path):
    result=maintain_offer(URL,dict(ITEM,issue_price=None),now=NOW,
        fetch=lambda *a,**k:(_ for _ in ()).throw(AssertionError('must wait for published terms')),db_path=tmp_path/'test.db')
    assert result['state']=='awaiting_terms'


def test_wrong_company_source_is_rejected_and_scheduled_for_retry(tmp_path):
    db=tmp_path/'test.db'
    result=maintain_offer(URL,ITEM,now=NOW,fetch=lambda *a,**k:Response(HTML.replace('Automation Example','Different Company')),db_path=db)
    assert result['state']=='retrying' and 'identity' in result['error']
    assert attach_offer_context(dict(ITEM,url=URL),db).get('offer_facts') is None


def test_unspecified_use_of_proceeds_remains_unknown_and_retries_without_zero_fill(tmp_path):
    db=tmp_path/'test.db'
    html=HTML.replace('Repayment of borrowings','Investment in a subsidiary')
    result=maintain_offer(URL,ITEM,now=NOW,fetch=lambda *a,**k:Response(html),db_path=db)
    assert result['state']=='partial' and result['missing_fields']==['objects_capex_ratio']
    assert attach_offer_context(dict(ITEM,url=URL),db).get('objects_capex_ratio') is None


def test_rejected_source_rotates_to_alternative_and_survives_restart(tmp_path):
    db=tmp_path/'test.db';calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        return Response(HTML.replace('Automation Example','Wrong Issuer') if url==URL else HTML)
    first=maintain_offer(URL,ITEM,now=NOW,fetch=fetch,db_path=db)
    assert first['state']=='retrying'
    second=maintain_offer(URL,ITEM,now=NOW+timedelta(minutes=1),fetch=fetch,db_path=db)
    assert second['state']=='ready'
    assert calls==[URL,'https://www.ipoji.com/ipo/automation-example-ipo']
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(*) FROM data_source_rejection_v1').fetchone()[0]==1
        assert c.execute("SELECT state FROM offer_source_state_v1 WHERE source_url=?",(URL,)).fetchone()[0]=='rejected'
    assert attach_offer_context(dict(ITEM,url=URL),db)['offer_facts']['source_url']==calls[-1]


def test_recovered_objectives_stop_hourly_retries_without_inventing_capex(tmp_path):
    db=tmp_path/'test.db';calls=[]
    html=HTML.replace('Repayment of borrowings','Investment in a subsidiary')
    def fetch(url,**kwargs):calls.append(url);return Response(html+f'<!-- source: {url} -->')
    for m in (0,1):
        result=maintain_offer(URL,ITEM,now=NOW+timedelta(minutes=m),fetch=fetch,db_path=db)
    assert result['state']=='ready_with_limitations'
    assert len(calls)==2 and len(set(calls))==2
    assert result['missing_fields']==['objects_capex_ratio']
    assert result['objects_capex_status']=='NOT_SEPARATELY_ITEMISED'
    assert result['proceeds_uses']
    assert maintain_offer(URL,ITEM,now=NOW+timedelta(hours=1),fetch=fetch,db_path=db) is None
    assert attach_offer_context(dict(ITEM,url=URL),db).get('objects_capex_ratio') is None
    # On the next daily check, reuse the verified allocation instead of fetching
    # every source again merely because no standalone CAPEX ratio was published.
    again=maintain_offer(URL,ITEM,now=NOW+timedelta(days=2),
        fetch=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('verified allocation already cached')),db_path=db)
    assert again['state']=='ready_with_limitations' and len(calls)==2


def test_all_unusable_sources_pause_for_document_review_without_hourly_refetch(tmp_path):
    db=tmp_path/'test.db';calls=[]
    def fetch(url,**kwargs):calls.append(url);return Response(HTML.replace('Automation Example','Unrelated Issuer'))
    for m in (0,1,2,3):
        result=maintain_offer(URL,ITEM,now=NOW+timedelta(minutes=m),fetch=fetch,db_path=db)
    assert result['state']=='needs_document_review' and len(calls)==3
    assert maintain_offer(URL,ITEM,now=NOW+timedelta(hours=1),fetch=fetch,db_path=db) is None


def test_ipoji_definition_list_offer_parser_and_price_guard(tmp_path):
    from services.live_offer_facts import parse_offer
    html='''<title>Automation Example IPO</title><dl>
    <dt>Issue price</dt><dd>₹95-100 per share</dd><dt>Lot size</dt><dd>150 shares</dd>
    <dt>Total issue size</dt><dd>₹100 crore</dd><dt>Fresh issue</dt><dd>₹80 crore</dd>
    <dt>Offer for sale (OFS)</dt><dd>₹20 crore</dd><dt>Retail Portion</dt><dd>35%</dd>
    <dd id="issue-objective-0">Investing in technology infrastructure</dd></dl>'''
    parsed=parse_offer(html,100)
    assert parsed['lot_size_shares']==150 and parsed['ofs_ratio']==.2
    assert parsed['objects_capex_status']=='NOT_SEPARATELY_ITEMISED'
    assert parsed.get('objects_capex_ratio') is None
    import pytest
    with pytest.raises(ValueError,match='cap price'):
        archive_offer_context(ITEM,html.replace('95-100','95-101'),URL,tmp_path/'test.db')


def test_decision_failure_does_not_abort_other_ipo_captures(tmp_path,monkeypatch):
    from services import snapshot_service as snapshots
    from scrapers import live_ipo_scraper as live
    db=tmp_path/'test.db';snapshots.init_db(str(db))
    cache={URL:{**ITEM,'date_range':'28 - 30 Oct'},URL.replace('987650','987651'):{**ITEM,'company_name':'Second Example','date_range':'28 - 30 Oct'}}
    evidence=dict(status='verified',fetched_at_ist=NOW.isoformat(),raw_html='<table>captured</table>',values=dict(qib_subscription=1.,hni_subscription=2.,retail_subscription=3.))
    monkeypatch.setattr(live,'fetch_subscription_evidence',lambda u:evidence)
    monkeypatch.setattr(snapshots,'record_automatic_decision',lambda **k:(_ for _ in ()).throw(RuntimeError('decision downstream failed')))
    assert snapshots.capture_active_subscriptions(now_ist=NOW,cache=cache,db_path=str(db),capture_exchange=False)==2
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(DISTINCT ipo_id) FROM subscription_capture_v2').fetchone()[0]==2


def test_null_nse_publication_clock_preserves_raw_evidence_for_next_poll(monkeypatch):
    from services import nse_subscription_service as service
    class R:
        def __init__(self,payload):self.payload=payload;self.text=json.dumps(payload);self.status_code=200
        def json(self):return self.payload
        def raise_for_status(self):pass
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url):
            if 'ipo-current-issue' in url:return R([dict(companyName='Automation Example',issueEndDate='28-Oct-2026',symbol='AUTO')])
            return R(dict(symbol='AUTO',updateTime='Updated as on null',dataList=[]))
    monkeypatch.setattr(service.httpx,'Client',Client)
    result=service.fetch_nse_subscription_evidence('Automation Example',NOW.date())
    assert result['status']=='awaiting_publication' and result['values'] is None
    assert result['raw_sha256'] and result['raw_html'] and result['error'] is None
