import hashlib
import json
import sqlite3

from services.preclose_context import restore_preclose_gmp, archive_offer_context, attach_offer_context


def test_refresh_cannot_hide_saved_closing_day_gmp_or_change_issued_call(tmp_path):
    db=tmp_path/'snapshots.db'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE automatic_decision_v1(id INTEGER PRIMARY KEY,ipo_id TEXT,company_name TEXT,input_json TEXT,recorded_at_ist TEXT)')
        for i,observed,recorded,value in [(1,'16:40','16:41',21.45),(2,'16:55','16:56',0),(3,'17:05','17:06',40)]:
            inputs=dict(gmp_pct=value,gmp_observed_at=f'2026-10-23T{observed}:00+05:30',application_cutoff_at_ist='2026-10-23T17:00:00+05:30',probability=.69)
            c.execute('INSERT INTO automatic_decision_v1 VALUES(?,?,?,?,?)',(i,'chitto_3152','Jio Platforms',json.dumps(inputs),f'2026-10-23T{recorded}:00+05:30'))
    before=hashlib.sha256(db.read_bytes()).hexdigest()
    item=dict(company_name='Jio Platforms',url='https://www.chittorgarh.com/ipo/jio-platforms-ipo/3152/',category='LISTED',listing_date='2026-10-28',gmp_pct=22.34,gmp_observed_at='2026-10-28T14:00:00+05:30')
    restored=restore_preclose_gmp(item,db)
    assert restored['gmp_pct']==0
    assert restored['gmp_observed_at']=='2026-10-23T16:55:00+05:30'
    assert restored['gmp_preclose_provenance']['row_id']==2
    assert restored['later_gmp_context']['gmp_pct']==22.34
    assert item['gmp_pct']==22.34
    assert hashlib.sha256(db.read_bytes()).hexdigest()==before
    assert restore_preclose_gmp(dict(item,category='OPEN'),db)==dict(item,category='OPEN')


def test_offer_page_identity_cap_price_and_proceeds_are_preserved_for_listed_view(tmp_path):
    db=tmp_path/'snapshots.db'
    item=dict(company_name='Example',issue_price=1119,url='https://www.chittorgarh.com/ipo/example/1/',category='LISTED')
    raw='''<title>Example IPO Details</title><table>
    <tr><td>Price Band</td><td>₹1065 to ₹1119</td></tr><tr><td>Issue Price</td><td>₹1119</td></tr>
    <tr><td>Total Issue Size</td><td>100 crore</td></tr><tr><td>Fresh Issue</td><td>100 crore</td></tr></table>
    <table><tr><th>Investor Category</th><th>Shares</th><th>% of Net Issue</th><th>% of Total Issue</th></tr>
    <tr><td>Retail</td><td>100</td><td>35%</td><td>31.5%</td></tr></table>
    <table id="ObjectiveIssue"><tr><th>#</th><th>Issue Objects</th><th>Amt (Cr)</th></tr>
    <tr><td>1</td><td>Repayment of borrowings</td><td>90</td></tr><tr><td>2</td><td>Issue Expenses</td><td>10</td></tr>
    <tr><td></td><td>Total</td><td>100</td></tr></table>'''
    archive_offer_context(item,raw,'https://www.chittorgarh.com/ipo/example/1/',db)
    result=attach_offer_context(item,db)
    assert result['promoter_ofs_ratio']==0
    assert result['retail_quota_pct']==35 and result['retail_quota_total_pct']==31.5
    assert result['objects_capex_ratio']==0
    assert result['offer_facts']['evidence_class']=='recovered_static_offer_terms'
    assert attach_offer_context(dict(item,issue_price=2222),db).get('offer_facts') is None
