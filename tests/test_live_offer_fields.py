import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from services.live_offer_facts import parse_offer,total_book


def test_definition_list_fresh_ofs_and_independent_retail_quota():
    html='''<dl><dt>Total issue size</dt><dd>₹100 crore</dd><dt>Fresh issue</dt><dd>₹25 crore</dd>
    <dt>Offer for sale (OFS)</dt><dd>₹75 crore</dd><dt>Retail Portion</dt><dd>35%</dd></dl>
    <script id="ipo-reservation-chart-data">{"anchor":30,"qib":20,"nii":15,"retail":35}</script>'''
    r=parse_offer(html,100)
    assert r['ofs_ratio']==.75 and r['retail_quota_pct']==35 and r['retail_quota_exact']


def test_blank_general_corporate_amount_does_not_hide_explicit_capex():
    html='''<dl><dt>Total issue size</dt><dd>100 crore</dd><dt>Fresh issue</dt><dd>100 crore</dd></dl>
    <table id="ObjectiveIssue"><tr><th>#</th><th>Issue Objects</th><th>Est Amt (Cr)</th></tr>
    <tr><td>1</td><td>Capital expenditure for manufacturing capacity</td><td>30</td></tr>
    <tr><td>2</td><td>General Corporate Purposes</td><td></td></tr>
    <tr><td></td><td>Total</td><td>30</td></tr></table>'''
    result=parse_offer(html,100)
    assert result['objects_capex_cr']==30 and result['objects_capex_ratio']==.3


@pytest.mark.parametrize('label,amount,expected',[('Fresh issue',100,0),('Offer for sale (OFS)',100,1)])
def test_full_fresh_and_full_ofs_are_derived_from_reported_total(label,amount,expected):
    r=parse_offer(f'<dl><dt>Total issue size</dt><dd>₹100 crore</dd><dt>{label}</dt><dd>₹{amount} crore</dd></dl>',100)
    assert r['ofs_ratio']==expected
    assert r['fresh_issue_cr']+r['ofs_issue_cr']==100


def test_missing_ofs_is_not_assumed_zero_and_null_chart_does_not_drop_retail():
    r=parse_offer('<dl><dt>Retail Portion</dt><dd>35%</dd></dl><script id="ipo-reservation-chart-data">null</script>',10)
    assert r['retail_quota_pct']==35 and 'ofs_ratio' not in r
    assert 'retail_quota_exact' not in parse_offer('<table><tr><td>Retail</td><td>At least 35%</td></tr></table>',10)
    with pytest.raises(ValueError,match='cap price'):
        parse_offer('<dl><dt>Issue price</dt><dd>₹90–100 per share</dd></dl>',200)


def test_conflicting_offer_components_and_quota_are_rejected():
    with pytest.raises(ValueError,match='components'):
        parse_offer('<dl><dt>Total issue size</dt><dd>100 crore</dd><dt>Fresh issue</dt><dd>90 crore</dd><dt>OFS</dt><dd>50 crore</dd></dl>',10)
    with pytest.raises(ValueError,match='reservations'):
        parse_offer('<dl><dt>Retail Portion</dt><dd>35%</dd></dl><script id="ipo-reservation-chart-data">{"retail":10,"qib":75,"nii":15}</script>',10)


def test_nse_total_uses_shares_not_an_average_or_duplicate_child_rows():
    raw=json.dumps({'updateTime':'Updated as on 26-Oct-2026 16:51:00','dataList':[
        {'category':'QIB','noOfShareOffered':'10','noOfSharesBid':'100'},
        {'category':'QIB child','noOfShareOffered':'','noOfSharesBid':'100'},
        {'category':'Total','noOfShareOffered':'100','noOfSharesBid':'200','noOfTotalMeant':'2'}]})
    result=total_book(raw,'nse_exchange')
    assert result['multiple']==2 and result['bid_shares']==200
    assert total_book(raw,'unverified_family') is None
    assert total_book('{"dataList":[]}','nse_exchange') is None
    with pytest.raises(ValueError,match='conflicts'):
        total_book(raw.replace('"2"','"7"'),'nse_exchange')
    html='<table><tr><th>Category</th><th>Subscription (Times)</th></tr><tr><td>Total</td><td>0.0x</td></tr></table>'
    assert total_book(html,'chittorgarh_investorgain')['multiple']==0.
    assert total_book(html.replace('0.0x','Not published'),'chittorgarh_investorgain') is None


def test_real_active_issuer_archives_all_have_complete_static_fields():
    expected={'2061':1.,'2070':517.53/667.53,'2559':0.,'2671':0.,'3100':.2}
    for ipo_id,ratio in expected.items():
        path=Path('data/telemetry/prospective_financials')/f'chitto_{ipo_id}_20261026.html.gz'
        if not path.exists():pytest.skip('Optional dated local source archive absent')
        r=parse_offer(gzip.decompress(path.read_bytes()).decode('utf8'),None)
        assert r['ofs_ratio']==pytest.approx(ratio)
        assert r['retail_quota_pct']==35


def test_live_total_is_separate_context_and_cannot_change_frozen_score():
    from services.prospective_module1 import predict
    features={'gmp_pct':10.,'log_qib':2.,'log_hni':1.,'log_retail':1.}
    original=predict(features)
    assert predict({**features,'context_total_mult':1000.})==original
