from datetime import datetime
from zoneinfo import ZoneInfo
import json,sqlite3
import pytest
from services.retail_hni_economics import institutional_bouncer,category_lots
from services.filing_governance import filing_governance
from services.module1_milestones import capture,milestone

IST=ZoneInfo('Asia/Kolkata')

def inputs():
    return dict(p_win=.8,gmp_pct=15,qib=5,nii=3,retail=3,ofs_ratio=.2,retail_quota_pct=35,total_mult=4,original_lot_size=50,offer_cap_price=300,bnii_mult=25)

def test_minimum_category_strictly_exceeds_rupee_threshold():
    sizes=category_lots(1000,20)
    assert sizes['snii_min_lots']==11 and sizes['bnii_min_lots']==51

def test_layer2_never_changes_layer1_and_flags_are_separate():
    args=inputs();a=institutional_bouncer(**args)
    assert a['tier']==1 and a['standard_binary_call']=='POSITIVE'
    b=institutional_bouncer(**{**args,'nii':.89})
    assert b['verdict']=='HARD_SKIP' and b['standard_binary_call']=='POSITIVE'
    c=institutional_bouncer(**{**args,'ofs_ratio':1.,'gmp_pct':5.,'qib':4.})
    assert c['flags']['high_ofs_weak_gmp'] is True
    assert not a['flags']['retail_heavy_quota_weak_demand']
    assert institutional_bouncer(**{**args,'ofs_ratio':None})['verdict']=='DATA_PENDING'

def test_funding_cost_uses_borrowed_amount_even_when_not_allotted():
    a=institutional_bouncer(**inputs());b=institutional_bouncer(**{**inputs(),'bnii_mult':50})
    assert a['nbfc_interest_rupees']==b['nbfc_interest_rupees']
    assert b['nbfc_breakeven_open_pct']>a['nbfc_breakeven_open_pct']
    expected=a['bnii_app_rupees']*.095*5/365
    assert a['nbfc_interest_rupees']==pytest.approx(expected)
    assert a['approx_bnii_allot_prob']==pytest.approx(a['parcel_ratio']/25)
    assert a['probability_method']=='RECIPROCAL_APPROXIMATION'

def test_future_or_undated_filings_cannot_trigger_governance():
    cutoff='2026-10-26T15:30:00+05:30'
    bad=dict(filing_sentiment_score=-.7,source_timestamp_verified=True,source_timestamp='2026-10-26T15:31:00+05:30')
    assert not filing_governance([bad],cutoff)['override_sell']
    assert filing_governance([{**bad,'source_timestamp':'2026-10-26T15:30:00+05:30'}],cutoff)['override_sell']
    assert not filing_governance([{**bad,'source_timestamp':None}],cutoff)['override_sell']

def test_milestone_completion_after_cutoff_is_not_forecast(tmp_path):
    item=dict(company_name='Example',date_range='26 - 28 Oct',category='OPEN')
    db=tmp_path/'snapshots.db'
    def fetch():return dict(status='verified',raw_html='observed table',fetched_at_ist='2026-10-28T17:00:01+05:30',values=dict(qib_subscription=2,hni_subscription=3,retail_subscription=4))
    r=capture(item,'https://www.chittorgarh.com/ipo/example/123/',db=db,clock=datetime(2026,10,28,16,59,tzinfo=IST),fetch=fetch)
    assert r['status']=='unavailable'
    with sqlite3.connect(db) as c:
        data=json.loads(c.execute('select evidence_json from module1_milestone_capture_v1').fetchone()[0])
        assert data['categories']['sNII'] is None
        with pytest.raises(sqlite3.IntegrityError):c.execute('delete from module1_milestone_capture_v1')

def test_previous_bidding_session_is_not_calendar_minus_one():
    opened=datetime(2026,10,22,10,tzinfo=IST);closed=datetime(2026,10,26,17,tzinfo=IST)
    assert milestone({},datetime(2026,10,23,17,15,tzinfo=IST),opened,closed)[0]=='DAY2_EOD_SNAPSHOT'
    assert milestone({},datetime(2026,10,25,17,15,tzinfo=IST),opened,closed) is None
