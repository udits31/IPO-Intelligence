import json
import numpy as np
import pandas as pd
from core.features import availability
from services.ipo_regime_service import rolling_regime,adjusted_gmp
from ml.research_module1_progressive import time_weights,passes
from ml.recover_progressive_totals import parse_history
from core.prove_it_monitor_engine import decision
from core.prove_it_monitor_engine import listing_day_rule
from services.retail_hni_economics import capital_plan


def test_static_availability_assumption_is_explicit_not_fetch_backdating():
    args=dict(kind='PROSPECTUS_STATIC',decision_timestamp='2022-03-02T17:00:00+05:30',issue_open_date='2022-03-01',fetch_timestamp='2026-10-26T10:00:00+05:30')
    assert not availability(**args)['eligible']
    r=availability(**args,allow_historical_assumptions=True)
    assert r['eligible'] and r['availability_assumed'] and r['source_timestamp'] is None


def test_same_day_eod_does_not_enter_close_decision():
    args=dict(kind='EXCHANGE_EOD_TAPE',decision_timestamp='2025-06-03T15:30:00+05:30',trading_date='2025-06-03',allow_historical_assumptions=True)
    assert not availability(**args)['eligible']
    assert availability(**{**args,'trading_date':'2025-06-02'})['eligible']


def test_half_life_weights_do_not_discard_old_rows():
    rows=pd.DataFrame({'listing_date':['2025-01-01','2022-07-03']})
    w=time_weights(rows,'2025-01-01')
    assert w[0]==1 and .49<w[1]<.51 and np.all(w>0)


def test_regime_excludes_future_and_same_day_listings():
    h=pd.DataFrame([dict(issuer_key=str(i),listing_date=date,actual_return_pct=ret,gmp_pct=gmp,gmp_preclose_verified=True) for i,(date,ret,gmp) in enumerate([('2025-01-01',-10,10),('2025-01-03',100,0),('2025-01-04',200,0)])])
    r=rolling_regime(h,'2025-01-03')
    assert r['prior_listings']==1 and r['rolling_10_gmp_slippage']==-20 and r['rolling_10_hit_rate']==0
    assert adjusted_gmp(10,r['rolling_10_gmp_slippage'])==0 and adjusted_gmp(10,20)==10


def test_promotion_requires_both_cohorts_and_false_positive_limit():
    assert not passes(dict(correct=59,wrong_positive_calls=3),dict(correct=14))
    assert not passes(dict(correct=57,wrong_positive_calls=6),dict(correct=18))
    assert passes(dict(correct=56,wrong_positive_calls=5),dict(correct=16))


def valid_features():
    return dict(sessions=31,close=120.,issue_price=100.,day1_low=99.,sma_10=115.,sma_20=110.,ret_10d=.03,
        rs_vs_nifty_20d=.02,drawdown_pct=5.,atr_14_pct=3.,filing_sentiment_score=.1,unabsorbed_unlock_window=False,delivery_accumulation_ratio=1.1)


def test_prove_it_requires_all_gates_and_never_treats_unknown_as_clean():
    f=valid_features();assert decision(f)['action']=='HOLD'
    assert decision({**f,'rs_vs_nifty_20d':.04})['action']=='ACCUMULATE'
    assert decision({**f,'sma_20':125})['action']=='REVIEW / REDUCE'
    assert decision({**f,'day1_low':None})['sell'] is True
    assert decision({**f,'close':100.5})['action']=='EXIT / REDUCE'
    assert decision({**f,'filing_sentiment_score':None})['sell'] is True


def test_prove_it_not_applied_to_listing_day_without_intraday_tape():
    assert decision({**valid_features(),'sessions':1})['sell'] is None


def test_tiered_capital_does_not_claim_exact_odds_or_empirical_miss_loss():
    r=capital_plan(score=.9,regime_adjusted_gmp_pct=20,qib=10,hni=10,retail=10,cap_price=300,lot_size=50,snii=10,bnii=20,financing_apr_pct=15)
    assert r['tier']==1 and r['deployable'] is False
    assert r['assumptions']['miss_loss_pct']==6.5
    assert r['category_scenarios']['bnii']['full_allotment_loss_stress_EU_rupees']<0
    assert r['category_scenarios']['bnii']['funded_viable_under_scenario'] is False


def test_total_recovery_excludes_future_rows_and_matches_actual_category_book():
    payload=[dict(key='Day 2 - 02-01-25 05:00 PM',value={'QIB':'(2)','NII':'(3)','RII':'(4)','Total':'(3)'}),dict(key='Day 3 - 03-01-25 05:00 PM',value={'QIB':'(20)','NII':'(30)','RII':'(40)','Total':'(30)'})]
    html='Example Limited<script id="ipo-subscription-info-for-chart">'+json.dumps(payload)+'</script>'
    row=dict(company_name='Example Limited',cutoff_at_ist='2025-01-02T17:00:00+05:30',log_qib=np.log1p(2),log_hni=np.log1p(3),log_retail=np.log1p(4))
    r=parse_history(html,row)
    assert r['day2_to_cutoff_conviction']==1 and '2025-01-02' in r['matched_bid_timestamp']
    assert parse_history(html,{**row,'log_qib':np.log1p(200)}) is None


def test_listing_day_rule_uses_complete_continuous_bars_not_auction_indications():
    opened=pd.Timestamp('2026-10-21T10:00:00+05:30');bars=[]
    for i in range(16):
        start=opened+pd.Timedelta(minutes=i);end=start+pd.Timedelta(minutes=1)
        bars.append(dict(bar_start_ist=start.isoformat(),bar_end_ist=end.isoformat(),source_timestamp=end.isoformat(),source_timestamp_verified=True,session_phase='CONTINUOUS',low=99,close=98 if i==15 else 100))
    args=dict(continuous_open_timestamp=opened.isoformat(),decision_timestamp='2026-10-21T10:16:00+05:30',listing_open=100,issue_price=100)
    assert listing_day_rule(bars,**args)['action']=='EXIT / REDUCE'
    assert listing_day_rule([{**r,'session_phase':'AUCTION'} for r in bars],**args)['action']=='DATA_PENDING'
    assert listing_day_rule(bars[:-2],**args)['action']=='DATA_PENDING'
