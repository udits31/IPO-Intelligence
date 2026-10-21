import numpy as np
import pandas as pd
import pytest
from ml.stage_alignment_bridge import parse_ipoji,StageBridge,CATS
from ml.run_module1_ablation_20261026 import day2_track,route_track
from ml.run_module2_rule_ablation_20261026 import gate_action


def test_publisher_parentheses_are_not_blank_or_negative():
    page='''<script id="ipo-subscription-info-for-chart">[{"key":"Day 2 - 16-12-21 05:00 PM","value":{"QIB":"(0.00)","NII":"(1.90)","RII":"-","Total":"(8.03)"}}]</script>'''
    r=parse_ipoji(page)[0]
    assert r['qib']==0 and r['hni']==1.9 and r['retail'] is None
    assert r['timestamp']=='2021-12-16T17:00:00+05:30'


def pairs(n):
    return pd.DataFrame([dict(issuer_key=str(i),log_issue_size=5+i/100,gmp_pct=i-10,
      **{'day2_'+c:1+i/10 for c in CATS},**{'cutoff_'+c:3+i/5 for c in CATS}) for i in range(n)])


def test_bridge_rejects_heldout_issuers_and_duplicate_pairs():
    p=pairs(25)
    with pytest.raises(ValueError):StageBridge().fit(p,allowed_ids=['1'])
    with pytest.raises(ValueError):StageBridge().fit(pd.concat([p,p]),allowed_ids=p.issuer_key)


def test_no_bridge_estimates_without_enough_real_pairs():
    p=pairs(3);m=StageBridge().fit(p,allowed_ids=p.issuer_key)
    assert not m.models
    with pytest.raises(ValueError):m.predict(p,'inverse')


def test_bridge_is_regularized_training_only_and_returns_nonnegative_values():
    p=pairs(25);m=StageBridge().fit(p,allowed_ids=p.issuer_key)
    assert m.status=='fitted_training_only'
    assert m.predict(p,'forward').shape==(25,3)
    assert np.all(m.predict(p,'inverse')>=0)


def test_day2_track_does_not_use_same_day_unclocked_gmp_or_future_macro():
    row=dict(day2_timestamp='2025-01-02T17:00:00+05:30',day2_qib=2,day2_hni=3,day2_retail=4,
      day1_qib=1,day1_hni=1,day1_retail=1,gmp_date='2025-01-02',gmp_pct=20,gmp_quote_age_days=0,
      gmp_slope_pp_per_day=2,macro_source_date='2025-01-03',macro_vix=15)
    out=day2_track(pd.DataFrame([row])).iloc[0]
    assert pd.isna(out.gmp_pct) and pd.isna(out.macro_vix)
    assert out.log_qib==np.log1p(2) and np.isclose(out.log_qib_growth,np.log(3/2))


def features():
    return dict(sessions=31,close=120.,issue_price=100.,day1_low=99.,sma_10=115.,sma_20=110.,ret_10d=-.02,
      rs_vs_nifty_20d=.02,drawdown_pct=5.,filing_sentiment_score=.1,unabsorbed_unlock_window=False)


def test_gate_ablation_adds_constraints_without_treating_unknown_as_clean():
    f=features();assert gate_action(f,2)['sell'] is False
    assert gate_action({**f,'rs_vs_nifty_20d':-.03},2)['sell'] is False
    assert gate_action({**f,'rs_vs_nifty_20d':-.03},3)['sell'] is True
    assert gate_action({**f,'filing_sentiment_score':None},4)['sell'] is True
    assert route_track(has_preclose=False,has_day2=True)=='M1_DAY2_TRACK'
