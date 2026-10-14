import numpy as np
import pandas as pd
from ml.research_module2_multihorizon import policy_positions,select


def sample(n):
    return pd.DataFrame(dict(ipo_id=[1]*n,decision_date=pd.date_range('2024-01-01',periods=n),
        rule_position=[1]*n,ma20_gap=[-1.]*n,ma10_gap=[-1.]*n,ret5=[-2.]*n))


def policy():
    return dict(weekly_exit=-.5,confirmation=2,require_daily=True,ema_span=1,min_hold=3,cooldown=2)


def test_negative_daily_signal_cannot_exit_positive_weekly_outlook():
    f=sample(6)
    assert policy_positions(f,np.full(6,-1.),np.full(6,2.),policy()).tolist()==[1]*6


def test_exit_requires_both_forecasts_and_price_breakdown():
    f=sample(6)
    result=policy_positions(f,np.full(6,-1.),np.full(6,-2.),policy())
    assert result.tolist()==[1,1,1,0,0,0]
    assert result.attrs['reasons'].iloc[3]=='SELL_CONFIRMED_MULTIHORIZON'
    f.ma20_gap=2.;f.ret5=2.
    assert policy_positions(f,np.full(6,-1.),np.full(6,-2.),policy()).tolist()==[1]*6


def test_risk_exit_overrides_weekly_bullish_forecast_and_is_causal():
    f=sample(5);f.loc[1,'rule_position']=0
    result=policy_positions(f,np.ones(5),np.full(5,5.),policy())
    assert result.iloc[1]==0
    assert policy_positions(f.iloc[:2],np.ones(2),np.full(2,5.),policy()).tolist()==result.iloc[:2].tolist()


def test_selection_prefers_candidate_passing_each_earlier_fold():
    def candidate(group,gains):
        return dict(group=group,folds={str(year):dict(mean_gain_vs_rules_pct=gain,
            drawdown_gain_vs_rules_pct=1.,strategy=dict(buys=20)) for year,gain in zip((2023,2024),gains)})
    a=candidate('market',(1.,1.));b=candidate('market_flow_events',(-1.,10.))
    chosen,qualified=select([a,b])
    assert qualified and chosen is a
