import numpy as np
import pandas as pd
from ml.tune_module2_holding import holding_positions,choose


def frame(n):
    return pd.DataFrame(dict(ipo_id=[1]*n,decision_date=pd.date_range('2024-01-01',periods=n),
                             rule_position=[1]*n,ma20_gap=[-1.]*n,ret5=[-1.]*n))


def policy(**kw):
    return dict(original=False,ema_span=1,exit_confirmation=2,min_hold=3,
                exit_threshold=-.15,trend_protection=False,cooldown=2,entry_threshold=.35,**kw)


def test_exit_confirmation_and_minimum_holding_reduce_one_day_whipsaw():
    f=frame(6);pred=np.array([1.,-.5,.3,-.5,-.5,1.])
    pos=holding_positions(f,pred,policy())
    assert pos.tolist()==[1,1,1,1,0,0]
    assert holding_positions(f.iloc[:3],pred[:3],policy()).tolist()==pos.iloc[:3].tolist()


def test_risk_exit_overrides_minimum_holding_and_cooldown_blocks_reentry():
    f=frame(5);f.loc[1,'rule_position']=0
    pos=holding_positions(f,np.ones(5),policy())
    assert pos.tolist()==[1,0,0,0,1]
    assert pos.attrs['reasons'].iloc[1]=='SELL_RISK_RULE'


def test_trend_protection_does_not_ignore_hard_risk_exit():
    f=frame(6);f.ma20_gap=2.;f.ret5=3.;f.loc[5,'rule_position']=0
    p=policy();p['trend_protection']=True
    assert holding_positions(f,[1.,-.5,-.5,-.5,-.5,-.5],p).tolist()==[1,1,1,1,1,0]


def test_selection_uses_only_earlier_metrics_and_rejects_churnier_candidate():
    def candidate(name,gain,buys):
        return dict(policy=dict(name=name),folds={str(y):dict(gain_vs_rules_pct=gain,
            drawdown_gain_vs_rules_pct=1,strategy=dict(buys=buys),original=dict(buys=20)) for y in (2023,2024)})
    a=candidate('quieter',1,10);b=candidate('churnier',2,30)
    selected,qualified=choose([a,b])
    assert qualified and selected['policy']['name']=='quieter'


def test_rule_led_entry_retains_rally_until_both_price_and_forecast_weaken():
    f=frame(7);f.ma20_gap=2.;f.ret5=3.
    p=policy();p.update(rule_led_entry=True,require_price_deterioration=True,trend_protection=True)
    # A weak ML entry estimate must not delay the configured rule-led entry.
    f.loc[4:,'ma20_gap']=-1.;f.loc[4:,'ret5']=-2.
    result=holding_positions(f,np.full(7,-.5),p)
    assert result.tolist()==[1,1,1,1,1,0,0]
    assert result.attrs['reasons'].iloc[0]=='BUY_RULE_ENTRY'
