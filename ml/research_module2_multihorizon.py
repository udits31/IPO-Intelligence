"""Bounded multi-horizon exit challenger, selected on earlier issuer cohorts.

One-day forecast weakness alone must not manufacture a trade. The five-session
model is a separate target with a separate causal label-maturity purge.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ml.improve_module2_next_session import ROOT,PRICE,FLOW,EXTRA,TEXT,EVENT,pipeline,realised,summary,paired,sha,native
from ml.train_module2_next_session import PRICE_PATHS,OFFICIAL
from ml.tune_module2_holding import SOURCE,PRIOR,FEATURES,fit

OUT=ROOT/'output/module2_multihorizon_20261029'
GROUPS={'market':PRICE+FLOW,'market_flow_events':PRICE+FLOW+EXTRA+TEXT+EVENT}
SETTINGS=[('ridge',1000.),('boosting',20.)]
POLICIES=[dict(weekly_exit=exit,confirmation=confirm,require_daily=daily,ema_span=3,min_hold=5,cooldown=2)
          for exit in (-.5,-1.) for confirm in (2,3) for daily in (False,True)]


def weekly_labels(frame):
    prices=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in PRICE_PATHS])
    official=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in OFFICIAL]).drop_duplicates(['ipo_id','date'])
    verified={(int(r.ipo_id),r.date):float(r.close_price_adjusted) for r in official.itertuples()}
    lookup={}
    for id,g in prices.groupby('ipo_id'):
        g=g.sort_values('date').reset_index(drop=True)
        for i in range(len(g)-6):
            window=g.iloc[i+1:i+7]
            if window.date.diff().dt.days.dropna().gt(7).any():continue
            if not all(np.isfinite(r.open) and r.open>0 and verified.get((int(id),r.date),0)>0
                and abs(r.ohlcv_close/verified[(int(id),r.date)]-1)<=.02 for r in window.itertuples()):continue
            lookup[(int(id),g.iloc[i].date)]=(100*(g.iloc[i+6].open/g.iloc[i+1].open-1),g.iloc[i+6].date)
    f=frame.copy();keys=[(int(r.ipo_id),r.decision_date) for r in f.itertuples()]
    f['weekly_return_pct']=[lookup.get(k,(np.nan,pd.NaT))[0] for k in keys]
    f['weekly_outcome_date']=[lookup.get(k,(np.nan,pd.NaT))[1] for k in keys]
    return f


def fit_weekly(frame,features,family,alpha):
    weights=1/frame.groupby('ipo_id').ipo_id.transform('size');weights*=len(weights)/weights.sum()
    model=pipeline(family,alpha)
    model.fit(frame[features],frame.weekly_return_pct,**{model.steps[-1][0]+'__sample_weight':weights})
    return model


def policy_positions(frame,daily,weekly,policy):
    f=frame.copy();f['daily_prediction']=np.asarray(daily);f['weekly_prediction']=np.asarray(weekly)
    positions=pd.Series(0,index=f.index,dtype=int);reasons=pd.Series('',index=f.index,dtype=object)
    for _,g in f.groupby('ipo_id'):
        held=False;ever=False;age=0;confirm=0;cooldown=0;ema=None;dema=None
        for r in g.sort_values('decision_date').itertuples():
            alpha=2/(policy['ema_span']+1)
            ema=r.weekly_prediction if ema is None else alpha*r.weekly_prediction+(1-alpha)*ema
            dema=r.daily_prediction if dema is None else alpha*r.daily_prediction+(1-alpha)*dema
            reason='CASH'
            if int(r.rule_position)==0:
                if held:cooldown=policy['cooldown'];reason='SELL_RISK_RULE'
                else:reason='RISK_ENTRY_BLOCKED'
                held=False;age=confirm=0
            elif held:
                age+=1
                weak=ema<policy['weekly_exit'] and (not policy['require_daily'] or dema<-.15) and r.ma20_gap<0 and r.ret5<0
                confirm=confirm+1 if weak else 0
                if age>=policy['min_hold'] and confirm>=policy['confirmation']:
                    held=False;age=confirm=0;cooldown=policy['cooldown'];reason='SELL_CONFIRMED_MULTIHORIZON'
                else:reason='HOLD_NO_CONFIRMED_BREAKDOWN'
            elif cooldown:
                cooldown-=1;reason='REENTRY_COOLDOWN'
            elif not ever or (r.ma10_gap>0 and r.ret5>0) or ema>1.:
                held=True;ever=True;age=confirm=0;reason='BUY_RULE_ENTRY_OR_RECOVERY'
            positions.loc[r.Index]=int(held);reasons.loc[r.Index]=reason
    positions.attrs['reasons']=reasons
    return positions


def select(candidates):
    qualified=[c for c in candidates if all(f['mean_gain_vs_rules_pct']>=0 and f['drawdown_gain_vs_rules_pct']>=0 for f in c['folds'].values())]
    pool=qualified or candidates
    winner=max(pool,key=lambda c:(min(v['mean_gain_vs_rules_pct'] for v in c['folds'].values()),
        -sum(v['strategy']['buys'] for v in c['folds'].values()),-len(GROUPS[c['group']])))
    return winner,bool(qualified)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    frame=weekly_labels(pd.read_csv(SOURCE,parse_dates=['decision_date','outcome_date','trade_outcome_date']))
    candidates=[dict(group=group,family=family,alpha=alpha,policy=p,folds={})
        for group in GROUPS for family,alpha in SETTINGS for p in POLICIES]
    earlier_predictions=[]
    for year in (2023,2024):
        val=frame[frame.listing_year.eq(year)&frame.outcome_date.lt(f'{year+1}-01-01')].copy()
        earlier=frame[frame.listing_year.lt(year)].copy()
        daily_train=earlier[earlier.trade_outcome_date.notna()&earlier.trade_outcome_date.lt(val.decision_date.min())]
        weekly_train=earlier[earlier.weekly_outcome_date.notna()&earlier.weekly_outcome_date.lt(val.decision_date.min())]
        assert not set(weekly_train.ipo_id)&set(val.ipo_id)
        assert weekly_train.weekly_outcome_date.max()<val.decision_date.min()
        daily=fit(daily_train,FEATURES,'boosting',20.)['holding'].predict(val[FEATURES])
        rules=summary(realised(val,val.rule_position))
        for group,features in GROUPS.items():
            for family,alpha in SETTINGS:
                model=fit_weekly(weekly_train,features,family,alpha);pred=model.predict(val[features])
                earlier_predictions.append(val[['ipo_id','decision_date']].assign(fold=year,group=group,family=family,daily_prediction=daily,weekly_prediction=pred))
                for c in candidates:
                    if (c['group'],c['family'],c['alpha'])!=(group,family,alpha):continue
                    pos=policy_positions(val,daily,pred,c['policy']);s=summary(realised(val,pos))
                    c['folds'][str(year)]=dict(weekly_train_issuers=int(weekly_train.ipo_id.nunique()),validation_issuers=int(val.ipo_id.nunique()),
                        last_training_label=str(weekly_train.weekly_outcome_date.max().date()),first_validation_decision=str(val.decision_date.min().date()),
                        strategy=s,rules=rules,mean_gain_vs_rules_pct=s['mean_net_return_pct']-rules['mean_net_return_pct'],
                        drawdown_gain_vs_rules_pct=s['mean_max_drawdown_pct']-rules['mean_max_drawdown_pct'])
        print(f'Completed earlier fold {year}',flush=True)
    selected,qualified=select(candidates)
    lock=dict(selected=selected,candidates=candidates,qualified_on_both_earlier_folds=qualified)
    (OUT/'selection_before_test.json').write_text(json.dumps(lock,indent=2,default=native)+'\n',encoding='utf8')
    pd.concat(earlier_predictions).to_csv(OUT/'earlier_predictions.csv',index=False)
    test=frame[frame.listing_year.ge(2025)].copy()
    assert test.ipo_id.nunique()==159 and len(test)==5163
    train=frame[frame.listing_year.le(2024)&frame.weekly_outcome_date.notna()&frame.weekly_outcome_date.lt(test.decision_date.min())].copy()
    assert not set(train.ipo_id)&set(test.ipo_id)
    bundle=joblib.load(PRIOR/'module2_retrained_research.joblib')
    daily=bundle['models']['holding'].predict(test[FEATURES])
    weekly=fit_weekly(train,GROUPS[selected['group']],selected['family'],selected['alpha'])
    predicted_weekly=weekly.predict(test[GROUPS[selected['group']]])
    pos=policy_positions(test,daily,predicted_weekly,selected['policy'])
    paths=dict(multihorizon=realised(test,pos),current_rules=realised(test,test.rule_position),always_hold=realised(test,np.ones(len(test))))
    comparisons={k:summary(v) for k,v in paths.items()};pairs={name:paired(paths['multihorizon'],paths[name]) for name in ('current_rules','always_hold')}
    costs={str(cost):{name:summary(realised(test,p,cost)) for name,p in [('multihorizon',pos),('current_rules',test.rule_position)]} for cost in (15,35,60)}
    by_year={str(y):{name:summary([r for r in rows if r['listing_year']==y]) for name,rows in paths.items()} for y in (2025,2026)}
    promotion=(qualified and pairs['current_rules']['bootstrap95_pct'][0]>0 and pairs['always_hold']['bootstrap95_pct'][0]>0
        and all(v['multihorizon']['mean_net_return_pct']>=v['current_rules']['mean_net_return_pct'] for v in by_year.values())
        and costs['60']['multihorizon']['mean_net_return_pct']>=costs['60']['current_rules']['mean_net_return_pct'])
    ledger=test[['ipo_id','company_name','listing_year','decision_date','outcome_date','next_open','next_close','rule_position']].copy()
    ledger['daily_expected_return_pct']=daily;ledger['five_session_expected_return_pct']=predicted_weekly
    ledger['position']=pos;ledger['reason']=pos.attrs['reasons'];ledger.to_csv(OUT/'signals.csv',index=False)
    (OUT/'issuer_trades.json').write_text(json.dumps(paths,indent=2,default=native)+'\n',encoding='utf8')
    joblib.dump(dict(weekly_model=weekly,daily_model=bundle['models']['holding'],weekly_features=GROUPS[selected['group']],daily_features=FEATURES,
        policy=selected['policy'],research_only=True),OUT/'multihorizon_research.joblib')
    report=dict(selected=selected,earlier_selection_qualified=qualified,comparison=comparisons,paired_comparisons=pairs,costs=costs,by_year=by_year,
        test_issuers=159,test_decisions=5163,training_issuers=int(train.ipo_id.nunique()),training_examples=len(train),
        protocol='Five-session target: open[t+6]/open[t+1]-1; targets must mature before the next cohort. Signals at 18:00 IST execute next open. Same later IPOs and dates; 17.5bps on each buy/sell including terminal liquidation. Daily forecast is unchanged.',
        interpretation='Research after repeated inspection of later outcomes. All model/policy settings selected on 2023-2024, but this is not an independent prospective test. No shorting; terminal windows and circuit/slippage assumptions remain limitations.',
        promotion_gate_passed=bool(promotion),live_promoted=False,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,PRIOR/'module2_retrained_research.joblib',Path(__file__)]+PRICE_PATHS+OFFICIAL})
    (OUT/'report.json').write_text(json.dumps(report,indent=2,default=native)+'\n',encoding='utf8')
    lines=['# Module 2 multi-horizon exit research','',str(selected),'',
        '| Strategy | Mean net return per IPO window | Mean drawdown | Buy fills |','|---|---:|---:|---:|']
    for name,s in comparisons.items():lines.append(f"| {name} | {s['mean_net_return_pct']:.3f}% | {s['mean_max_drawdown_pct']:.3f}% | {s['buys']} |")
    lines+=['',str(pairs),'',report['protocol'],'',report['interpretation'],f"Promotion gate: {promotion}; live promoted: False."]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(report,default=native),flush=True)
    return report


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
