"""Earlier-only holding/exit tuning; identical 159-IPO later replay.

Model architecture and weights stay fixed. Only execution state changes.
Earlier folds determine every policy choice before later paths are evaluated.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ml.improve_module2_next_session import ROOT,PRICE,FLOW,fit,realised,summary,paired,positions,sha,native

SOURCE=ROOT/'output/module2_next_session_improvement_20261029/daily_examples.csv'
PRIOR=ROOT/'output/module2_next_session_improvement_20261029'
OUT=ROOT/'output/module2_holding_stability_20261029'
FEATURES=PRICE+FLOW


def grid():
    return [dict(name='original',original=True)]+[
        dict(name=f'ema{span}_confirm{confirm}_hold{hold}_exit{exit}_trend{trend}',original=False,
             ema_span=span,exit_confirmation=confirm,min_hold=hold,exit_threshold=exit,
             trend_protection=trend,cooldown=2,entry_threshold=.35)
        for span in (1,3,5) for confirm in (2,3) for hold in (3,5)
        for exit in (-.15,-.35) for trend in (False,True)]


def holding_positions(frame,predictions,policy):
    if policy.get('original'):return positions(frame,predictions,('return_hysteresis',.35,0.))
    f=frame.copy();f['prediction']=np.asarray(predictions,dtype=float)
    result=pd.Series(0,index=frame.index,dtype=int);reasons=pd.Series('',index=frame.index,dtype=object)
    for _,g in f.groupby('ipo_id'):
        held=False;age=0;streak=0;cooldown=0;ema=None
        for r in g.sort_values('decision_date').itertuples():
            alpha=2/(policy['ema_span']+1);ema=r.prediction if ema is None else alpha*r.prediction+(1-alpha)*ema
            # All earlier candidates share the existing dated risk floor.
            # A minimum holding period can never suppress a risk exit.
            risk_exit=int(r.rule_position)==0
            reason='CASH'
            if risk_exit:
                if held:cooldown=policy['cooldown'];reason='SELL_RISK_RULE'
                else:reason='RISK_ENTRY_BLOCKED'
                held=False;age=streak=0
            elif held:
                age+=1
                trend=policy['trend_protection'] and r.ma20_gap>0 and r.ret5>0
                streak=streak+1 if ema<policy['exit_threshold'] and not trend else 0
                if age>=policy['min_hold'] and streak>=policy['exit_confirmation']:
                    held=False;age=streak=0;cooldown=policy['cooldown'];reason='SELL_CONFIRMED_FORECAST'
                else:reason='HOLD_TREND' if trend else 'HOLD_AWAIT_CONFIRMATION'
            elif cooldown:
                cooldown-=1;reason='REENTRY_COOLDOWN'
            elif r.prediction>policy['entry_threshold']:
                held=True;age=0;streak=0;reason='BUY_FORECAST'
            result.loc[r.Index]=int(held);reasons.loc[r.Index]=reason
    result.attrs['reasons']=reasons
    return result


def choose(candidates):
    challengers=[c for c in candidates if not c['policy'].get('original')]
    # Prefer policies that cut churn and retain at least the earlier current-
    # rule mean return and drawdown in BOTH earlier periods.
    qualified=[c for c in challengers if all(v['gain_vs_rules_pct']>=0 and v['drawdown_gain_vs_rules_pct']>=0
        and v['strategy']['buys']<=v['original']['buys'] for v in c['folds'].values())]
    pool=qualified or challengers
    selected=max(pool,key=lambda c:(min(v['gain_vs_rules_pct'] for v in c['folds'].values()),
        -sum(v['strategy']['buys'] for v in c['folds'].values())))
    return selected,bool(qualified)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SOURCE,parse_dates=['decision_date','outcome_date','trade_outcome_date'])
    policies=grid();candidates=[dict(policy=p,folds={}) for p in policies];validation_predictions=[]
    for year in (2023,2024):
        val=data[data.listing_year.eq(year)].copy()
        val=val[val.outcome_date.lt(pd.Timestamp(f'{year+1}-01-01'))]
        train=data[data.listing_year.lt(year)&data.trade_outcome_date.notna()&data.trade_outcome_date.lt(val.decision_date.min())].copy()
        assert not set(train.ipo_id)&set(val.ipo_id)
        assert train.trade_outcome_date.max()<val.decision_date.min()
        # Exact prior shallow-boosting family; no later model/feature search.
        models=fit(train,FEATURES,'boosting',20.)
        pred=models['holding'].predict(val[FEATURES]);validation_predictions.append(val[['ipo_id','decision_date']].assign(prediction=pred,fold=year))
        rules=summary(realised(val,val.rule_position));old=summary(realised(val,holding_positions(val,pred,policies[0])))
        print(f'Policy selection fold {year}: {train.ipo_id.nunique()} training / {val.ipo_id.nunique()} validation issuers',flush=True)
        for c in candidates:
            pos=holding_positions(val,pred,c['policy']);s=summary(realised(val,pos))
            c['folds'][str(year)]=dict(train_issuers=int(train.ipo_id.nunique()),validation_issuers=int(val.ipo_id.nunique()),
                last_training_label=str(train.trade_outcome_date.max().date()),first_decision=str(val.decision_date.min().date()),
                strategy=s,current_rules=rules,original=old,gain_vs_rules_pct=s['mean_net_return_pct']-rules['mean_net_return_pct'],
                drawdown_gain_vs_rules_pct=s['mean_max_drawdown_pct']-rules['mean_max_drawdown_pct'])
    selected,qualified=choose(candidates)
    locked=dict(selection=selected,qualified_in_both_earlier_folds=qualified,candidates=candidates,
        protocol='Minimise worst-earlier-fold return deficit vs current rules, with turnover tie-break. Prefer candidates satisfying return, drawdown and churn constraints in both 2023 and 2024. No later-year selection.')
    (OUT/'selection_before_later_evaluation.json').write_text(json.dumps(locked,indent=2,default=native)+'\n',encoding='utf8')
    pd.concat(validation_predictions).to_csv(OUT/'earlier_predictions.csv',index=False)
    # Later outcomes enter policy evaluation only after the selection file.
    test=data[data.listing_year.ge(2025)].copy()
    original_keys=pd.read_csv(ROOT/'output/module2_next_session_20261029/test_predictions.csv')
    original_keys=original_keys[original_keys.variant=='price'][['ipo_id','decision_date']]
    assert set(zip(test.ipo_id,test.decision_date.dt.strftime('%Y-%m-%d')))==set(map(tuple,original_keys.to_numpy()))
    assert test.ipo_id.nunique()==159 and len(test)==5163
    bundle=joblib.load(PRIOR/'module2_retrained_research.joblib')
    if bundle['features']!=FEATURES or bundle['selection']['family']!='boosting':raise ValueError('Unexpected frozen research artifact')
    pred=bundle['models']['holding'].predict(test[FEATURES])
    selection_pos=holding_positions(test,pred,selected['policy']);old_pos=holding_positions(test,pred,policies[0])
    paths={name:realised(test,pos) for name,pos in dict(tuned=selection_pos,original_ml=old_pos,current_rules=test.rule_position,always_hold=np.ones(len(test))).items()}
    comparison={name:summary(rows) for name,rows in paths.items()}
    ledger=test[['ipo_id','company_name','listing_year','decision_date','outcome_date','next_open','next_close','rule_position']].copy()
    ledger['expected_return_pct']=pred;ledger['original_position']=old_pos;ledger['tuned_position']=selection_pos
    ledger['reason']=selection_pos.attrs['reasons'];ledger.to_csv(OUT/'signals.csv',index=False)
    pairs={name:paired(paths['tuned'],paths[name]) for name in ('current_rules','original_ml','always_hold')}
    costs={str(cost):{name:summary(realised(test,pos,cost)) for name,pos in
        dict(tuned=selection_pos,original_ml=old_pos,current_rules=test.rule_position).items()} for cost in (15,35,60)}
    by_year={str(year):{name:summary([r for r in rows if r['listing_year']==year]) for name,rows in paths.items()} for year in (2025,2026)}
    promotion=(qualified and pairs['current_rules']['bootstrap95_pct'][0]>0 and pairs['always_hold']['bootstrap95_pct'][0]>0
        and comparison['tuned']['buys']<comparison['original_ml']['buys']
        and all(v['tuned']['mean_net_return_pct']>=v['current_rules']['mean_net_return_pct'] for v in by_year.values())
        and costs['60']['tuned']['mean_net_return_pct']>=costs['60']['current_rules']['mean_net_return_pct'])
    # Exit attribution: compare the exact first old exit fill with each tuned
    # position at that date. Future realised P&L diagnoses, never selects, policy.
    details=[]
    for id,g in ledger.groupby('ipo_id'):
        ordered=g.sort_values('outcome_date');old_before=ordered.original_position.shift(fill_value=0)
        exits=ordered[(old_before==1)&(ordered.original_position==0)]
        blocked=exits[exits.tuned_position==1]
        details.append(dict(ipo_id=int(id),company_name=g.company_name.iloc[0],old_exit_signals=len(exits),old_exits_held_by_tuned=len(blocked)))
    pd.DataFrame(details).to_csv(OUT/'exit_diagnostics.csv',index=False)
    (OUT/'issuer_trades.json').write_text(json.dumps(paths,default=native,indent=2)+'\n',encoding='utf8')
    bundle.update(holding_policy=selected['policy'],research_only=True)
    joblib.dump(bundle,OUT/'holding_policy_research.joblib')
    report=dict(selected_policy=selected['policy'],earlier_selection=locked,comparison=comparison,
        paired_comparisons=pairs,costs=costs,by_year=by_year,n_issuers=159,n_decisions=5163,
        old_exits_held_by_tuned=sum(r['old_exits_held_by_tuned'] for r in details),
        promotion_gate_passed=bool(promotion),live_promoted=False,
        interpretation='Earlier-only selection; later results are an inspected retrospective replay, not an untouched future test. Direction-model predictions are unchanged by holding-policy tuning.',
        execution='18:00 signal, next-open fills, carry shares overnight, 17.5bps each buy/sell, final liquidation at last common close. Risk exits override minimum holding periods.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,PRIOR/'module2_retrained_research.joblib',Path(__file__)]})
    (OUT/'report.json').write_text(json.dumps(report,default=native,indent=2)+'\n',encoding='utf8')
    lines=['# Module 2: earlier-only holding and exit tuning','',str(selected['policy']),'',
        '| Strategy | Net return per issuer window | Mean drawdown | Buy fills |','|---|---:|---:|---:|']
    for name,s in comparison.items():lines.append(f"| {name} | {s['mean_net_return_pct']:.3f}% | {s['mean_max_drawdown_pct']:.3f}% | {s['buys']} |")
    lines+=['',f"Qualified on both earlier periods: {qualified}. Paired comparisons: {pairs}.",
        '',report['execution'],'',report['interpretation'],f"Promotion gate passed: {promotion}; live promoted: False."]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('earlier_selection','input_sha256')},default=native),flush=True)
    return report


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
