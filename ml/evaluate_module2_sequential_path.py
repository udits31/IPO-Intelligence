"""Causal daily R0/R1 replay, explicit session numbering, and executable paths."""
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from core.monitor_r0_frozen_20261026 import HoldSellMonitor
from services.filing_governance import filing_governance
from services.point_in_time import eligible,timestamp
from services.enrichment_runtime import market_records
from services.saved_preclose_service import ROOT

OUT=ROOT/'output/module2_sequential_path_evaluation_20261026.json'
LOCK=ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json'
FEE=.003


def json_native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(datetime,pd.Timestamp)):return value.isoformat()
    raise TypeError(f'Unexpected non-JSON value {type(value).__name__}')


def dated_filings(records,cutoff):
    limit=timestamp(cutoff)
    return [r for r in records if eligible(r,cutoff) and timestamp(r.get('broadcast_datetime_ist')) is not None
            and timestamp(r['broadcast_datetime_ist'])<=limit]


def replay_calls(prices,ipo,filings):
    """No future prices are passed to the monitor. All calls use dated closes."""
    monitor=HoldSellMonitor(prices_frame=prices.iloc[:0].copy(),ipos_frame=pd.DataFrame([ipo]),prefer_supplied_prices=True)
    calls=[]
    for index in range(30,60):
        history=prices.iloc[:index+1].copy()
        monitor.attribution_engine.df_prices=history
        date=str(pd.Timestamp(history.iloc[-1].date).date());cutoff=date+'T15:30:00+05:30'
        result=monitor.evaluate_holding(int(ipo['id']),as_of_date=date,fetch_news=False,disclosures=[])
        if 'verdict' not in result:raise ValueError(f"Cannot evaluate {ipo['id']} at {date}: {result}")
        verdict=result['verdict'];sell=('SELL' in verdict or 'TAKE PROFIT' in verdict) and 'WATCH' not in verdict
        news=filing_governance(dated_filings(filings,cutoff),cutoff)
        calls.append(dict(session=index+1,python_index=index,decision_date=date,decision_timestamp=cutoff,
            r0_sell=bool(sell),r1_sell=bool(sell or news['override_sell']),r0_action=verdict,
            r1_action='SELL / REDUCE' if news['override_sell'] else verdict,
            price_reason=result['decision_reason']['code'],admitted_filing_count=news['admitted_filings'],
            filing_sentiment_score=news['filing_sentiment_score'],filing_override=news['override_sell'],
            eligible_filing_evidence=news['evidence'],last_feature_price_session=index+1,
            anchor_calendar=result.get('lockin_status')))
    return calls


def first_exit_path(closes,sells,*,reference_index=31,terminal_index=61,signal_start_index=30,fee=FEE):
    """sells[j] is a close-time call at signal_start_index+j; execute next close."""
    closes=np.asarray(closes,dtype=float)
    if np.any(~np.isfinite(closes[:terminal_index+1])) or np.any(closes[:terminal_index+1]<=0):raise ValueError('Missing/invalid realised prices')
    exit_signal=next((signal_start_index+j for j,s in enumerate(sells) if s),None)
    execution=exit_signal+1 if exit_signal is not None else None
    if execution is not None and (execution<reference_index or execution>terminal_index):raise ValueError('Execution outside evaluation window')
    reference=float(closes[reference_index]);gross_terminal=float(closes[terminal_index]/reference-1)
    if execution is None:
        a=gross_terminal;b=float(closes[terminal_index]*(1-fee)/reference-1)
    else:a=b=float(closes[execution]*(1-fee)/reference-1)
    # Include wealth before and after each close fill, so immediate selling costs
    # cannot disappear from drawdown by starting the path after the fill.
    wealth=[1.]
    for i in range(reference_index,terminal_index+1):
        if execution is not None and i>execution:wealth.append(wealth[-1]);continue
        wealth.append(float(closes[i]/reference))
        if execution==i:wealth.append(float(closes[i]*(1-fee)/reference))
    wealth_a=np.array(wealth);wealth_b=np.append(wealth_a,wealth_a[-1]*(1-fee)) if execution is None else wealth_a
    def dd(w):return float(np.min(w/np.maximum.accumulate(w)-1))*100
    flips=int(np.count_nonzero(np.asarray(sells[1:],dtype=bool)!=np.asarray(sells[:-1],dtype=bool)))
    return dict(exit_signal_index=exit_signal,execution_index=execution,exited=execution is not None,
        net_return_A_pct=100*a,net_return_B_pct=100*b,buy_hold_gross_return_pct=100*gross_terminal,
        buy_hold_net_return_pct=100*float(closes[terminal_index]*(1-fee)/reference-1),
        max_drawdown_A_pct=dd(wealth_a),max_drawdown_B_pct=dd(wealth_b),signal_flips=flips,
        executed_sell_count=int(execution is not None),sell_turnover_reference_pct=100*float(closes[execution]/reference) if execution is not None else 0.)


def multi_trade_path(closes,sells,*,reference_index=31,terminal_index=61,signal_start_index=30,fee=FEE):
    closes=np.asarray(closes,dtype=float);reference=closes[reference_index]
    shares=1/reference;cash=0.;held=True;trades=[];wealth=[1.]
    by_execution={signal_start_index+j+1:bool(s) for j,s in enumerate(sells)}
    for i in range(reference_index,terminal_index+1):
        wealth.append(float(cash+shares*closes[i]))
        desired_sell=by_execution.get(i)
        if desired_sell is True and held:
            notional=shares*closes[i];cash=notional*(1-fee);shares=0.;held=False
            trades.append(dict(execution_index=i,side='SELL',notional_reference_pct=float(notional*100)))
        elif desired_sell is False and not held:
            notional=cash;shares=cash*(1-fee)/closes[i];cash=0.;held=True
            trades.append(dict(execution_index=i,side='BUY',notional_reference_pct=float(notional*100)))
        wealth.append(float(cash+shares*closes[i]))
    final=float(cash+shares*closes[terminal_index]);b=float(cash+shares*closes[terminal_index]*(1-fee))
    wa=np.array(wealth);wb=np.append(wa,b)
    return dict(net_return_A_pct=100*(final-1),net_return_B_pct=100*(b-1),
        max_drawdown_A_pct=float(np.min(wa/np.maximum.accumulate(wa)-1))*100,
        max_drawdown_B_pct=float(np.min(wb/np.maximum.accumulate(wb)-1))*100,
        turnover_reference_pct=sum(r['notional_reference_pct'] for r in trades),
        terminal_liquidation_turnover_reference_pct=float(shares*closes[terminal_index]*100) if held else 0.,
        transition_count=len(trades),signal_flips=int(np.count_nonzero(np.asarray(sells[1:],dtype=bool)!=np.asarray(sells[:-1],dtype=bool))),trades=trades)


def summary(records):
    result=dict(n=len(records))
    for convention in ('A','B'):
        returns=np.array([r['net_return_'+convention+'_pct'] for r in records]);drawdowns=np.array([r['max_drawdown_'+convention+'_pct'] for r in records])
        count=max(1,int(np.ceil(.1*len(records))))
        result[convention]=dict(mean_net_return_pct=float(returns.mean()),median_net_return_pct=float(np.median(returns)),
            return_10th_percentile_pct=float(np.quantile(returns,.1)),mean_max_drawdown_pct=float(drawdowns.mean()),
            worst_decile_mean_max_drawdown_pct=float(np.sort(drawdowns)[:count].mean()),worst_decile_n=count,
            fraction_profitable_pct=100*float((returns>0).mean()))
    result['first_exits']=sum(r.get('exited',False) for r in records)
    result['signal_flip_total']=int(sum(r.get('signal_flips',0) for r in records))
    result['executed_transition_total']=int(sum(r.get('transition_count',r.get('executed_sell_count',0)) for r in records))
    result['mean_signal_turnover_reference_pct']=float(np.mean([r.get('turnover_reference_pct',r.get('sell_turnover_reference_pct',0)) for r in records]))
    return result


def held_fates(records,call_map,prices_map,*,terminal_index,reference_index):
    held=[r for r in records if not call_map[r['ipo_id']][0]['r1_sell']]
    groups={}
    for positive,label in [(False,'held_eventual_losers'),(True,'held_eventual_winners')]:
        group=[r for r in held if r['locked_actual_positive']==positive]
        exited=[r for r in group if r['exited'] and r['exit_signal_index']>30]
        savings=[]
        for r in exited:
            prices=prices_map[r['ipo_id']].close_price_adjusted.to_numpy()
            comparable_hold=100*(prices[terminal_index]*(1-FEE)/prices[reference_index]-1)
            savings.append(r['net_return_B_pct']-comparable_hold)
        groups[label]=dict(n=len(group),midwindow_exits=len(exited),still_held=len(group)-len(exited),
            mean_exit_net_return_pct=float(np.mean([r['net_return_B_pct'] for r in exited])) if exited else None,
            mean_terminal_gross_hold_return_pct=float(np.mean([r['buy_hold_gross_return_pct'] for r in exited])) if exited else None,
            mean_saved_return_after_comparable_exit_cost_pp=float(np.mean(savings)) if savings else None,
            mean_upside_forfeited_after_comparable_cost_pp=float(-np.mean(savings)) if savings and positive else None,
            exits_that_saved_money=sum(s>0 for s in savings),exits_that_hurt_returns=sum(s<0 for s in savings),
            premature_winner_exits=sum(s<0 for s in savings) if positive else None,
            mean_positive_upside_forfeited_pp=float(np.mean([-s for s in savings if s<0])) if positive and any(s<0 for s in savings) else None,
            exit_issuer_ids=[r['ipo_id'] for r in exited])
    return dict(initial_held_n=len(held),**groups)


def run():
    original=LOCK.read_bytes();lock_sha=hashlib.sha256(original).hexdigest();locked=json.loads(original)
    ledger_path=ROOT/'output/module2_holdout_decision_ledger_governance_20261023.csv'
    price_path=ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full_ca_reconciled.csv'
    ledger=pd.read_csv(ledger_path,parse_dates=['decision_date','outcome_date'])
    prices=pd.read_csv(price_path,parse_dates=['date']);ipos=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    baseline=pd.read_csv(ROOT/'output/module2_rule_ablation_20261026.csv')
    baseline=baseline[baseline.variant.eq('M2_R1_WITH_DATED_FILINGS')].set_index('ipo_id')
    groups={int(k):g.sort_values('date').reset_index(drop=True) for k,g in prices.groupby('ipo_id')}
    if len(ledger)!=132 or ledger.ipo_id.nunique()!=132:raise ValueError('Changed governance cohort')
    daily=[];call_map={};snapshot=[]
    for count,row in enumerate(ledger.itertuples(),1):
        group=groups[row.ipo_id]
        if group.iloc[30].date!=row.decision_date or group.iloc[61].date!=row.outcome_date:raise ValueError('Session identity changed')
        ipo=ipos.loc[row.ipo_id].to_dict();ipo['id']=row.ipo_id
        news=market_records('corporate_announcements',row.ipo_id)
        news=[r for r in news if eligible(r,str(group.iloc[59].date.date())+'T15:30:00+05:30')]
        snapshot.append(dict(ipo_id=row.ipo_id,records=news))
        calls=replay_calls(group,ipo,news)
        if calls[0]['r0_sell']!=bool(row.sell):raise ValueError(f'R0 checkpoint mismatch {row.ipo_id}')
        if calls[0]['r1_sell']!=bool(baseline.loc[row.ipo_id,'sell']):raise ValueError(f'R1 checkpoint mismatch {row.ipo_id}')
        call_map[row.ipo_id]=calls
        daily.extend(dict(ipo_id=row.ipo_id,company_name=row.company_name,**r) for r in calls)
        if count%12==0:print(json.dumps(dict(issuers_replayed=count,total=132)),flush=True)
    windows={}
    for window,reference,terminal in [('LOCKED_EXECUTION_ALIGNED',31,61),('LITERAL_SESSION_31_TO_61',30,60)]:
        records=[]
        for row in ledger.itertuples():
            group=groups[row.ipo_id];close=group.close_price_adjusted.to_numpy();calls=call_map[row.ipo_id][:30]
            paths={'M2_R0_CONTINUOUS':[r['r0_sell'] for r in calls], 'M2_R1_CONTINUOUS':[r['r1_sell'] for r in calls],
                   'M2_R1_STATIC_31':[calls[0]['r1_sell']]+[False]*29,
                   'ALWAYS_HOLD_31_TO_61':[False]*30,'ALWAYS_SELL_AT_31':[True]+[False]*29}
            for policy,sells in paths.items():
                path=first_exit_path(close,sells,reference_index=reference,terminal_index=terminal)
                path.update(ipo_id=row.ipo_id,company_name=row.company_name,policy=policy,window=window,
                    locked_actual_positive=bool(row.day30_return_pct>0),reference_session=reference+1,terminal_session=terminal+1,
                    reference_date=str(group.iloc[reference].date.date()),terminal_date=str(group.iloc[terminal].date.date()),
                    exit_signal_session=path['exit_signal_index']+1 if path['exited'] else None,
                    exit_execution_session=path['execution_index']+1 if path['exited'] else None,
                    exit_decision_date=str(group.iloc[path['exit_signal_index']].date.date()) if path['exited'] else None,
                    exit_execution_date=str(group.iloc[path['execution_index']].date.date()) if path['exited'] else None)
                records.append(path)
            for policy,key in [('M2_R0_ALL_SUCCESSIVE_CALLS','r0_sell'),('M2_R1_ALL_SUCCESSIVE_CALLS','r1_sell')]:
                p=multi_trade_path(close,[r[key] for r in calls],reference_index=reference,terminal_index=terminal)
                records.append(dict(ipo_id=row.ipo_id,company_name=row.company_name,policy=policy,window=window,**p))
        policies={name:summary([r for r in records if r['policy']==name]) for name in dict.fromkeys(r['policy'] for r in records)}
        r1=[r for r in records if r['policy']=='M2_R1_CONTINUOUS']
        fate=held_fates(r1,call_map,groups,terminal_index=terminal,reference_index=reference)
        assert fate['initial_held_n']==102 and fate['held_eventual_winners']['n']==44 and fate['held_eventual_losers']['n']==58
        if window=='LOCKED_EXECUTION_ALIGNED':
            expected=locked['module2']['results']['M2_R1_WITH_DATED_FILINGS']['conditional_strategy_net_mean_pct']
            if not np.isclose(policies['M2_R1_STATIC_31']['A']['mean_net_return_pct'],expected,atol=1e-9):raise ValueError('Static R1 replay does not reproduce locked return')
            assert np.isclose(policies['ALWAYS_SELL_AT_31']['A']['mean_net_return_pct'],-.3)
        comparisons={}
        for convention in ('A','B'):
            current=policies['M2_R1_CONTINUOUS'][convention]
            comparisons[convention]={name:{metric:dict(delta=current[metric]-base[convention][metric],better=bool(current[metric]-base[convention][metric]>1e-9),equal_within_tolerance=bool(abs(current[metric]-base[convention][metric])<=1e-9)) for metric in ['mean_net_return_pct','median_net_return_pct','worst_decile_mean_max_drawdown_pct']} for name,base in policies.items() if name in ('M2_R1_STATIC_31','ALWAYS_SELL_AT_31')}
            for name in comparisons[convention]:
                candidate=np.array([r['net_return_'+convention+'_pct'] for r in records if r['policy']=='M2_R1_CONTINUOUS'])
                control=np.array([r['net_return_'+convention+'_pct'] for r in records if r['policy']==name])
                delta=candidate-control;rng=np.random.default_rng(42)
                boot=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1)
                comparisons[convention][name]['mean_paired_bootstrap_95_pp']=np.quantile(boot,[.025,.975]).tolist()
        windows[window]=dict(reference_session=reference+1,terminal_session=terminal+1,signal_sessions=[31,60],policies=policies,held_at_31_fates=fate,comparisons=comparisons,records=records)
    if LOCK.read_bytes()!=original:raise ValueError('Locked evidence changed during evaluation')
    snapshot_path=ROOT/'data/audit/module2_sequential_filings_snapshot_20261026.jsonl'
    snapshot_path.write_text(''.join(json.dumps(r,default=str)+'\n' for r in snapshot),encoding='utf8')
    inputs=[LOCK,ledger_path,price_path,ROOT/'data/canonical_listed_ipos.csv',ROOT/'core/monitor_r0_frozen_20261026.py',ROOT/'services/filing_governance.py',snapshot_path,ROOT/'ml/evaluate_module2_sequential_path.py']
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),locked_evidence_sha256=lock_sha,cohort_n=132,windows=windows,
        daily_decision_rows=len(daily),checkpoint_parity=dict(r0=True,r1=True,initial_r1_holds=102,winners_retained=44,eventual_losers_retained=58),
        locked_static_reference=dict(correct=locked['module2']['results']['M2_R1_WITH_DATED_FILINGS']['correct'],n=132,
            mean_net_return_A_pct=locked['module2']['results']['M2_R1_WITH_DATED_FILINGS']['conditional_strategy_net_mean_pct']),
        index_note='Locked strategy uses Python close[31] (Session32 execution) through close[61] (Session62). Direction labels still use Session31 to Session61. Literal one-based Session31-to61 is separately reported and has a different always-sell return because next-close slippage is included.',
        cost_conventions=dict(A='0.30% multiplicative cost at each actual sell; unexited holdings marked to market at terminal close.',B='Same, with terminal 0.30% liquidation cost also charged to every unexited holding.'),
        execution_assumptions=['Next observed close execution; no intraday fills or guaranteed liquidity.','No cash interest, dividends, tax or borrow cost. Re-entry companion charges 0.30% on each buy/sell transition.','First-exit policy remains in cash; all-successive-calls is a separate optional strategy.'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limitations=['Retrospective corrected price series and archived summaries, not contemporaneously logged daily predictions.','Verified filing publication times gate every step; unavailable filings do not become neutral evidence.','Anchor calendar warnings are retained, not promoted to unverified hard exits.','No rule thresholds or model weights were selected using these outcomes.','Each IPO is evaluated at equal initial notional; this is not an overlapping, capital-constrained portfolio backtest.'])
    temporary=OUT.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False,default=json_native),encoding='utf8');temporary.replace(OUT)
    pd.DataFrame([{**r,'eligible_filing_evidence':json.dumps(r['eligible_filing_evidence'],default=json_native),'anchor_calendar':json.dumps(r['anchor_calendar'],default=json_native)} for r in daily]).to_csv(OUT.with_name(OUT.stem+'_daily_calls.csv'),index=False)
    pd.DataFrame([r for w in windows.values() for r in w['records']]).to_csv(OUT.with_name(OUT.stem+'_trades.csv'),index=False)
    lines=['# Module 2: sequential daily monitoring, not just one checkpoint','',report['index_note'],'',
        'Decisions use only closes through that session and verified filings published by 15:30 IST. Signals at Sessions 31–60 execute at the next observed close. No locked benchmark, model artifact or prediction ledger is changed.','']
    for name,w in windows.items():
        lines+=['## '+name,'', '| Policy | Mean A | Median A | Mean B | Median B | Worst-decile drawdown B |','|---|---:|---:|---:|---:|---:|']
        for policy,p in w['policies'].items():lines.append(f"| {policy} | {p['A']['mean_net_return_pct']:.3f}% | {p['A']['median_net_return_pct']:.3f}% | {p['B']['mean_net_return_pct']:.3f}% | {p['B']['median_net_return_pct']:.3f}% | {p['B']['worst_decile_mean_max_drawdown_pct']:.3f}% |")
        f=w['held_at_31_fates'];lines+=['',f"Of the 58 eventual losers initially held, {f['held_eventual_losers']['midwindow_exits']} exited later. Of the 44 eventual winners initially held, {f['held_eventual_winners']['midwindow_exits']} also exited later. These winner exits are reported with actual return forfeited or saved, rather than automatically called premature.",'']
        for label,g in f.items():
            if isinstance(g,dict):lines.append(f"- {label}: mean exited return {g['mean_exit_net_return_pct']}; same-case gross terminal hold return {g['mean_terminal_gross_hold_return_pct']}; mean benefit after comparable exit costs {g['mean_saved_return_after_comparable_exit_cost_pp']} percentage points.")
        p=w['policies'];c=p['M2_R1_CONTINUOUS']['B']['mean_net_return_pct'];a=p['ALWAYS_SELL_AT_31']['B']['mean_net_return_pct'];s=p['M2_R1_STATIC_31']['B']['mean_net_return_pct']
        lines+=['',f"Continuous R1 {'beats' if c>a else 'does not beat'} always-sell and {'beats' if c>s else 'does not beat'} static R1 on mean net return under consistent liquidation costs. The JSON compares mean, median and worst-decile drawdown separately; one improved metric is not labelled an overall edge.",'']
    lines+=['## Limits','']+['- '+r for r in report['limitations']]
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:{p:v['A']['mean_net_return_pct'] for p,v in w['policies'].items()} for k,w in windows.items()}),flush=True)
    return report

if __name__=='__main__':run()
