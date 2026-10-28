"""Deterministic gate ablations on identical 132 next-close governance outcomes."""
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from core.prove_it_monitor_engine import decision
from ml.calibrate_module2_rules import metrics
from ml.run_full_enrichment_audit import integrity


def gate_action(features,level):
    result=decision(features);gates=result['gates']
    required=['issue_and_listing_floor','trend','drawdown']
    if level>=3:required.append('relative_strength')
    if level>=4:required.append('filings_and_flow')
    if not gates:return dict(sell=None,required_gates=required,missing=required)
    return dict(sell=not all(gates[k] is True for k in required),required_gates=required,
                failed=[k for k in required if gates[k] is False],missing=[k for k in required if gates[k] is None])


def run():
    replay_path=ROOT/'output/module2_prove_it_benchmark_20261026.csv'
    replay=pd.read_csv(replay_path).set_index('ipo_id')
    ledger=pd.read_csv(ROOT/'output/module2_holdout_decision_ledger_governance_20261023.csv')
    enriched=pd.read_csv(ROOT/'output/module2_enriched_benchmark_20261025.csv').set_index('ipo_id')
    assert len(ledger)==132 and set(replay.index)==set(ledger.ipo_id)
    features=[json.loads(replay.loc[r.ipo_id,'features']) for r in ledger.itertuples()]
    labels=ledger.day30_return_pct.to_numpy()>0
    assert sum(~labels)==77 and sum(labels)==55
    old=ledger.sell.to_numpy(dtype=bool)
    # Filing-only means exactly one additional override, not the entire enriched rule engine.
    filing_bad=np.array([f.get('hard_governance_red_flag') is True or (f.get('filing_sentiment_score') is not None and f['filing_sentiment_score']<=-.30) for f in features])
    variants={'M2_R0_OLD_BASELINE':old,'M2_R1_WITH_DATED_FILINGS':old|filing_bad}
    variants.update({name:np.array([gate_action(f,level)['sell'] for f in features],dtype=bool) for name,level in
        [('M2_R2_FLOOR_AND_SMA20_GATE',2),('M2_R3_PLUS_RELATIVE_STRENGTH',3),('M2_R4_FULL_5_GATE_PROVE_IT',4)]})
    variants['ALWAYS_SELL']=np.ones(132,dtype=bool)
    results={};rows=[]
    for name,sell in variants.items():
        stats=metrics(ledger,sell)
        stats['consistent_exit_cost_net_mean_pct']=float(np.where(sell,-.30,ledger.executed_hold_return_pct-.30).mean())
        stats['winner_sensitivity_n']=int((~sell&labels).sum());stats['loser_specificity_n']=int((sell&~labels).sum())
        results[name]=stats
        for r,s,f in zip(ledger.itertuples(),sell,features):
            rows.append(dict(variant=name,ipo_id=r.ipo_id,company_name=r.company_name,decision_date=r.decision_date,sell=bool(s),
               actual_positive=bool(r.day30_return_pct>0),correct=bool(s!=(r.day30_return_pct>0)),net_strategy_return_pct=-.30 if s else r.executed_hold_return_pct,
               failed_gates=decision(f).get('failed_gates'),missing_gates=decision(f).get('missing_gates')))
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),cohort_n=132,winners=55,losers=77,results=results,
        previous_full_enriched_rule_correct=int(enriched.loc[ledger.ipo_id].enriched_correct.sum()),
        filing_only_vs_previous_enrichment='R1 isolates the filing override. The prior 63/132 included other structural/flow rules and is not labelled filing-only.',
        eligible_counts={k:sum(f.get(k) is not None for f in features) for k in ['day1_low','sma_20','rs_vs_nifty_20d','filing_sentiment_score','unabsorbed_unlock_window','delivery_accumulation_ratio']},
        protocol='31st observed close decision; next observed close execution; session31->61 direction; 0.30% sell cost. Also reports consistent exit cost on retained positions.',
        live_promoted=False,learned_classifier_used=False,integrity_preserved=integrity()['all_preserved'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [replay_path,ROOT/'output/module2_holdout_decision_ledger_governance_20261023.csv',ROOT/'core/prove_it_monitor_engine.py']},
        limitations=['Archived exchange trading dates, not download dates, govern retrospective tape admission; same-day EOD without an original publication clock cannot be presumed available at 15:30.',
          'Static allotment calendar and next-morning tape publication remain explicit archive assumptions, not verified historical publication instants.',
          'Unknown required gates cause reduce, not a claim that missing evidence is bearish.',
          'Rules selected after inspecting this cohort remain research comparisons; no live promotion from this report.'])
    path=ROOT/'output/module2_rule_ablation_20261026.json';path.write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(rows).to_csv(path.with_suffix('.csv'),index=False)
    print(json.dumps({k:{f:v[f] for f in ['correct','accuracy_pct','hold_recall_pct','sell_recall_pct','conditional_strategy_net_mean_pct']} for k,v in results.items()}),flush=True)
    return report

if __name__=='__main__':run()
