"""Rebuild presentation-facing evidence from the active models and real cases.

No live model is trained or promoted here. Candidate fits are retrospective
research; all later comparisons share exact exchange-verified opening labels.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score

from ml.model_pipeline import prepare_dataset,fit_model,score_model,summarize,FEATURE_COLS
from ml.model_search import candidates
from ml.audit_module1_listing_opens import verified_outcomes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/performance_evidence.json'
M2=ROOT/'output/module2_holdout_accuracy_2025_2026_governance_20261023.json'
M2_LEDGER=ROOT/'output/module2_holdout_decision_ledger_governance_20261023.csv'
LEGACY_M2=ROOT/'output/module2_holdout_accuracy_2025_2026presentation_20261022.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    frame=prepare_dataset()
    frame=frame[~frame.id.isin([293,307])].copy()
    frame['log_issue_size']=np.log1p(frame.issue_size_cr.clip(lower=0))
    frame['log_issue_price']=np.log1p(frame.issue_price.clip(lower=0))
    bundle=json.loads((ROOT/'ml/model_bundle.json').read_text(encoding='utf-8'))
    for name,digest in bundle['input_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Active model data no longer matches its recorded input hashes')
    audit=verified_outcomes()
    later=frame[frame.year.isin([2025,2026])].merge(audit[['ipo_id','exchange_open']],left_on='id',right_on='ipo_id',validate='one_to_one')
    y=later.exchange_open.gt(later.issue_price).to_numpy(dtype=int)
    train=frame[frame.year.between(2021,2024)]
    scaler,model=fit_model(train)
    if not (np.allclose(scaler.mean_,bundle['scaler_mean'],rtol=1e-12) and
            np.allclose(scaler.scale_,bundle['scaler_scale'],rtol=1e-12) and
            np.allclose(model.coef_[0],bundle['coefficients'],rtol=1e-12) and
            np.isclose(model.intercept_[0],bundle['intercept'],rtol=1e-12)):
        raise ValueError('Exact replay differs from active model coefficients')
    p=score_model(scaler,model,later)
    predicted={};development={}
    for name,(features,factory) in candidates().items():
        early_y=[];early_p=[]
        for year in (2022,2023):
            before=frame[frame.year.between(2021,year-1)]
            test=frame[frame.year.eq(year)]
            fitted=factory().fit(before[features],before.target_binary)
            early_y.extend(test.target_binary.to_numpy())
            early_p.extend(fitted.predict_proba(test[features])[:,1])
        development[name]={'y':np.asarray(early_y),'p':np.asarray(early_p)}
        fitted=factory().fit(train[features],train.target_binary)
        predicted[name]=fitted.predict_proba(later[features])[:,1]
    blends={'blend_lr_hgb':('lr3_c05','hgb3_d2',.7),'blend_lr_xgb':('lr3_c05','xgb3_d1',.7),
            'blend_lr3_lr5':('lr3_c05','lr5_c1',.5),'blend_lr3_poly':('lr3_c05','poly3_c005',.5)}
    for name,(left,right,weight) in blends.items():
        predicted[name]=weight*predicted[left]+(1-weight)*predicted[right]
        development[name]={'y':development[left]['y'],
            'p':weight*development[left]['p']+(1-weight)*development[right]['p']}
    np.testing.assert_allclose(predicted['lr3_c05'],p,rtol=1e-10)
    rng=np.random.default_rng(42)
    positive,negative=np.flatnonzero(y),np.flatnonzero(1-y)
    draws=[np.r_[rng.choice(positive,len(positive)),rng.choice(negative,len(negative))] for _ in range(1000)]
    base_auc=np.array([roc_auc_score(y[ix],p[ix]) for ix in draws])
    comparison=[]
    for name,probabilities in predicted.items():
        scores=np.array([roc_auc_score(y[ix],probabilities[ix]) for ix in draws])
        ci=np.quantile(scores-base_auc,[.025,.975]).tolist()
        comparison.append({'candidate':name,'auc':float(roc_auc_score(y,probabilities)),
            'development_auc':float(roc_auc_score(development[name]['y'],development[name]['p'])),
            'delta_vs_retained_lr_auc':float(roc_auc_score(y,probabilities)-roc_auc_score(y,p)),
            'paired_delta_95_interval':ci,'reliable_improvement_demonstrated':bool(ci[0]>0)})
    comparison.sort(key=lambda row:(-row['development_auc'],row['candidate']))
    ledger=later[['id','company_name','year','issue_price','exchange_open']].copy()
    ledger['actual_positive']=y
    ledger['retained_probability']=p
    for name,probabilities in predicted.items():
        ledger[name]=probabilities
    ledger_path=OUT.with_name('performance_model_comparison_ledger.csv')
    ledger.to_csv(ledger_path,index=False)
    m2=json.loads(M2.read_text(encoding='utf-8'))
    for name,digest in m2['input_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Current Module 2 replay inputs changed; rerun its actual rule engine')
    market_path=ROOT/'output/module2_market_rule_calibration_20261021.json'
    market=json.loads(market_path.read_text(encoding='utf-8'))
    legacy=json.loads(LEGACY_M2.read_text(encoding='utf-8'))
    if market['later']['default']['n']!=legacy['day30']['n'] or not np.isclose(
            market['later']['default']['accuracy_pct'],legacy['day30']['model_accuracy_pct']):
        raise ValueError('Archived improvement timeline must match its original policy and cases')
    current_ledger=pd.read_csv(M2_LEDGER)
    legacy_ledger=pd.read_csv(ROOT/'output/module2_holdout_decision_ledgerpresentation_20261022.csv')
    if set(current_ledger.ipo_id)!=set(legacy_ledger.ipo_id):
        raise ValueError('Policy refresh must retain the same historical cases')
    review_count=int(current_ledger.verdict.str.contains('WATCH',case=False).sum())
    result={'schema_version':1,'built_at_utc':datetime.now(timezone.utc).isoformat(),
        'runtime_versions':{'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__},
        'module1':{'model_version':bundle['model_version'],'training_n':len(train),'features':['QIB','Retail','HNI'],
            'later_n':len(later),'requested_later_n':int(frame.year.isin([2025,2026]).sum()),
            'auc_95_interval':np.quantile(base_auc,[.025,.975]).tolist(),
            'direction':summarize(y,p,float(bundle['benchmark']['direction_threshold'])),
            'application_policy':summarize(y,p,float(bundle['decision_threshold'])),
            'protocol':'2021–2024 training; 2025–2026 retrospective exchange-verified openings. Final subscription books, NOT pre-application accuracy.',
            'conclusion':'Useful retrospective risk separation; a high APPLY precision is not overall accuracy or an application-time success guarantee.'},
        'complexity':{'candidate_count':len(comparison),'rows':comparison,'retained_candidate':'lr3_c05',
            'auc_min':min(r['auc'] for r in comparison),'auc_max':max(r['auc'] for r in comparison),
            'reliable_improvement_demonstrated':any(r['reliable_improvement_demonstrated'] for r in comparison),
            'protocol':'Same original 252 corporate training records and identical 179 verified later outcomes. Earlier-year 2022/2023 folds order candidates; later outcomes do not select a deployment.',
            'limitations':'Paired percentile intervals are descriptive, not multiplicity-adjusted equivalence tests. Earlier and later outcomes have already been inspected.'},
        'module2':{'coverage':m2['coverage'],'day30':m2['day30'],'first_day':m2['first_day'],
            'protocol':m2['decision']+'; '+m2['outcome'],'limitations':m2['limitations'],
            'policy_note':f'Current risk-alert policy replay: {review_count} WATCH/review calls are treated as no exit (hold) only for this binary back-test; they are not affirmative HOLD recommendations. News and disclosures are not evaluated in this price-only test.',
            'review_calls':review_count,
            'conclusion':'No demonstrated 30-session direction edge. Present this as a transparent risk heuristic, not a validated forecast.'},
        'improvements':[
            {'title':'Richer pre-close output','type':'Capability, not proven accuracy gain',
             'detail':'Added gain/loss estimates, downside/upside scenarios and saved priority comparisons against GMP-only. No selection advantage established.'},
            {'title':'Exchange-verified outcomes','type':'Evidence correction, not a new model',
             'detail':'Rechecked listing openings against exact exchange archives; unchanged probabilities are rescored without rewriting the original predictions.'},
            {'title':'Comparable share-price units','type':'Correctness improvement',
             'detail':'Reconciled explicit bonus/split ratios, restored supported histories, and prevented double adjustment of provider prices.'},
            {'title':'Issuer-specific news','type':'Input-quality improvement',
             'detail':'Rejected shared-brand/company mismatches and fixed profit-decline sentiment; the dated test did not show a binary hold/sell accuracy gain.'},
        ],
        'tested_accuracy_changes':[{'title':'Market-aware thresholds · Module 2 research',
            'change':'Archived experiment against the earlier automatic-sell policy, not the current risk-alert policy. Keep the earlier loss/trailing rules, but choose tighter or wider thresholds from the observed Nifty regime. Settings selected only on matured pre-2025 outcomes.',
            'before_pct':market['later']['default']['accuracy_pct'],
            'after_pct':market['later']['market_rules']['accuracy_pct'],
            'n':market['later']['market_rules']['n'],
            'baseline_pct':market['later']['market_rules']['always_sell_accuracy_pct'],
            'paired_change':market['later']['paired_change'],
            'status':'Earlier-policy research · not promoted',
            'verdict':'The sample score increased, but it still loses to always-sell and its change interval includes zero. Do not claim a reliable live accuracy gain.'}],
        'input_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in [
            ROOT/'ml/model_bundle.json',ROOT/'data/canonical_listed_ipos.csv',ROOT/'data/subscription_observations.csv',
            ROOT/'data/audit/module1_exchange_listing_open_audit.csv',M2,M2_LEDGER,
            ROOT/'ml/model_pipeline.py',ROOT/'ml/model_search.py',Path(__file__),ROOT/'core/monitor_engine.py',ROOT/'core/attribution_engine.py',ledger_path,market_path,LEGACY_M2]},
        'claims_policy':'Never substitute direction accuracy for APPLY precision, retrospective evidence for live forecasts, or one decision horizon for another.'}
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    policy=result['module1']['application_policy'];direction=result['module1']['direction']
    change=result['tested_accuracy_changes'][0]
    notes=['# Performance page — presenter notes','',
        '## What Module 1 supports','',
        f"On the same {len(later)} exchange-verified 2025–2026 outcomes, the retained three-input logistic model has AUC {direction['roc_auc']:.3f}. "
        f"At its {policy['threshold']*100:.0f}% APPLY threshold, precision is {policy['apply_precision_pct']:.1f}% and overall policy accuracy is {policy['classifier_accuracy_pct']:.1f}%, "
        f"versus {policy['always_positive_baseline_pct']:.1f}% for always predicting positive. "
        f"The separate 50% direction cutoff has {direction['classifier_accuracy_pct']:.1f}% accuracy. Do not interchange those metrics.",'',
        'These are retrospective final-book results, not the accuracy of the application-time return model.','',
        '## Negative results','',
        f"We reran {len(comparison)} candidates on the same training and later cases. Later AUCs range from {result['complexity']['auc_min']:.3f} to {result['complexity']['auc_max']:.3f}. "
        'No paired AUC improvement over the retained logistic model was established. Small differences alone do not prove equivalence.','',
        f"Module 2: {m2['day30']['n']} matured cases, {m2['day30']['model_accuracy_pct']:.1f}% 30-session direction accuracy, "
        f"versus {m2['day30']['always_sell_accuracy_pct']:.1f}% always sell. It remains a transparent risk heuristic, not a validated forecast.",'',
        '## How to discuss improvement honestly','',
        f"An earlier-data-selected market-regime threshold experiment increased the sample score from {change['before_pct']:.1f}% to {change['after_pct']:.2f}% on the same {change['n']} cases. "
        f"It corrected {change['paired_change']['corrected_errors']} errors and introduced {change['paired_change']['introduced_errors']}. "
        'Its change interval includes zero and it still loses to the baseline, so it was not promoted. This is research progress, not a proven live gain.','',
        'Split/bonus reconciliation, exchange-opening verification, stricter issuer/news matching and pre-close priority logs improve correctness and evidence. They must not be described as measured accuracy gains.','',
        '## Design challenges and next steps','',
        'Small sample, look-ahead leakage, missing/interim bids, capital actions, incomplete disclosures/news and retrospective evidence remain explicit. '
        'Next: genuine pre-cutoff prospective captures, calibrated probabilities, same-case significance tests, cost-aware monitoring and production access/storage/collector hardening. '
        'Do not claim formal ISO/IEEE conformance or invent a bibliography count.','',
        'The current application and full per-candidate ledger contain the detailed evidence. No paper title, paper body or model bundle was changed by this presentation work.']
    OUT.with_name('performance_presenter_notes_20261022.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print(json.dumps({'module1':result['module1'],'complexity':{k:v for k,v in result['complexity'].items() if k!='rows'},
                      'module2':{'n':m2['day30']['n'],'accuracy':m2['day30']['model_accuracy_pct']}},indent=2))
    return result


if __name__=='__main__':
    run()
