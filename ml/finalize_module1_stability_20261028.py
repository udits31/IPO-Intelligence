"""Pre-2026 selection and confirmation of a fixed-0.5 scoring policy."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml.test_historical_joins_and_cutoff_20261028 import ROOT,SOURCE,core,EXPERTS,logits,sha,threshold_metrics
from ml.test_through2025_cutoff_20261028 import ARTIFACT,check_split

OUT=ROOT/'output/module1_stability_lock_20261028'
POLICIES=['incumbent_C05','regularized_C005','equal_expert_probability_mean']


def fit_policy(policy,fit,held):
    if policy=='equal_expert_probability_mean':return held[EXPERTS].to_numpy().mean(axis=1),None
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.5 if policy=='incumbent_C05' else .05,max_iter=3000,random_state=42))
    model.fit(logits(fit[EXPERTS]),fit.target_binary.astype(int))
    return model.predict_proba(logits(held[EXPERTS]))[:,1],model


def select_and_confirm(validation):
    if set(validation.year)!={2024,2025}:raise ValueError('Only 2024 selection and 2025 confirmation are allowed')
    table={}
    for policy in POLICIES:
        table[policy]={str(int(year)):threshold_metrics(g,g[policy].to_numpy(),.5) for year,g in validation.groupby('year')}
    base=table[POLICIES[0]]
    eligible=[p for p in POLICIES if table[p]['2024']['classifier_accuracy_pct']>=base['2024']['classifier_accuracy_pct'] and table[p]['2024']['balanced_accuracy_pct']>=base['2024']['balanced_accuracy_pct']]
    chosen=min(eligible,key=lambda p:(table[p]['2024']['brier_score'],POLICIES.index(p)))
    confirmed=(table[chosen]['2025']['classifier_accuracy_pct']>=base['2025']['classifier_accuracy_pct'] and table[chosen]['2025']['balanced_accuracy_pct']>=base['2025']['balanced_accuracy_pct'] and table[chosen]['2025']['brier_score']<=base['2025']['brier_score'])
    return dict(selection=chosen,confirmation_passed=confirmed,locked_policy=chosen if confirmed else POLICIES[0],validation=table)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(policies=POLICIES,fixed_cutoff=.5,selection_year=2024,confirmation_year=2025,test_year=2026,
        selection='2024 minimum Brier subject to accuracy and balanced accuracy no worse than incumbent; stable tie order',
        confirmation='2025 accuracy and balanced accuracy no worse, Brier no worse; otherwise retain incumbent',
        inputs='Identical numeric-audited baseline feature joins; no new extraction or calibration',
        training='Expanding causal base-expert OOF and meta learner; final training 2021–2025',
        promotion='Do not deploy a new candidate unless pre-2026 confirmation passes and same69 accuracy/false-positive count is no worse than actual frozen artifact; any change needs future prospective confirmation',
        inspected_test_warning='2026 repeatedly inspected during development; not an independent new holdout')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    original=sha(ARTIFACT)
    data=pd.read_csv(SOURCE);data=data[data.cohort.eq('historical')].copy()
    for k in ['listing_date','decision_market_date']:data[k]=pd.to_datetime(data[k])
    train=data[data.year.le(2025)].copy();test=data[data.year.eq(2026)].copy().reset_index(drop=True);check_split(train,test)
    core.SETS['gmp_only']=['gmp_pct']
    candidates=core.candidates()+[dict(name='gmp_only_control',family='logistic',C=.5,class_weight=None,transform='raw',feature_set='gmp_only')]
    configs={name:next(c for c in candidates if c['name']==name) for name in EXPERTS}
    oof=[];trace=[]
    for year in range(2022,2026):
        held=train[train.year.eq(year)].copy();fit=train[train.year.lt(year)&train.listing_date.lt(held.decision_market_date.min())]
        assert set(fit.ipo_id).isdisjoint(held.ipo_id)
        for name,c in configs.items():
            model=core.estimator(c).fit(core.features(fit,c),fit.target_binary)
            held[name]=model.predict_proba(core.features(held,c))[:,1]
        oof.append(held);trace.append(dict(year=year,train_ids=fit.ipo_id.tolist(),held_ids=held.ipo_id.tolist()))
    oof=pd.concat(oof,ignore_index=True)
    oof.to_csv(OUT/'base_expert_causal_oof.csv',index=False)
    validation=[]
    for year in [2024,2025]:
        held=oof[oof.year.eq(year)].copy();fit=oof[oof.year.lt(year)]
        if fit.listing_date.max()>=held.decision_market_date.min():raise ValueError('Meta outcome timing leakage')
        for policy in POLICIES:held[policy],_=fit_policy(policy,fit,held)
        validation.append(held[['ipo_id','year','target_binary',*POLICIES]])
    validation=pd.concat(validation,ignore_index=True);validation.to_csv(OUT/'pre2026_validation.csv',index=False)
    decision=select_and_confirm(validation)
    # Freeze selection before producing any 2026 policy scores.
    (OUT/'pre2026_decision.json').write_text(json.dumps(decision,indent=2))
    models={}
    for name,c in configs.items():
        models[name]=core.estimator(c).fit(core.features(train,c),train.target_binary)
        test[name]=models[name].predict_proba(core.features(test,c))[:,1]
    results={}
    for policy in POLICIES:
        test[policy],meta=fit_policy(policy,oof,test)
        results[policy]={scope:threshold_metrics(g,g[policy].to_numpy(),.5) for scope,g in [('same69',test),('same64_verified_day2',test[test.verified_stage.eq('day2')])]}
        joblib.dump(dict(models=models,meta=meta,policy=policy,threshold=.5,configs=configs,training_ids=train.ipo_id.tolist(),training_end_year=2025,research_only=True),OUT/(policy+'.joblib'))
    artifact=joblib.load(ARTIFACT)
    # Score the actual deployment's own expert fits, not the research refit's.
    raw=np.column_stack([artifact['models'][name].predict_proba(core.features(test,artifact['expert_configs'][name]))[:,1] for name in artifact['experts']])
    test['actual_frozen_artifact']=artifact['meta_model'].predict_proba(logits(raw))[:,1]
    frozen=threshold_metrics(test,test.actual_frozen_artifact.to_numpy(),.5)
    chosen=results[decision['locked_policy']]['same69']
    promoted=decision['locked_policy']!=POLICIES[0] and decision['confirmation_passed'] and chosen['correct']>=frozen['correct'] and chosen['fp']<=frozen['fp']
    report=dict(protocol=protocol,pre2026_decision=decision,results=results,actual_frozen=frozen,
        recommendation='retain_existing_frozen_artifact' if not promoted else 'candidate_passed_research_gate_review_before_live_change',
        candidate_research_gate_passed=promoted,training_n=len(train),test_n=len(test),artifact_sha256=original,
        artifact_unchanged=sha(ARTIFACT)==original,live_changes=False,trace=trace)
    test[['ipo_id','company_name','year','verified_stage','target_binary',*POLICIES,'actual_frozen_artifact']].to_csv(OUT/'paired_predictions.csv',index=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Module 1 stability and release decision','',
        'Fixed cutoff 0.50 for all candidates. Policies selected on 2024 and confirmed on 2025 before computing 2026 scores. Final training: 290 IPOs from 2021–2025. Test: identical 69 IPOs from 2026.','',
        '| Scoring policy | 2024 accuracy | 2025 accuracy | 2026 correct /69 | 2026 false positives |','|---|---:|---:|---:|---:|']
    for policy in POLICIES:
        v=decision['validation'][policy];r=results[policy]['same69']
        lines.append(f"| {policy} | {v['2024']['classifier_accuracy_pct']}% | {v['2025']['classifier_accuracy_pct']}% | {r['correct']}/69 ({r['classifier_accuracy_pct']}%) | {r['fp']} |")
    lines+=['',f"Selected before 2026: {decision['selection']}. Separate 2025 confirmation passed: {decision['confirmation_passed']}. Locked pre-test policy: {decision['locked_policy']}.",
        f"Actual frozen artifact: {frozen['correct']}/69 ({frozen['classifier_accuracy_pct']}%), {frozen['fp']} false positives.",
        f"Recommendation: {report['recommendation']}. No deployment or ledger changes.",
        '', 'A fixed threshold belongs to a fixed versioned scoring artifact. No daily threshold adjustment or automatic retraining is permitted by this release decision. New retraining creates a separate challenger, checked on chronological validation and new prospective outcomes before any future promotion.',
        '',protocol['inspected_test_warning']]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))


if __name__=='__main__':run()
