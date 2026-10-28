"""Matched 2026 research comparison. Never promotes or rewrites live artifacts."""
import json
import joblib
import pandas as pd
from ml.test_historical_joins_and_cutoff_20261028 import (
    ROOT, SOURCE, sha, core, EXPERTS, fit_candidate, threshold_metrics,
)
from services.historical_bid_service import name_key

OUT = ROOT/'output/through2025_cutoff_test_20261028'
ARTIFACT = ROOT/'ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib'


def check_split(train, test):
    if not train.year.between(2021,2025).all() or not test.year.eq(2026).all():
        raise ValueError('Wrong training/test years')
    if train.ipo_id.duplicated().any() or test.ipo_id.duplicated().any():
        raise ValueError('Duplicate issuer')
    if set(train.ipo_id)&set(test.ipo_id):raise ValueError('Issuer ID leakage')
    if set(train.company_name.map(name_key))&set(test.company_name.map(name_key)):
        raise ValueError('Issuer name leakage')
    if train.listing_date.max()>=test.decision_market_date.min():
        raise ValueError('Training outcome unavailable at test decision')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(training_years=list(range(2021,2026)),test_year=2026,
        cutoff_selection_year=2024,confirmation_year=2025,
        threshold_grid_start=.35,threshold_grid_end=.85,threshold_grid_step=.025,
        objective='balanced accuracy, then accuracy, then closest to 0.5',
        architecture='same five experts and C=0.5 standardized log-odds stack',
        prior_fixed_cutoff=.625,
        test_labels_used_for_selection=False,live_promotion=False,
        note='Previously examined retrospective cohorts; no independent prospective claim.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    original_sha=sha(ARTIFACT)
    core.SETS['gmp_only']=['gmp_pct']
    candidates=core.candidates()+[dict(name='gmp_only_control',family='logistic',C=.5,class_weight=None,transform='raw',feature_set='gmp_only')]
    configs={name:next(c for c in candidates if c['name']==name) for name in EXPERTS}
    report=dict(protocol=protocol,source_sha256=sha(SOURCE),deployed_artifact_sha256=original_sha,variants={},cohorts={})
    for label,path in [('baseline',SOURCE),('financial_join',ROOT/'output/historical_join_cutoff_test_20261028/enriched_derived_features.csv')]:
        data=pd.read_csv(path);data=data[data.cohort.eq('historical')].copy()
        for key in ['listing_date','decision_market_date']:data[key]=pd.to_datetime(data[key])
        train=data[data.year.le(2025)].copy().reset_index(drop=True)
        test=data[data.year.eq(2026)].copy().reset_index(drop=True)
        check_split(train,test)
        p,cutoff,confirmation,validation,bundle=fit_candidate(train,test,configs,train_end_year=2025,selection_year=2024)
        validation.to_csv(OUT/(label+'_pre2026_validation.csv'),index=False)
        train[['ipo_id','company_name','year','listing_date','decision_market_date']].to_csv(OUT/(label+'_training_membership.csv'),index=False)
        joblib.dump(dict(**bundle,training_ids=train.ipo_id.tolist(),training_end_year=2025,research_only=True,threshold=cutoff['threshold']),OUT/(label+'_research.joblib'))
        test['score']=p;test['selected_cutoff']=cutoff['threshold']
        test[['ipo_id','company_name','year','verified_stage','target_binary','target_return','score','selected_cutoff']].to_csv(OUT/(label+'_test_predictions.csv'),index=False)
        report['cohorts'][label]=dict(train_n=len(train),test_n=len(test),train_year_counts=train.year.value_counts().sort_index().to_dict(),training_earnings_observed=int(train.pub_ipo_earnings_yield_pct.notna().sum()),training_pricebook_observed=int(train.pub_log_ipo_price_book.notna().sum()))
        for suffix,threshold in [('fixed50',.5),('prior_fixed625',.625),('validated_cutoff',cutoff['threshold'])]:
            report['variants'][label+'_'+suffix]=dict(threshold=threshold,selection_2024=cutoff,confirmation_2025=confirmation,
                metrics={scope:threshold_metrics(rows,rows.score.to_numpy(),threshold) for scope,rows in [('all69',test),('same64_clean_day2',test[test.verified_stage.eq('day2')])]})
    assert sha(ARTIFACT)==original_sha,'Live artifact changed unexpectedly'
    report['live_artifact_unchanged']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Training through 2025: matched 2026 comparison','',
        '290 training IPOs from 2021–2025. Cutoff selected on out-of-time 2024 predictions, confirmed on 2025, then fixed for 2026. No model promotion.','',
        '| Variant | Cutoff | All 69 in 2026 | Same 64 stage-verified IPOs | Flat/negative caught (64) | Positive caught (64) |',
        '|---|---:|---:|---:|---:|---:|']
    for label,v in report['variants'].items():
        a=v['metrics']['all69'];b=v['metrics']['same64_clean_day2']
        lines.append(f"| {label} | {v['threshold']:.3f} | {a['correct']}/{a['sample_size']} ({a['classifier_accuracy_pct']}%) | {b['correct']}/{b['sample_size']} ({b['classifier_accuracy_pct']}%) | {b['tn']}/{b['nonpositive_listings']} | {b['tp']}/{b['positive_listings']} |")
    lines+=['','The all-69 sensitivity includes five records without verified Day-2 stage alignment; the same-64 comparison excludes those identical records from every variant.',
        'These are previously examined historical cohorts. The updated through-2025 models are causal research refits and are separate from the deployed artifact.',
        '', '## Pre-2026 confirmation', '']
    for label,v in report['variants'].items():
        if label.endswith('validated_cutoff'):
            m=v['confirmation_2025'];lines.append(f"{label}: 2025 confirmation accuracy {m['classifier_accuracy_pct']}%, balanced accuracy {m['balanced_accuracy_pct']}%, cutoff {v['threshold']:.3f}.")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':run()
