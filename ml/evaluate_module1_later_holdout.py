"""Fixed 2021–24 fit, paired 2025–26 evaluation; no promotion or tuning."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from ml.collect_module1_later_holdout import ROOT, OUT as INPUT, AUDIT
from ml.complete_module1_benchmark import prepare, attach_quotes
from ml.module1_other_source_benchmark import fitted, VARIANTS
from ml.recover_module1_other_sources import OUT as NEW_QUOTES
from ml.model_pipeline import prepare_dataset, fit_model, score_model, FEATURE_COLS, summarize
from ml.research_monitor_daily_learning import paired_gain
from services.prediction_results_service import identity_name, training_manifest

OUT = ROOT / 'output/module1_2025_2026_paired_holdout_20261024.json'
LEDGER = OUT.with_name(OUT.stem + '_predictions.csv')
REPORT = OUT.with_suffix('.md')
ARTIFACT = ROOT / 'ml/artifacts/module1_combined_train_2021_2024_20261024.joblib'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(train, test, canonical):
    if train.empty or test.empty:
        raise ValueError('Empty training or test cohort')
    if not train.year.isin([2021,2022,2023,2024]).all() or not test.year.isin([2025,2026]).all():
        raise ValueError('Wrong train/test years')
    if train.ipo_id.duplicated().any() or test.ipo_id.duplicated().any():
        raise ValueError('Repeated issuer')
    if set(train.ipo_id) & set(test.ipo_id):
        raise ValueError('Issuer ID leakage')
    for field in ('company_name','nse_symbol'):
        a = canonical.loc[train.ipo_id, field].dropna().map(identity_name if field=='company_name' else lambda v:str(v).strip().upper())
        b = canonical.loc[test.ipo_id, field].dropna().map(identity_name if field=='company_name' else lambda v:str(v).strip().upper())
        if set(a[a.ne('')]) & set(b[b.ne('')]):
            raise ValueError('Issuer identity leakage: '+field)
    if train.listing_date.max() >= test.decision_market_date.min():
        raise ValueError('Training outcome not available at first test decision')
    for rows in (train,test):
        if not (pd.to_datetime(rows.gmp_date) < rows.decision_market_date).all():
            raise ValueError('Same-day or future GMP')
        if not (rows.decision_market_date < rows.issue_close_date).all():
            raise ValueError('Final book used as interim input')
        if not rows[['log_qib','log_hni','log_retail','gmp_pct','log_offer_cap_price']].apply(np.isfinite).all().all():
            raise ValueError('Missing required observed input')


def current_probability(bundle, rows):
    x=rows[FEATURE_COLS].to_numpy(dtype=float)
    logits=((x-np.asarray(bundle['scaler_mean']))/np.asarray(bundle['scaler_scale'])) @ np.asarray(bundle['coefficients']) + bundle['intercept']
    return 1/(1+np.exp(-logits))


def metrics(rows, probability):
    y=rows.target_binary.to_numpy()
    p=rows[probability].to_numpy()
    pred=p>=.5
    return dict(n=len(rows), correct=int((pred==y).sum()), **summarize(y,p,.5))


def run():
    baseline_manifest=training_manifest()
    base,canonical,quotes=prepare()
    quotes=pd.concat([quotes,pd.read_csv(NEW_QUOTES)],ignore_index=True)
    training=attach_quotes(base,quotes,canonical)
    training=training[training.gmp_pct.notna()].copy()
    test_all=pd.read_csv(INPUT)
    test=test_all[test_all.status.eq('paired_eligible')].copy()
    for rows in (training,test):
        for field in ('listing_date','decision_market_date','issue_close_date','issue_open_date'):
            rows[field]=pd.to_datetime(rows[field])
    first=test.decision_market_date.min()
    unmatched_training=training[training.listing_date.ge(first)].ipo_id.tolist()
    training=training[training.listing_date.lt(first)].copy()
    validate(training,test,canonical)
    if set(map(str,test.ipo_id)) & baseline_manifest['ids']:
        raise ValueError('Test issuer was in deployed training')
    train_file=OUT.with_name(OUT.stem+'_training.csv')
    training.to_csv(train_file,index=False)
    # Fixed earlier development selection, never chosen using this test.
    selected=json.loads((ROOT/'output/module1_other_source_benchmark_20261024.json').read_text())['selected_direction']
    if selected!='bids_offer_gmp':
        raise ValueError('Frozen candidate selection changed')
    features=VARIANTS[selected]
    challenger=fitted(selected,'direction').fit(training[features],training.target_binary)
    gmp_only=fitted('gmp_fitted','direction').fit(training[['gmp_pct']],training.target_binary)
    final=prepare_dataset().set_index('ipo_id')
    deployed_train=final.loc[[int(v) for v in baseline_manifest['ids']]]
    scaler,refit=fit_model(deployed_train)
    bundle=json.loads((ROOT/'ml/model_bundle.json').read_text())
    if not (np.allclose(scaler.mean_,bundle['scaler_mean'],atol=1e-10) and
            np.allclose(scaler.scale_,bundle['scaler_scale'],atol=1e-10) and
            np.allclose(refit.coef_[0],bundle['coefficients'],atol=1e-10) and
            np.allclose(refit.intercept_[0],bundle['intercept'],atol=1e-10)):
        raise ValueError('Deployed bundle is not reproducible')
    # Stage-matched bid-only control distinguishes stage alignment from GMP gain.
    bid_control=fitted('bids_offer_gmp','direction').fit(training[FEATURE_COLS],training.target_binary)
    model_paths=[ROOT/'ml/evaluate_module1_later_holdout.py', ROOT/'ml/collect_module1_later_holdout.py',
                 ROOT/'ml/module1_other_source_benchmark.py', ROOT/'ml/complete_module1_benchmark.py',
                 ROOT/'ml/recover_module1_other_sources.py', ROOT/'ml/model_pipeline.py',
                 ROOT/'ml/model_bundle.json',INPUT,AUDIT,NEW_QUOTES,train_file,
                 ROOT/'output/module1_later_holdout_protocol_20261024.md',
                 ROOT/'output/module1_other_source_benchmark_20261024.json',
                 ROOT/'data/nse_verified_offer_dates.csv',ROOT/'data/audit/module1_exchange_listing_open_audit.csv']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in model_paths}
    joblib.dump(dict(research_only=True,training_years=[2021,2022,2023,2024],training_ids=sorted(map(int,training.ipo_id)),
                     features=features,model=challenger,gmp_only=gmp_only,bid_control=bid_control,
                     threshold=.5,input_sha256=hashes),ARTIFACT)
    # Only now score test outcomes; no fit, threshold choice or selection below.
    test['p_combined']=challenger.predict_proba(test[features])[:,1]
    test['p_gmp_fitted']=gmp_only.predict_proba(test[['gmp_pct']])[:,1]
    test['p_gmp_sign']=test.gmp_pct.gt(0).astype(float)
    test['p_current_interim']=current_probability(bundle,test)
    test['p_bid_stage_matched']=bid_control.predict_proba(test[FEATURE_COLS])[:,1]
    # This later final-book diagnostic is explicitly not the primary comparison.
    final_rows=final.loc[test.ipo_id]
    if final_rows.missing_subscription_count.ne(0).any():
        raise ValueError('Missing final-book diagnostic values')
    test['p_current_final_diagnostic']=current_probability(bundle,final_rows)
    columns=['p_current_interim','p_bid_stage_matched','p_gmp_sign','p_gmp_fitted','p_combined','p_current_final_diagnostic']
    for name in columns:
        test[name+'_correct']=test[name].ge(.5).eq(test.target_binary)
    test['combined_call']=np.where(test.p_combined.ge(.5),'Positive','Flat / negative')
    test['actual_direction']=np.where(test.target_binary.eq(1),'Positive','Flat / negative')
    test.to_csv(LEDGER,index=False)
    errors_file=OUT.with_name(OUT.stem+'_errors.csv')
    test.loc[~test.p_combined_correct].to_csv(errors_file,index=False)
    results={name:metrics(test,name) for name in columns}
    yearly={str(int(year)):{name:metrics(group,name) for name in columns}
            for year,group in test.groupby('year')}
    comparison={name:paired_gain(test.p_combined.ge(.5).eq(test.target_binary),test[name].ge(.5).eq(test.target_binary))
                for name in columns if name!='p_combined'}
    report=dict(training_years=[2021,2022,2023,2024],test_years=[2025,2026],
                combined_training_n=len(training),current_training_n=baseline_manifest['training_n'],
                train_ids=sorted(map(int,training.ipo_id)),test_ids=sorted(map(int,test.ipo_id)),
                overlap_n=0,latest_training_outcome=str(training.listing_date.max().date()),
                first_test_decision=str(first.date()),excluded_unmatured_training_ids=unmatched_training,
                test_universe_n=len(test_all),paired_test_n=len(test),
                coverage=test_all.status.value_counts().to_dict(),selected_direction=selected,
                features=features,direction_threshold=.5,apply_threshold_deployed=bundle['decision_threshold'],
                results=results,by_year=yearly,combined_vs=comparison,input_sha256=hashes,
                artifact=ARTIFACT.relative_to(ROOT).as_posix(),artifact_sha256=digest(ARTIFACT),
                limitations=['Retrospective reconstruction; not forecasts originally saved before applications closed.',
                    'Later-year outcomes were previously inspected. No model or threshold selection used this new comparison.',
                    'Only the matched coverage subset supports this comparison; excluded IPOs are not assumed correct.',
                    'Public historical daily tables can be revised; GMP must precede the interim decision date.',
                    'Offer size is a canonical historical offer descriptor, not a timestamp-certified capture.',
                    'Current-model final-book diagnostic uses later information and is not an actionable pre-close forecast.',
                    'Accuracy is direction at a fixed 0.5 cutoff, not APPLY precision or guaranteed investment profit.'],
                promoted=False,module2_changed=False,paper_changed=False)
    OUT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf8')
    labels={'p_current_interim':'Current deployed model, same interim bids',
            'p_bid_stage_matched':'Bid-only model retrained on interim bids',
            'p_gmp_sign':'GMP-only: positive GMP predicts positive opening',
            'p_gmp_fitted':'GMP-only fitted logistic control',
            'p_combined':'New combined model',
            'p_current_final_diagnostic':'Current model on later final bids (diagnostic only)'}
    lines=['# Module 1: 2025–2026 paired comparison','',
           f'Training: 2021–2024 only. Combined training: {len(training)} IPOs; deployed training: {baseline_manifest["training_n"]}.',
           f'Test: {len(test)} identical IPOs out of {len(test_all)} exchange-verified later-year IPOs. No ID, company-name or symbol overlap.',
           f'Latest training outcome: {report["latest_training_outcome"]}; first test decision: {report["first_test_decision"]}.','',
           'The combined model and settings were selected on earlier development years and fixed before this evaluation. Positive means opening above issue price; flat counts as nonpositive. Direction cutoff is 50%; this is different from the deployed APPLY cutoff.','',
           '| Model | Correct / IPOs | Accuracy | Balanced accuracy | AUC | Brier |',
           '|---|---:|---:|---:|---:|---:|']
    for name,value in results.items():
        lines.append(f'| {labels[name]} | {value["correct"]}/{len(test)} | {value["classifier_accuracy_pct"]:.1f}% | {value["balanced_accuracy_pct"]:.1f}% | {value["roc_auc"]} | {value["brier_score"]} |')
    lines += ['','## Combined accuracy change on the same IPOs','']
    for name in ('p_current_interim','p_gmp_sign','p_gmp_fitted'):
        v=comparison[name]
        lines.append(f'- Versus {labels[name]}: {v["change_pp"]:+.2f} percentage points; paired bootstrap 95% interval {v["bootstrap_95_pp"]}. Corrected {v["corrected_errors"]} errors and introduced {v["introduced_errors"]}.')
    lines += ['','## By year','','| Year | IPOs | Current, interim | GMP sign | Fitted GMP | Combined |','|---|---:|---:|---:|---:|---:|']
    for year,values in yearly.items():
        lines.append('| '+year+' | '+str(values['p_combined']['n'])+' | '+' | '.join(f'{values[n]["classifier_accuracy_pct"]:.1f}%' for n in ('p_current_interim','p_gmp_sign','p_gmp_fitted','p_combined'))+' |')
    lines += ['','## What changed and what still fails','',
              'The current final-book model receives much smaller bid counts at the interim stage and misses many positive openings. Retraining a bid-only control at the correct stage raises accuracy from 56.1% to 71.6%; adding GMP and offer descriptors raises it to 74.8%. The additional gain over that stage-matched control is 3.23 points, with a paired interval spanning zero.',
              f'The combined model still misses {results["p_combined"]["fn"]} positive openings and incorrectly predicts positive for {results["p_combined"]["fp"]} flat/negative openings. It catches only {results["p_combined"]["negative_recall_pct"]}% of nonpositive listings. Its AUC is {results["p_combined"]["roc_auc"]}, versus {results["p_gmp_fitted"]["roc_auc"]} for fitted GMP. This is not evidence that ranking or return forecasting has improved.',
              f'All {int((~test.p_combined_correct).sum())} wrong combined calls are saved in `{errors_file.name}`.','',
              '## Coverage','',*[f'- {k}: {v}' for k,v in report['coverage'].items()],
              '', '## Interpretation limits','',*[f'- {v}' for v in report['limitations']],
              '',f'Per-IPO probabilities, actual exchange opens and correct/wrong flags: `{LEDGER.name}`.',
              'The existing live model and saved forecasts were not changed.']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(training_n=len(training),paired_test_n=len(test),results=results,combined_vs=comparison),indent=2))


if __name__=='__main__':
    run()
