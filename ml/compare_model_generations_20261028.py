"""Research-only matched model comparison; never changes deployment or ledgers."""
from pathlib import Path
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml import model_pipeline as legacy
from ml import research_module1_forward_update as core
from ml.research_module1_stacking import EXPERTS, logits, stacker
from services.historical_bid_service import name_key

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/model_comparison_20261028'
SOURCE=OUT/'numeric_audited_features.csv'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(train,test):
    if not train.year.between(2021,2024).all() or not test.year.between(2025,2026).all():
        raise ValueError('Wrong year split')
    if train.ipo_id.duplicated().any() or test.ipo_id.duplicated().any():raise ValueError('Duplicate issuer IDs')
    if set(train.ipo_id)&set(test.ipo_id):raise ValueError('Issuer ID leakage')
    if set(train.company_name.map(name_key))&set(test.company_name.map(name_key)):raise ValueError('Issuer-name leakage')
    if train.listing_date.max()>=test.decision_market_date.min():raise ValueError('Outcome available after first test decision')


def legacy_features(frame):
    x=frame.copy()
    for source,destination in [('qib_subscription','log_qib'),('retail_subscription','log_retail'),('hni_subscription','log_hni')]:
        if x[source].isna().any() or (x[source]<0).any():raise ValueError('Unobserved category passed to legacy model')
        x[destination]=np.log1p(x[source].clip(lower=.01))
    return x


def ridge():
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=20))


def metrics(rows,p):
    y=rows.target_binary.astype(int).to_numpy();m=legacy.summarize(y,p,.5)
    m['correct']=int(((p>=.5)==y).sum());return m


def paired(rows,a,b):
    y=rows.target_binary.astype(int).to_numpy();ca=(a>=.5)==y;cb=(b>=.5)==y
    wins=int((ca&~cb).sum());losses=int((~ca&cb).sum())
    rng=np.random.default_rng(42);difference=ca.astype(int)-cb.astype(int)
    boot=[]
    for _ in range(2000):
        idx=np.concatenate([rng.choice(np.flatnonzero(rows.year.to_numpy()==year),size=int((rows.year==year).sum()),replace=True) for year in sorted(rows.year.unique())])
        boot.append(float(difference[idx].mean()*100))
    return dict(candidate_only_correct=wins,baseline_only_correct=losses,
        accuracy_difference_pp=float(difference.mean()*100),
        difference_bootstrap_95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
        exact_mcnemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.)


def return_metrics(rows,estimates):
    actual=rows.target_return.to_numpy()
    rho=spearmanr(estimates,actual).statistic if np.unique(estimates).size>1 else np.nan
    groups=[]
    for week,g in rows.assign(week=rows.decision_market_date.dt.to_period('W-SUN').astype(str)).groupby('week'):
        if len(g)<2:continue
        chosen=g.loc[pd.Series(estimates,index=rows.index).loc[g.index].idxmax()]
        groups.append(dict(week=week,n=len(g),selected_id=str(chosen.ipo_id),selected_return=float(chosen.target_return),
                           equal_weight_return=float(g.target_return.mean()),best_return=float(g.target_return.max())))
    return dict(mae_pp=float(mean_absolute_error(actual,estimates)),rmse_pp=float(np.sqrt(mean_squared_error(actual,estimates))),
        spearman=float(rho) if np.isfinite(rho) else None,
        weekly_ranking=dict(weeks=len(groups),mean_selected_listing_return_pct=float(np.mean([g['selected_return'] for g in groups])) if groups else None,
            mean_equal_weight_return_pct=float(np.mean([g['equal_weight_return'] for g in groups])) if groups else None,
            mean_regret_to_best_pct=float(np.mean([g['best_return']-g['selected_return'] for g in groups])) if groups else None,
            groups=groups),
        limitation='Retrospective listing-return ranking among same decision-week IPOs; not an executable portfolio or allotment-adjusted P&L.')


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    from ml.audit_subscription_numeric_formats_20261028 import run as audit_numbers
    audit_numbers()
    protocol=dict(training_years=[2021,2022,2023,2024],test_years=[2025,2026],
        experts=EXPERTS,meta_C=.5,direction_threshold=.5,return_ridge_alpha=20,
        meta_training='Expanding-year out-of-fold expert predictions from 2022, 2023 and 2024 only',
        no_test_tuning=True,calibration='none',live_promotion=False,
        primary_test='Matched issuers with verified Day-2 evidence; all richer-input rows and prior 155 also reported separately',
        warning='Architecture and these years have already been inspected during development; this is a leakage-checked retrospective comparison, not a fresh independent holdout.',
        source_sha256=digest(SOURCE))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    raw=pd.read_csv(SOURCE);frame=raw[raw.cohort.eq('historical')].copy()
    frame['ipo_id']=pd.to_numeric(frame.ipo_id).astype(int)
    for field in ['listing_date','decision_market_date']:frame[field]=pd.to_datetime(frame[field])
    if not np.allclose(frame.target_binary,(frame.target_return>0).astype(int)):raise ValueError('Outcome sign conflict')
    if not np.isfinite(frame.target_return).all():raise ValueError('Unverified return label')
    train=frame[frame.year.between(2021,2024)].copy().reset_index(drop=True)
    test=frame[frame.year.between(2025,2026)].copy().reset_index(drop=True)
    guard(train,test)
    original=legacy.prepare_dataset();original_train=original[original.year.between(2021,2024)&~original.id.isin(legacy.EXCLUDED_NON_CORPORATE_IDS)].copy()
    original_test=original[original.year.between(2025,2026)].copy()
    if set(original_train.id)&set(original_test.id):raise ValueError('Original-dataset overlap')
    if set(original_train.company_name.map(name_key))&set(original_test.company_name.map(name_key)):raise ValueError('Original-dataset company overlap')
    scaler,model=legacy.fit_model(original_train)
    original_scores=legacy.score_model(scaler,model,original_test)
    original_test=original_test.assign(probability=original_scores,correct=(original_scores>=.5)==original_test.target_binary)
    original_test.to_csv(OUT/'original_dataset_189_predictions.csv',index=False)
    # The original model is also scored on the EXACT early books supplied to
    # the newer architecture. Closing-book diagnostics remain a separate table.
    test_legacy=legacy_features(test);train_legacy=legacy_features(train)
    test['p_legacy_original252']=legacy.score_model(scaler,model,test_legacy)
    matched_scaler,matched_model=legacy.fit_model(train_legacy)
    test['p_legacy_matched186']=legacy.score_model(matched_scaler,matched_model,test_legacy)
    return_legacy=ridge().fit(train_legacy[legacy.FEATURE_COLS],train.target_return)
    test['r_legacy_three_input_ridge']=return_legacy.predict(test_legacy[legacy.FEATURE_COLS])
    core.SETS['gmp_only']=['gmp_pct']
    configs=core.candidates()+[dict(name='gmp_only_control',family='logistic',C=.5,class_weight=None,transform='raw',feature_set='gmp_only')]
    configs={name:next(c for c in configs if c['name']==name) for name in EXPERTS}
    meta_x=[];meta_y=[];meta_r=[];meta_returns=[];trace=[]
    for year in (2022,2023,2024):
        held=train[train.year.eq(year)];earlier=train[train.year.lt(year)&train.listing_date.lt(held.decision_market_date.min())]
        if held.empty or earlier.empty:continue
        assert set(earlier.ipo_id).isdisjoint(held.ipo_id) and earlier.listing_date.max()<held.decision_market_date.min()
        p=[];r=[]
        for name,c in configs.items():
            m=core.estimator(c).fit(core.features(earlier,c),earlier.target_binary)
            p.append(m.predict_proba(core.features(held,c))[:,1])
            reg=ridge().fit(core.features(earlier,c),earlier.target_return)
            r.append(reg.predict(core.features(held,c)))
        meta_x.append(np.column_stack(p));meta_r.append(np.column_stack(r));meta_y.extend(held.target_binary);meta_returns.extend(held.target_return)
        trace.append(dict(year=year,fit_ids=earlier.ipo_id.tolist(),held_ids=held.ipo_id.tolist(),latest_fit_outcome=str(earlier.listing_date.max().date()),first_held_decision=str(held.decision_market_date.min().date())))
    meta=stacker().fit(logits(np.vstack(meta_x)),np.array(meta_y,dtype=int))
    return_meta=make_pipeline(StandardScaler(),Ridge(alpha=20)).fit(np.vstack(meta_r),np.array(meta_returns))
    models={};return_models={};p=[];r=[]
    for name,c in configs.items():
        models[name]=core.estimator(c).fit(core.features(train,c),train.target_binary)
        return_models[name]=ridge().fit(core.features(train,c),train.target_return)
        p.append(models[name].predict_proba(core.features(test,c))[:,1]);r.append(return_models[name].predict(core.features(test,c)))
    test['p_five_expert_strict2024']=meta.predict_proba(logits(np.column_stack(p)))[:,1]
    test['p_gmp_only']=p[-1];test['p_gmp_sign']=(test.gmp_pct>0).astype(float);test['p_always_positive']=1.
    test['r_five_view_ridge']=return_meta.predict(np.column_stack(r))
    test['r_gmp_only']=test.gmp_pct
    test['r_constant_training_mean']=float(train.target_return.mean())
    for col in [c for c in test if c.startswith('p_') and c in ['p_legacy_original252','p_legacy_matched186','p_five_expert_strict2024','p_gmp_only','p_gmp_sign','p_always_positive']]:
        test[col+'_correct']=(test[col]>=.5)==test.target_binary
    old_ids=set(pd.read_csv(ROOT/'output/module1_stacking_20261025_through2024_predictions.csv').ipo_id.astype(int))
    cohorts={'clean_day2_163':test[test.verified_stage.eq('day2')], 'all_richer_inputs_173':test,
        'previous_fixed155':test[test.ipo_id.isin(old_ids)]}
    directions=['p_legacy_original252','p_legacy_matched186','p_five_expert_strict2024','p_gmp_only','p_always_positive']
    results={}
    for label,rows in cohorts.items():
        scopes={'combined':rows,**{str(int(year)):g for year,g in rows.groupby('year')}}
        results[label]={}
        for scope,g in scopes.items():
            results[label][scope]=dict(n=len(g),direction={col:metrics(g,g[col].to_numpy()) for col in directions},
                new_vs_original=paired(g,g.p_five_expert_strict2024.to_numpy(),g.p_legacy_original252.to_numpy()),
                new_vs_matched=paired(g,g.p_five_expert_strict2024.to_numpy(),g.p_legacy_matched186.to_numpy()),
                new_vs_gmp=paired(g,g.p_five_expert_strict2024.to_numpy(),g.p_gmp_only.to_numpy()),
                returns={col:return_metrics(g,g[col].to_numpy()) for col in ['r_legacy_three_input_ridge','r_five_view_ridge','r_constant_training_mean']})
            paired_gmp=g[g.gmp_pct.notna()]
            results[label][scope]['dated_gmp_direction_comparison']=dict(n=len(paired_gmp),
                models={col:metrics(paired_gmp,paired_gmp[col].to_numpy()) for col in directions},
                new_vs_gmp=paired(paired_gmp,paired_gmp.p_five_expert_strict2024.to_numpy(),paired_gmp.p_gmp_only.to_numpy()))
            results[label][scope]['dated_gmp_return_comparison']=dict(n=len(paired_gmp),
                models={col:return_metrics(paired_gmp,paired_gmp[col].to_numpy()) for col in ['r_legacy_three_input_ridge','r_five_view_ridge','r_gmp_only']})
    # Actual deployed artifact is valid for 2026 only, since it includes 2025
    # training labels. Do not misrepresent its score on any 2025 issuer.
    deployed_path=ROOT/'ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib'
    deployed_hash=digest(deployed_path);deployed=joblib.load(deployed_path)
    forward_train=frame[frame.year.between(2021,2025)].copy()
    forward_test=test[test.year.eq(2026)].copy()
    assert set(forward_train.ipo_id).isdisjoint(forward_test.ipo_id)
    assert set(forward_train.company_name.map(name_key)).isdisjoint(forward_test.company_name.map(name_key))
    assert forward_train.listing_date.max()<forward_test.decision_market_date.min()
    if set(map(int,deployed['training_ids']))!=set(forward_train.ipo_id):raise ValueError('Deployed training membership differs')
    dp=np.column_stack([deployed['models'][name].predict_proba(core.features(forward_test,deployed['expert_configs'][name]))[:,1] for name in deployed['experts']])
    forward_test['p_deployed_five_expert']=deployed['meta_model'].predict_proba(logits(dp))[:,1]
    ss,mm=legacy.fit_model(legacy_features(forward_train))
    forward_test['p_stage_logistic_through2025']=legacy.score_model(ss,mm,legacy_features(forward_test))
    deployment_comparison={}
    for label,rows in [('all69',forward_test),('verified_day2_64',forward_test[forward_test.verified_stage.eq('day2')])]:
        deployment_comparison[label]=dict(n=len(rows),training_n=len(forward_train),training_end_year=2025,
            direction={k:metrics(rows,rows[k].to_numpy()) for k in ['p_deployed_five_expert','p_stage_logistic_through2025','p_legacy_original252']},
            new_vs_same_stage_logistic=paired(rows,rows.p_deployed_five_expert.to_numpy(),rows.p_stage_logistic_through2025.to_numpy()))
    forward_test.to_csv(OUT/'actual_deployed_2026_only_predictions.csv',index=False)
    if digest(deployed_path)!=deployed_hash:raise ValueError('Deployed artifact changed')
    coverage=original_test[['id','company_name','year']].copy();coverage['eligible_richer_inputs']=coverage.id.isin(test.ipo_id)
    stage=dict(zip(test.ipo_id,test.verified_stage));coverage['stage']=coverage.id.map(stage)
    coverage['exclusion_reason']=np.where(~coverage.eligible_richer_inputs,'No matched recoverable dated early-book feature row',np.where(coverage.stage.eq('day2'),'','Interim reading does not have verified Day-2 stage'))
    coverage.to_csv(OUT/'coverage_all189.csv',index=False);test.to_csv(OUT/'paired_predictions.csv',index=False)
    train[['ipo_id','company_name','year','listing_date','decision_market_date']].to_csv(OUT/'new_training_membership.csv',index=False)
    original_train[['id','company_name','year','listing_date']].to_csv(OUT/'original_training_membership.csv',index=False)
    previous=original.set_index('id').listing_open_price
    outcome_differences=test.loc[(test.exchange_open-test.ipo_id.map(previous)).abs()>.01,['ipo_id','company_name','exchange_open']].copy()
    outcome_differences['original_dataset_open']=outcome_differences.ipo_id.map(previous)
    artifact=OUT/'strict2024_research_models.joblib'
    joblib.dump(dict(models=models,meta=meta,expert_configs=configs,return_models=return_models,return_meta=return_meta,
        legacy_original_scaler=scaler,legacy_original_model=model,legacy_matched_scaler=matched_scaler,legacy_matched_model=matched_model,
        train_ids=train.ipo_id.tolist(),training_end_year=2024,research_only=True),artifact)
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
        training=dict(original_n=len(original_train),new_and_matched_n=len(train),meta_oof_n=len(meta_y),max_year=int(train.year.max()),train_test_id_overlap=0,train_test_company_overlap=0,folds=trace),
        training_feature_coverage={k:dict(observed=int(train[k].notna().sum()),missing=int(train[k].isna().sum()),total=len(train)) for k in ['gmp_pct','pub_ipo_earnings_yield_pct','pub_log_ipo_price_book','log_qib_growth','log_hni_growth','log_retail_growth']},
        training_positive_rate_pct=float(train.target_binary.mean()*100),
        original_final_book_diagnostic={str(int(y)):metrics(g,g.probability.to_numpy()) for y,g in original_test.groupby('year')},
        original_final_book_combined=metrics(original_test,original_scores),
        test_coverage=dict(original_n=len(original_test),matched_n=len(test),verified_day2_n=int(test.verified_stage.eq('day2').sum()),no_richer_input_row=int((~coverage.eligible_richer_inputs).sum()),unresolved_stage=int(test.verified_stage.ne('day2').sum())),
        matched_results=results,outcome_differences=outcome_differences.to_dict('records'),
        numeric_format_audit=json.loads((OUT/'numeric_format_audit.json').read_text()),
        actual_deployed_2026_only=deployment_comparison,deployed_artifact_sha256_unchanged=deployed_hash,
        outcome_policy='Matched comparisons use the same recovered exchange opening for every model; original-dataset diagnostic retains original labels. No source ledger was edited.',
        return_policy='Fixed Ridge return companions are separate research models; directional probabilities are never treated as return percentages.',
        input_hashes={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE,ROOT/'data/audit/module1_stage_aligned_inputs_20261026.csv',ROOT/'data/canonical_listed_ipos.csv',ROOT/'data/subscription_observations.csv',ROOT/'services/subscription_numbers.py',ROOT/'ml/audit_subscription_numeric_formats_20261028.py',Path(__file__)]},
        artifact_sha256=digest(artifact),live_models_changed=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    lines=['# OpenBell — matched model comparison','', 'Training: 2021–2024 only. Test labels: 2025–2026. Research replay; live artifacts and saved calls unchanged.','',
        f'Original training: {len(original_train)} IPOs. Matched early-book training: {len(train)}. Original holdout: {len(original_test)}; richer-input overlap: {len(test)}; verified Day-2 subset: {int(test.verified_stage.eq("day2").sum())}.','',
        '| Cohort | Period | Original model /252 | Same-book logistic /186 | Five-expert /186 | GMP-only | Always positive |',
        '|---|---|---:|---:|---:|---:|---:|']
    for label,scopes in results.items():
        for scope,v in scopes.items():
            cells=[f"{v['direction'][k]['correct']}/{v['n']} ({v['direction'][k]['classifier_accuracy_pct']:.1f}%)" for k in directions]
            lines.append('| '+' | '.join([label,scope,*cells])+' |')
    primary=results['clean_day2_163']['combined']
    lines+=['','Paired five-expert minus original-model accuracy difference: '+json.dumps(primary['new_vs_original']),
        '', '## Return estimation and weekly ranking','', '| Return companion | MAE (percentage points) | Spearman | Mean weekly selected listing return |', '|---|---:|---:|---:|']
    for k,v in primary['returns'].items():
        correlation=f"{v['spearman']:.3f}" if v['spearman'] is not None else 'Undefined (constant)'
        lines.append(f"| {k} | {v['mae_pp']:.2f} | {correlation} | {v['weekly_ranking']['mean_selected_listing_return_pct']:.2f}% |")
    lines+=['','## Actual deployed model: 2026-only comparison','',
        'This separate comparison permits training through 2025. It never scores 2025 as unseen.',
        '', '| Cohort | Deployed five-expert /290 | Same-stage logistic /290 | Original logistic /252 on the same early books |','|---|---:|---:|---:|']
    for label,v in deployment_comparison.items():
        cells=[f"{v['direction'][k]['correct']}/{v['n']} ({v['direction'][k]['classifier_accuracy_pct']:.1f}%)" for k in ['p_deployed_five_expert','p_stage_logistic_through2025','p_legacy_original252']]
        lines.append('| '+' | '.join([label,*cells])+' |')
    lines+=['','GMP return comparisons use only the identical rows with a real dated quote; see report.json for denominators, annual results and weekly selections.',
        'The full-cohort GMP-only classifier uses its training median for missing quotes. The report separately provides dated-GMP direction comparisons excluding missing quotes for every model.',
        '', '## Limits','',protocol['warning'],
        'The original model was trained on final subscriptions. Its early-book replay is a stage-shift diagnostic. The matched logistic control isolates model architecture on the same training issuers/books.',
        'Ten interim-stage rows are outside the primary verified-Day-2 comparison; sixteen original holdout IPOs lack matched early-book features. The exclusions are enumerated in coverage_all189.csv.',
        'Three exchange opening values differ from the original dataset; matched comparisons consistently use the recovered exchange openings, with differences recorded in report.json.',
        'Weekly rankings do not account for allotment, available application capital or execution costs. No profit or independent generalisation claim follows from this rerun.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(training=report['training']|{'folds':'recorded in report'},coverage=report['test_coverage'],primary=primary['direction'],paired=primary['new_vs_original']),indent=2))


if __name__=='__main__':main()
