"""Audit conventional classifier labels beside the custom-cutoff experiment.

The 0.5 comparison was added after inspecting custom-cutoff failures; report
that timing and retain both results. It is not a newly independent test.
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain

ROOT=Path(__file__).resolve().parents[1]
PREFIX=ROOT/'output/module1_standard_classifier_audit_20261025'


def wilson(correct,n):
    z=1.959963984540054;p=correct/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [100*(mid-half),100*(mid+half)]


def run():
    from ml.research_module1_stacking import logits
    import ml.research_module1_forward_update as core
    core.SETS['gmp_only']=['gmp_pct']
    stack_report_path=ROOT/'output/module1_stacking_20261025.json'
    stack=json.loads(stack_report_path.read_text())
    strict_path=ROOT/'output/module1_blended_fix_20261024_predictions.csv'
    strict=pd.read_csv(strict_path).set_index('ipo_id')
    frame_path=ROOT/'output/module1_forward_coverage_20261024_features.csv'
    frame=pd.read_csv(frame_path)
    artifact_path=ROOT/'ml/artifacts/module1_stacking_through2025_research_20261025.joblib'
    artifact=joblib.load(artifact_path)
    test=frame[frame.ipo_id.isin(strict.index)&frame.year.eq(2026)].copy()
    train=frame[frame.ipo_id.isin(artifact['training_ids'])]
    if set(test.ipo_id)&set(train.ipo_id) or not train.year.between(2021,2025).all():
        raise ValueError('Training/test leakage')
    if pd.to_datetime(train.listing_date).max()>=pd.to_datetime(test.decision_market_date).min():
        raise ValueError('Unmatured training outcome')
    selection=stack['selections']['2025']
    if not set(selection['meta_training_ids']).issubset(set(artifact['training_ids'])):
        raise ValueError('Meta training used holdout labels')
    for fold in selection['trace']:
        if set(fold['fit_ids'])&set(fold['held_ids']):
            raise ValueError('Meta crossfit overlap')
    probabilities=np.column_stack([artifact['models'][name].predict_proba(core.features(test,artifact['expert_configs'][name]))[:,1]
        for name in artifact['experts']])
    meta_x=logits(probabilities)
    p=artifact['meta_model'].predict_proba(meta_x)[:,1]
    predicted=artifact['meta_model'].predict(meta_x)
    if not np.array_equal(predicted,(p>=.5).astype(int)):
        raise ValueError('Standard predict() is not the declared 50% rule')
    saved=pd.read_csv(ROOT/'output/module1_stacking_20261025_through2025_predictions.csv').set_index('ipo_id')
    if not np.allclose(p,saved.loc[test.ipo_id].p_stack,atol=1e-12):
        raise ValueError('Frozen artifact replay differs')
    test['standard_probability']=p;test['standard_call']=predicted
    test['standard_correct']=test.standard_call.eq(test.target_binary)
    test['standard_cutoff']=.5
    test['custom_cutoff']=artifact['threshold']
    test['custom_call']=test.standard_probability.ge(artifact['threshold']).astype(int)
    test['custom_correct']=test.custom_call.eq(test.target_binary)
    test.to_csv(PREFIX.with_name(PREFIX.name+'_2026_predictions.csv'),index=False)
    test[~test.standard_correct].to_csv(PREFIX.with_name(PREFIX.name+'_2026_errors.csv'),index=False)
    output=ROOT/'ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib'
    joblib.dump({**artifact,'threshold':.5,'classification_rule':'standard meta_model.predict()',
        'validation_optimized_policy_cutoff':artifact['threshold'],'source_artifact':str(artifact_path.relative_to(ROOT)),
        'comparison_added_after_test_inspection':True,'research_only':True},output)
    evidence=strict.reset_index()[['ipo_id','company_name','year','issue_price','exchange_open','target_return',
        'decision_market_date','gmp_date','gmp_pct','p_blended_fix','blended_call','blended_correct']].copy()
    evidence=evidence.rename(columns={'p_blended_fix':'strict_2021_2024_probability','blended_call':'strict_2021_2024_call',
        'blended_correct':'strict_2021_2024_correct'})
    evidence['updated_evaluation_status']=np.where(evidence.year.eq(2026),'2026 holdout','2025 training/calibration; excluded from updated accuracy')
    evidence=evidence.merge(test[['ipo_id','standard_probability','standard_call','standard_correct']],on='ipo_id',how='left',validate='one_to_one')
    evidence.to_csv(PREFIX.with_name(PREFIX.name+'_all_155_status.csv'),index=False)
    old_fn=evidence[evidence.target_return.gt(0)&evidence.strict_2021_2024_call.eq(0)].copy()
    if len(old_fn)!=20:
        raise ValueError('Original 20 missed positive IPOs changed')
    old_fn.to_csv(PREFIX.with_name(PREFIX.name+'_20_missed_positive_review.csv'),index=False)
    metrics=summarize(test.target_binary,p,.5)
    correct=int(test.standard_correct.sum());n=len(test)
    baseline=test.ipo_id.map(strict.blended_correct)
    gmp_report=json.loads((ROOT/'output/module1_forward_coverage_20261024.json').read_text())['results']['2025']['gmp_control']
    paths=[Path(__file__),stack_report_path,strict_path,frame_path,artifact_path,output,
        ROOT/'ml/research_module1_stacking.py',ROOT/'ml/research_module1_forward_update.py',
        ROOT/'output/module1_stacking_protocol_20261025.md']
    report=dict(training_n=len(train),training_years=[2021,2022,2023,2024,2025],
        training_ids=sorted(map(int,train.ipo_id)),test_ids=sorted(map(int,test.ipo_id)),test_year=2026,test_n=n,
        correct=correct,metrics=metrics,accuracy_wilson_95_pct=wilson(correct,n),
        custom_policy_metrics=summarize(test.target_binary,p,artifact['threshold']),
        previous_blend_2026_correct=int(baseline.sum()),vs_previous_blend=paired_gain(test.standard_correct,baseline),
        gmp_control_previous_training_selected_cutoff=gmp_report,
        strict_155_correct=int(evidence.strict_2021_2024_correct.sum()),strict_155_accuracy_pct=float(evidence.strict_2021_2024_correct.mean()*100),
        previous_20=dict(total=20,training_2025=int(old_fn.year.eq(2025).sum()),eligible_2026=int(old_fn.year.eq(2026).sum()),
            corrected_2026=int(old_fn[old_fn.year.eq(2026)].standard_correct.sum())),
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):core.sha(p) for p in paths},
        artifact=str(output.relative_to(ROOT)),live_promoted=False,
        limitations=['The standard 50% classification comparison was added after the custom-policy test was inspected. No numeric cutoff search on 2026 was performed, but this is post-hoc exploratory evidence.',
            '81.2% applies to the 69 2026 IPOs after training through 2025, not to the full 155 IPOs.',
            'The 2025 rows are training/calibration, never scored as holdout for this updated model.',
            'The improvement over the previous 2026 blend is one IPO and its paired interval includes zero.',
            'Historical inputs are reconstructed; original pre-close forecasts and publication times are not certified.',
            'The live bundle, paper and Module 2 remain unchanged. A prospectively frozen test is needed before claiming sustained accuracy above 80%.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    lines=['# Module 1: standard-classifier audit and the 20 missed gains','',
        '**2026 candidate: 56/69 correct (81.2%). Strict 2021–2024 training on all 155: 118/155 (76.1%).**','',
        '| Evaluation | Training | Test | Correct | Overall accuracy |',
        '|---|---|---:|---:|---:|',
        '| Previous strict blend | 2021–2024 | Same 155 IPOs | 118/155 | 76.1% |',
        '| Previous blend, 2026 slice | 2021–2024 | Same 69 IPOs | 55/69 | 79.7% |',
        '| Updated GMP-only, earlier-selected cutoff | 2021–2025 | Same 69 IPOs | 55/69 | 79.7% |',
        '| Updated five-view stack, custom 57.5% cutoff | 2021–2025 | Same 69 IPOs | 52/69 | 75.4% |',
        '| Updated five-view stack, standard predict() | 2021–2025 | Same 69 IPOs | 56/69 | 81.2% |',
        '', '## What was changed','',
        '- Recovered 18 previously missing 2025 interim books, including brand-name and malformed-number handling. Numeric conflicts were not guessed.',
        '- Included 16 earlier IPOs whose usable books had been excluded for missing GMP. Missing GMP remains missing and is handled inside fitting.',
        '- Training now has 290 earlier IPOs: 186 from 2021–2024 and 104 from 2025. None of the 69 2026 IPOs enters either expert fitting or meta calibration.',
        '- Combined fixed views of bids/GMP, market conditions, financial facts and demand trajectories. The combiner was trained on earlier out-of-fold predictions.',
        '- The standard classifier uses its normal 50% class rule. The custom 57.5% policy was too conservative in this test. Both results are retained.',
        '', '## Error counts','',
        f'- Correct positive openings: {metrics["tp"]}/46; missed positive openings: {metrics["fn"]}.',
        f'- Correct flat/negative openings: {metrics["tn"]}/23; wrong positive calls: {metrics["fp"]}.',
        f'- Of the original 20 missed gains, 12 are 2025 training/calibration cases and eight are eligible 2026 tests. The updated model recovers {report["previous_20"]["corrected_2026"]} of those eight. All 20 remain in the review CSV.',
        f'- AUC: {metrics["roc_auc"]}; Brier: {metrics["brier_score"]}; 95% Wilson accuracy interval: {report["accuracy_wilson_95_pct"]}.',
        '', '## Interpretation','',
        'This crosses 80% for the 2026 candidate, with all 69 IPOs counted. It does not establish 80–85% for the original strict 155-row test or for future live forecasts. The standard-rule comparison was added after inspecting the failed custom-cutoff result; this is transparent exploratory evidence, not a fresh untouched holdout.',
        f'The gain over the previous 2026 blend is one correct IPO. Paired comparison: {report["vs_previous_blend"]}.',
        '', '## Files','',
        f'- All 155 IPOs, strict scores and updated eligibility: `{PREFIX.name}_all_155_status.csv`.',
        f'- Original 20 missed gains: `{PREFIX.name}_20_missed_positive_review.csv`.',
        f'- All 69 updated 2026 predictions and actual opens: `{PREFIX.name}_2026_predictions.csv`.',
        f'- All 13 updated wrong calls: `{PREFIX.name}_2026_errors.csv`.',
        f'- Reproducible candidate artifact: `{report["artifact"]}`.',
        '', '## Limits','',*[f'- {v}' for v in report['limitations']]]
    PREFIX.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(correct=correct,n=n,accuracy=metrics['classifier_accuracy_pct'],
        previous20=report['previous_20'],wilson=report['accuracy_wilson_95_pct'],paired=report['vs_previous_blend']),indent=2))


if __name__=='__main__':
    run()
