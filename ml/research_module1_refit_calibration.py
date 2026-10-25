"""Training-only cross-fit calibration of fixed previously selected families."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import ml.research_module1_forward_update as core
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain

ROOT=core.ROOT
PREFIX=ROOT/'output/module1_refit_calibration_20261025'


def choose_cutoff(rows,end_year,probability):
    if not rows.year.between(2021,end_year).all():
        raise ValueError('Test year cannot calibrate threshold')
    recent=rows[rows.year.eq(end_year)]
    if recent.empty:
        raise ValueError('No latest-year calibration rows')
    y=recent.target_binary.to_numpy();p=recent[probability].to_numpy();settings=[]
    for threshold in core.prior.THRESHOLDS:
        call=p>=threshold
        nr=float((~call[y==0]).mean());pr=float(call[y==1].mean())
        settings.append(dict(threshold=float(threshold),n=len(y),accuracy=float((call==y).mean()),balanced_accuracy=(nr+pr)/2))
    settings.sort(key=lambda r:(-r['accuracy'],-r['balanced_accuracy'],abs(r['threshold']-.5),r['threshold']))
    return settings[0],settings


def run():
    core.SETS['gmp_only']=['gmp_pct']
    frame_path=ROOT/'output/module1_forward_coverage_20261024_features.csv'
    data=pd.read_csv(frame_path)
    for field in ('listing_date','decision_market_date'):
        data[field]=pd.to_datetime(data[field])
    original=pd.read_csv(ROOT/'output/module1_blended_fix_20261024_predictions.csv').set_index('ipo_id')
    selections={};artifacts={};heldouts={}
    for end in (2024,2025):
        path=ROOT/f'ml/artifacts/module1_forward_coverage_through{end}_research_20261024.joblib'
        artifact=joblib.load(path);config=artifact['config']
        train=data[data.ipo_id.isin(artifact['training_ids'])].copy().reset_index(drop=True)
        test=data[data.ipo_id.isin(original.index)&data.year.gt(end)].copy()
        core.membership_guard(train,test,end)
        train['p_crossfit']=np.nan;train['p_gmp_crossfit']=np.nan
        gmp_config=dict(family='logistic',transform='raw',C=.5,class_weight=None,feature_set='gmp_only')
        trace=[]
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        for fold,(fit_index,held_index) in enumerate(cv.split(train,train.target_binary)):
            fit=train.iloc[fit_index];held=train.iloc[held_index]
            if set(fit.ipo_id)&set(held.ipo_id):
                raise ValueError('Crossfit issuer leakage')
            model=core.estimator(config).fit(core.features(fit,config),fit.target_binary)
            baseline=core.estimator(gmp_config).fit(core.features(fit,gmp_config),fit.target_binary)
            train.loc[held_index,'p_crossfit']=model.predict_proba(core.features(held,config))[:,1]
            train.loc[held_index,'p_gmp_crossfit']=baseline.predict_proba(core.features(held,gmp_config))[:,1]
            trace.append(dict(fold=fold,fit_ids=sorted(map(int,fit.ipo_id)),held_ids=sorted(map(int,held.ipo_id))))
        if train[['p_crossfit','p_gmp_crossfit']].isna().any().any():
            raise ValueError('Incomplete cross-fitting')
        choice,settings=choose_cutoff(train,end,'p_crossfit')
        gmp_choice,gmp_settings=choose_cutoff(train,end,'p_gmp_crossfit')
        selection=dict(chosen=choice,gmp_chosen=gmp_choice,settings=settings,gmp_settings=gmp_settings,
            training_end_year=end,training_ids=artifact['training_ids'],test_ids=sorted(map(int,test.ipo_id)),
            trace=trace,test_used_for_selection=False,config=config,previous_threshold=artifact['threshold'])
        train.to_csv(PREFIX.with_name(PREFIX.name+f'_through{end}_crossfit.csv'),index=False)
        selection_path=PREFIX.with_name(PREFIX.name+f'_through{end}_selection.json')
        selection_path.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf8')
        final_path=ROOT/f'ml/artifacts/module1_refit_calibrated_through{end}_research_20261025.joblib'
        joblib.dump({**artifact,'threshold':choice['threshold'],'previous_threshold':artifact['threshold'],
            'calibration_selection_sha256':core.sha(selection_path)},final_path)
        selections[str(end)]=selection;artifacts[end]=(artifact,final_path);heldouts[end]=test
    # Both earlier-only cutoff selections are saved before scoring test rows.
    results={};ledgers={}
    for end in (2024,2025):
        artifact,final_path=artifacts[end];selection=selections[str(end)];test=heldouts[end]
        test['p_calibrated_policy']=artifact['model'].predict_proba(core.features(test,artifact['config']))[:,1]
        gmp_config=dict(family='logistic',transform='raw',C=.5,class_weight=None,feature_set='gmp_only')
        test['p_gmp_control']=artifact['gmp_control'].predict_proba(core.features(test,gmp_config))[:,1]
        threshold=selection['chosen']['threshold']
        test['direction_cutoff']=threshold;test['trained_through']=end
        test['policy_call']=test.p_calibrated_policy.ge(threshold).astype(int)
        test['policy_correct']=test.policy_call.eq(test.target_binary)
        test.to_csv(PREFIX.with_name(PREFIX.name+f'_through{end}_predictions.csv'),index=False)
        results[str(end)]=dict(training_n=len(artifact['training_ids']),test_n=len(test),
            metrics=summarize(test.target_binary,test.p_calibrated_policy,threshold),
            previous_cutoff_metrics=summarize(test.target_binary,test.p_calibrated_policy,selection['previous_threshold']),
            gmp_control=summarize(test.target_binary,test.p_gmp_control,selection['gmp_chosen']['threshold']),
            by_year={str(int(y)):summarize(g.target_binary,g.p_calibrated_policy,threshold) for y,g in test.groupby('year')},
            paired_change=paired_gain(test.policy_correct,test.ipo_id.map(original.blended_correct)))
        ledgers[end]=test
    annual=pd.concat([ledgers[2024][ledgers[2024].year.eq(2025)],ledgers[2025]],ignore_index=True)
    if set(annual.ipo_id)!=set(original.index) or annual.ipo_id.duplicated().any():
        raise ValueError('Annual replay cohort changed')
    annual.to_csv(PREFIX.with_name(PREFIX.name+'_annual_replay.csv'),index=False)
    old_fn=original[original.target_binary.eq(1)&original.blended_call.eq(0)].index
    review=annual[annual.ipo_id.isin(old_fn)]
    review.to_csv(PREFIX.with_name(PREFIX.name+'_20_missed_positive_review.csv'),index=False)
    paths=[frame_path,ROOT/'ml/research_module1_refit_calibration.py',ROOT/'output/module1_refit_calibration_protocol_20261025.md']
    result=dict(selections=selections,results=results,
        annual=dict(n=len(annual),correct=int(annual.policy_correct.sum()),accuracy_pct=float(annual.policy_correct.mean()*100)),
        previous20corrected=int(review.policy_correct.sum()),live_promoted=False,
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):core.sha(p) for p in paths},
        limitations=['Internal cross-fitting uses only pre-test labels and is not itself a chronological forecasting backtest.',
            '2025 is training/calibration only for the updated 2026 test; it is not counted as that model’s holdout.',
            'All original 155 and all 69 2026 test IPOs remain; reused outcomes make this exploratory evidence.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(cutoffs={y:s['chosen'] for y,s in selections.items()},results=results,
        annual=result['annual'],previous20corrected=result['previous20corrected']),indent=2))


if __name__=='__main__':
    run()
