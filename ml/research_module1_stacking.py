"""Fixed expert views stacked with pre-test cross-fit calibration."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ml.research_module1_forward_update as core
from ml.model_pipeline import summarize
from ml.research_module1_refit_calibration import choose_cutoff

ROOT=core.ROOT
PREFIX=ROOT/'output/module1_stacking_20261025'
EXPERTS=['base__candidate_02','base_macro__candidate_07','financial_macro__candidate_08',
         'trajectory__candidate_19','gmp_only_control']


def logits(probabilities):
    p=np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))


def stacker():
    return Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,max_iter=3000,random_state=42))])


def run_enriched():
    """Explicit research entry point; never replaces the frozen stack artifact."""
    from ml.research_module1_enriched import run
    return run()


def run():
    core.SETS['gmp_only']=['gmp_pct']
    source=ROOT/'output/module1_forward_coverage_20261024_features.csv'
    validation_path=ROOT/'output/module1_forward_coverage_20261024_validation.csv'
    frame=pd.read_csv(source);validation=pd.read_csv(validation_path)
    for field in ('listing_date','decision_market_date'):
        frame[field]=pd.to_datetime(frame[field])
    configs=core.candidates()
    configs.append(dict(name='gmp_only_control',family='logistic',C=.5,class_weight=None,transform='raw',feature_set='gmp_only'))
    selected_configs={name:next(c for c in configs if c['name']==name) for name in EXPERTS}
    original=pd.read_csv(ROOT/'output/module1_blended_fix_20261024_predictions.csv').set_index('ipo_id')
    records={};artifacts={};tests={}
    for end in (2024,2025):
        meta_rows=validation[validation.year.isin([end-1,end])].copy().reset_index(drop=True)
        if not meta_rows.year.le(end).all():
            raise ValueError('Later labels reached meta fitting')
        x=logits(meta_rows[EXPERTS]);y=meta_rows.target_binary
        crossfit=np.full(len(meta_rows),np.nan)
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        trace=[]
        for fold,(fit_index,held_index) in enumerate(cv.split(x,y)):
            model=stacker().fit(x[fit_index],y.iloc[fit_index])
            crossfit[held_index]=model.predict_proba(x[held_index])[:,1]
            trace.append(dict(fold=fold,fit_ids=sorted(map(int,meta_rows.iloc[fit_index].ipo_id)),held_ids=sorted(map(int,meta_rows.iloc[held_index].ipo_id))))
        if not np.isfinite(crossfit).all():
            raise ValueError('Incomplete stack crossfit')
        meta_rows['p_meta_crossfit']=crossfit
        cutoff,ranked=choose_cutoff(meta_rows,end,'p_meta_crossfit')
        held=frame[frame.ipo_id.isin(original.index)&frame.year.gt(end)].copy()
        earlier=frame[frame.year.le(end)&frame.listing_date.lt(held.decision_market_date.min())]
        core.membership_guard(earlier,held,end)
        selection=dict(experts=EXPERTS,cutoff=cutoff,ranked=ranked,training_ids=sorted(map(int,earlier.ipo_id)),
            meta_training_ids=sorted(map(int,meta_rows.ipo_id)),test_ids=sorted(map(int,held.ipo_id)),
            trace=trace,test_used_for_selection=False,meta_validation_interpretation='Training-only crossfit on already observed pre-test labels, not a contemporaneous forecast ledger.')
        selection_path=PREFIX.with_name(PREFIX.name+f'_through{end}_selection.json')
        selection_path.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf8')
        meta_rows.to_csv(PREFIX.with_name(PREFIX.name+f'_through{end}_meta_crossfit.csv'),index=False)
        models={name:core.estimator(config).fit(core.features(earlier,config),earlier.target_binary) for name,config in selected_configs.items()}
        meta=stacker().fit(x,y)
        artifact=ROOT/f'ml/artifacts/module1_stacking_through{end}_research_20261025.joblib'
        value=dict(models=models,meta_model=meta,expert_configs=selected_configs,experts=EXPERTS,
            threshold=cutoff['threshold'],training_ids=selection['training_ids'],training_end_year=end,research_only=True,
            selection_sha256=core.sha(selection_path))
        joblib.dump(value,artifact)
        records[str(end)]=selection;artifacts[end]=value;tests[end]=held
    results={};ledgers={}
    for end in (2024,2025):
        held=tests[end];value=artifacts[end]
        p=np.column_stack([value['models'][name].predict_proba(core.features(held,selected_configs[name]))[:,1] for name in EXPERTS])
        for column,name in enumerate(EXPERTS):
            held['p_expert_'+name]=p[:,column]
        held['p_stack']=value['meta_model'].predict_proba(logits(p))[:,1]
        cutoff=value['threshold'];held['direction_cutoff']=cutoff;held['trained_through']=end
        held['stack_call']=held.p_stack.ge(cutoff).astype(int)
        held['stack_correct']=held.stack_call.eq(held.target_binary)
        held.to_csv(PREFIX.with_name(PREFIX.name+f'_through{end}_predictions.csv'),index=False)
        results[str(end)]=dict(training_n=len(value['training_ids']),test_n=len(held),
            metrics=summarize(held.target_binary,held.p_stack,cutoff),
            by_year={str(int(y)):summarize(g.target_binary,g.p_stack,cutoff) for y,g in held.groupby('year')})
        ledgers[end]=held
    annual=pd.concat([ledgers[2024][ledgers[2024].year.eq(2025)],ledgers[2025]],ignore_index=True)
    if set(annual.ipo_id)!=set(original.index) or annual.ipo_id.duplicated().any():
        raise ValueError('Changed annual cohort')
    annual.to_csv(PREFIX.with_name(PREFIX.name+'_annual_replay.csv'),index=False)
    previous_fn=original[original.target_binary.eq(1)&original.blended_call.eq(0)].index
    review=annual[annual.ipo_id.isin(previous_fn)]
    review.to_csv(PREFIX.with_name(PREFIX.name+'_20_missed_positive_review.csv'),index=False)
    paths=[source,validation_path,ROOT/'ml/research_module1_stacking.py',ROOT/'output/module1_stacking_protocol_20261025.md']
    result=dict(selections=records,results=results,
        annual=dict(n=len(annual),correct=int(annual.stack_correct.sum()),accuracy_pct=float(annual.stack_correct.mean()*100)),
        previous20corrected=int(review.stack_correct.sum()),live_promoted=False,
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):core.sha(p) for p in paths})
    PREFIX.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(cutoffs={y:s['cutoff'] for y,s in records.items()},results=results,annual=result['annual'],previous20corrected=result['previous20corrected']),indent=2))


if __name__=='__main__':
    run()
