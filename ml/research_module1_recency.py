"""Year-relative sample weights; only pre-test labels choose settings."""
import json

import joblib
import numpy as np
import pandas as pd

import ml.research_module1_forward_update as core
from ml.model_pipeline import summarize

ROOT=core.ROOT
PREFIX=ROOT/'output/module1_recency_20261025'


def configs():
    seeds=[dict(name='base',family='logistic',transform='raw',C=.5,class_weight=None,feature_set='base'),
        dict(name='macro_balanced',family='logistic',transform='signed_log',C=.05,class_weight='balanced',feature_set='base_macro'),
        dict(name='financial_macro',family='logistic',transform='signed_log',C=.5,class_weight=None,feature_set='financial_macro'),
        dict(name='trajectory_forest',family='random_forest',transform='raw',max_depth=5,class_weight='balanced',feature_set='trajectory'),
        dict(name='gmp_only',family='logistic',transform='raw',C=.5,class_weight=None,feature_set='gmp_only')]
    return [dict(c,half_life=half,name=c['name']+'__half'+str(half)) for c in seeds for half in (None,2.,1.,.5)]


def weights(rows,end_year,half_life):
    if not rows.year.le(end_year).all():
        raise ValueError('Future training row')
    value=np.ones(len(rows)) if half_life is None else np.power(2.,-(end_year-rows.year.to_numpy())/half_life)
    return value/value.mean()


def fit(rows,config,end_year):
    w=weights(rows,end_year,config['half_life'])
    model=core.estimator(config)
    return model.fit(core.features(rows,config),rows.target_binary,scale__sample_weight=w,model__sample_weight=w)


def select(candidates,validation,end_year):
    if not validation.year.between(2022,end_year).all():
        raise ValueError('Later-year labels cannot select recency')
    held=validation[validation.year.eq(end_year)];y=held.target_binary.to_numpy();ranked=[]
    for config in candidates:
        p=held[config['name']].to_numpy()
        for threshold in core.prior.THRESHOLDS:
            call=p>=threshold;nr=float((~call[y==0]).mean());pr=float(call[y==1].mean())
            ranked.append(dict(name=config['name'],threshold=float(threshold),n=len(y),accuracy=float((call==y).mean()),
                balanced_accuracy=(nr+pr)/2,brier=float(np.mean((p-y)**2)),feature_count=len(core.SETS[config['feature_set']]),
                half_life=config['half_life']))
    ranked.sort(key=lambda r:(-r['accuracy'],-r['balanced_accuracy'],r['brier'],r['feature_count'],abs(r['threshold']-.5),
        -(r['half_life'] if r['half_life'] is not None else 1000),r['name']))
    return ranked[0],ranked


def run():
    core.SETS['gmp_only']=['gmp_pct']
    source=ROOT/'output/module1_forward_coverage_20261024_features.csv'
    data=pd.read_csv(source)
    for field in ('listing_date','decision_market_date'):
        data[field]=pd.to_datetime(data[field])
    original=pd.read_csv(ROOT/'output/module1_blended_fix_20261024_predictions.csv').set_index('ipo_id')
    candidates=configs();folds=[];trace=[]
    for year in (2022,2023,2024,2025):
        held=data[data.year.eq(year)].copy();first=held.decision_market_date.min()
        earlier=data[data.year.lt(year)&data.listing_date.lt(first)]
        core.membership_guard(earlier,held,year-1)
        for config in candidates:
            model=fit(earlier,config,year-1)
            held[config['name']]=model.predict_proba(core.features(held,config))[:,1]
        folds.append(held)
        trace.append(dict(year=year,train_ids=sorted(map(int,earlier.ipo_id)),validation_ids=sorted(map(int,held.ipo_id)),
            latest_training_outcome=str(earlier.listing_date.max().date()),first_validation_decision=str(first.date())))
    validation=pd.concat(folds,ignore_index=True)
    validation.to_csv(PREFIX.with_name(PREFIX.name+'_validation.csv'),index=False)
    selections={};models={};heldouts={}
    for end in (2024,2025):
        chosen,ranked=select(candidates,validation[validation.year.le(end)],end)
        config=next(c for c in candidates if c['name']==chosen['name'])
        held=data[data.ipo_id.isin(original.index)&data.year.gt(end)].copy()
        earlier=data[data.year.le(end)&data.listing_date.lt(held.decision_market_date.min())]
        core.membership_guard(earlier,held,end)
        selected=dict(chosen=chosen,config=config,ranked=ranked,training_ids=sorted(map(int,earlier.ipo_id)),
            test_ids=sorted(map(int,held.ipo_id)),test_used_for_selection=False)
        selection_path=PREFIX.with_name(PREFIX.name+f'_through{end}_selection.json')
        selection_path.write_text(json.dumps(selected,indent=2)+'\n',encoding='utf8')
        model=fit(earlier,config,end)
        artifact=ROOT/f'ml/artifacts/module1_recency_through{end}_research_20261025.joblib'
        joblib.dump(dict(model=model,config=config,features=core.SETS[config['feature_set']],threshold=chosen['threshold'],
            training_ids=selected['training_ids'],training_end_year=end,research_only=True,
            selection_sha256=core.sha(selection_path)),artifact)
        selections[str(end)]=selected;models[end]=model;heldouts[end]=held
    results={};ledgers={}
    for end in (2024,2025):
        held=heldouts[end];selected=selections[str(end)];config=selected['config']
        held['p_recency']=models[end].predict_proba(core.features(held,config))[:,1]
        cutoff=selected['chosen']['threshold'];held['direction_cutoff']=cutoff;held['trained_through']=end
        held['recency_call']=held.p_recency.ge(cutoff).astype(int)
        held['recency_correct']=held.recency_call.eq(held.target_binary)
        held.to_csv(PREFIX.with_name(PREFIX.name+f'_through{end}_predictions.csv'),index=False)
        results[str(end)]=dict(training_n=len(selected['training_ids']),test_n=len(held),metrics=summarize(held.target_binary,held.p_recency,cutoff),
            by_year={str(int(y)):summarize(g.target_binary,g.p_recency,cutoff) for y,g in held.groupby('year')})
        ledgers[end]=held
    annual=pd.concat([ledgers[2024][ledgers[2024].year.eq(2025)],ledgers[2025]],ignore_index=True)
    if set(annual.ipo_id)!=set(original.index) or annual.ipo_id.duplicated().any():
        raise ValueError('Changed annual cohort')
    annual.to_csv(PREFIX.with_name(PREFIX.name+'_annual_replay.csv'),index=False)
    previous_fn=original[original.target_binary.eq(1)&original.blended_call.eq(0)].index
    review=annual[annual.ipo_id.isin(previous_fn)]
    review.to_csv(PREFIX.with_name(PREFIX.name+'_20_missed_positive_review.csv'),index=False)
    paths=[source,ROOT/'ml/research_module1_recency.py',ROOT/'output/module1_recency_protocol_20261025.md']
    result=dict(selections=selections,results=results,trace=trace,
        annual=dict(n=len(annual),correct=int(annual.recency_correct.sum()),accuracy_pct=float(annual.recency_correct.mean()*100)),
        previous20corrected=int(review.recency_correct.sum()),live_promoted=False,
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):core.sha(p) for p in paths})
    PREFIX.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(selected={k:s['config']|{'cutoff':s['chosen']['threshold']} for k,s in selections.items()},
        results=results,annual=result['annual'],previous20corrected=result['previous20corrected']),indent=2))


if __name__=='__main__':
    run()
