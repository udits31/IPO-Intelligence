"""Training-only selection of a financial extension with downside-recall guard."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

import ml.research_module1_downside_fix as search
from ml.collect_module1_public_financials import OUT as FINANCIALS,FINANCIAL
from ml.evaluate_module1_later_holdout import validate
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain

ROOT=search.ROOT
PREFIX=ROOT/'output/module1_financial_fix_20261024'
BASE=list(search.FEATURES)
SETS={'valuation_simple':BASE+['ipo_earnings_yield_pct','annual_loss','revenue_growth_asinh','financial_sector'],
      'financial_full':BASE+FINANCIAL}


def choose(configs,validation):
    if not validation.year.isin([2022,2023,2024]).all():
        raise ValueError('Later labels cannot select model')
    y=validation.target_binary.to_numpy();rows=[]
    for config in configs:
        p=validation[config['name']].to_numpy()
        for threshold in search.THRESHOLDS:
            call=p>=threshold;negative_recall=float((~call[y==0]).mean())
            positive_recall=float(call[y==1].mean())
            rows.append(dict(name=config['name'],threshold=float(threshold),n=len(y),
                accuracy=float((call==y).mean()),negative_recall=negative_recall,
                balanced_accuracy=(negative_recall+positive_recall)/2,
                brier=float(np.mean((p-y)**2)),n_features=len(SETS[config['feature_set']]),
                simplicity=0 if config['family']=='logistic' else 1 if config['family']=='gradient_boosting' else 2))
    eligible=[r for r in rows if r['negative_recall']>=.5]
    ordered=sorted(eligible or rows,key=lambda r:(-r['accuracy'] if eligible else -r['balanced_accuracy'],
        -r['balanced_accuracy'],r['brier'],r['n_features'],r['simplicity'],abs(r['threshold']-.5),r['name']))
    return ordered[0],ordered


def run():
    financial_input_sha256=hashlib.sha256(FINANCIALS.read_bytes()).hexdigest()
    train=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv')
    test=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv')
    facts=pd.read_csv(FINANCIALS)
    if facts.ipo_id.duplicated().any() or set(facts.ipo_id)!=set(train.ipo_id)|set(test.ipo_id):
        raise ValueError('Financial collection must cover every train/test issuer, including explicit missing records')
    fields=['ipo_id','financial_status','annual_period',*FINANCIAL]
    train=train.merge(facts[fields],on='ipo_id',how='left',validate='one_to_one',suffixes=('_old',''))
    test=test.merge(facts[fields],on='ipo_id',how='left',validate='one_to_one',suffixes=('_old',''))
    for rows in (train,test):
        for field in ('listing_date','decision_market_date','issue_close_date'):
            rows[field]=pd.to_datetime(rows[field])
        observed=rows.annual_period.notna()
        if not pd.to_datetime(rows.loc[observed,'annual_period']).lt(rows.loc[observed,'decision_market_date']).all():
            raise ValueError('Future financial period')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    validate(train,test,canonical)
    configs=[dict(config,feature_set=name,name=name+'_'+config['name']) for name in SETS for config in search.candidates()]
    trace=[];folds=[]
    for year in (2022,2023,2024):
        held=train[train.year.eq(year)].copy();first=held.decision_market_date.min()
        earlier=train[train.year.lt(year)&train.listing_date.lt(first)]
        if set(earlier.ipo_id)&set(held.ipo_id) or earlier.listing_date.max()>=first:
            raise ValueError('Temporal validation leak')
        for config in configs:
            search.FEATURES=SETS[config['feature_set']]
            model=search.estimator(config).fit(search.feature_frame(earlier,config),earlier.target_binary)
            held[config['name']]=model.predict_proba(search.feature_frame(held,config))[:,1]
        folds.append(held)
        trace.append(dict(year=year,train_ids=sorted(map(int,earlier.ipo_id)),validation_ids=sorted(map(int,held.ipo_id)),
            latest_training_outcome=str(earlier.listing_date.max().date()),first_validation_decision=str(first.date())))
        print(f'financial validation {year} complete',flush=True)
    validation=pd.concat(folds,ignore_index=True)
    selected,ranked=choose(configs,validation)
    config=next(c for c in configs if c['name']==selected['name'])
    selection=dict(selected=selected,config=config,ranked=ranked,trace=trace,test_used_for_selection=False)
    selection_path=PREFIX.with_name(PREFIX.name+'_selection.json')
    selection_path.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf8')
    validation.to_csv(PREFIX.with_name(PREFIX.name+'_validation.csv'),index=False)
    search.FEATURES=SETS[config['feature_set']]
    model=search.estimator(config).fit(search.feature_frame(train,config),train.target_binary)
    if hashlib.sha256(FINANCIALS.read_bytes()).hexdigest()!=financial_input_sha256:
        raise ValueError('Financial collection changed during fit; rerun on a stable snapshot')
    artifact=ROOT/'ml/artifacts/module1_financial_fix_research_20261024.joblib'
    joblib.dump(dict(model=model,config=config,features=search.FEATURES,threshold=selected['threshold'],
        training_ids=sorted(map(int,train.ipo_id)),selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        research_only=True,source_publication_time_certified=False),artifact)
    # No fitting or candidate selection after this line.
    test['p_financial_fix']=model.predict_proba(search.feature_frame(test,config))[:,1]
    test['financial_fix_call']=test.p_financial_fix.ge(selected['threshold']).astype(int)
    result=summarize(test.target_binary,test.p_financial_fix,selected['threshold'])
    baselines={name:summarize(test.target_binary,test[name],.5) for name in ('p_combined','p_gmp_sign','p_gmp_fitted')}
    gains={name:paired_gain(test.financial_fix_call.eq(test.target_binary),test[name].ge(.5).eq(test.target_binary))
           for name in baselines}
    old_fp=test.target_binary.eq(0)&test.p_combined.ge(.5)
    test['original_false_positive']=old_fp
    test['corrected_original_false_positive']=old_fp&test.financial_fix_call.eq(0)
    test['financial_fix_correct']=test.financial_fix_call.eq(test.target_binary)
    ledger=PREFIX.with_name(PREFIX.name+'_predictions.csv');test.to_csv(ledger,index=False)
    test[~test.financial_fix_correct].to_csv(PREFIX.with_name(PREFIX.name+'_errors.csv'),index=False)
    yearly={str(int(y)):summarize(g.target_binary,g.p_financial_fix,selected['threshold']) for y,g in test.groupby('year')}
    coverage={name:dict(n=len(rows),annual_tables=int(rows.financial_status.eq('period_checked_public_ipo_table').sum()),
        feature_observed={field:int(rows[field].notna().sum()) for field in FINANCIAL}) for name,rows in [('train',train),('test',test)]}
    cross=pd.read_csv(ROOT/'data/audit/module1_later_bid_crosscheck_20261024.csv').set_index('ipo_id')
    if set(cross.index)!=set(test.ipo_id):
        raise ValueError('Whole-cohort source sensitivity incomplete')
    sensitivity=test.copy()
    for index,row in sensitivity.iterrows():
        source=cross.loc[row.ipo_id]
        if source.status in ('corroborated','conflicting_sources'):
            for category in ('qib','hni','retail'):
                sensitivity.loc[index,'log_'+category]=np.log1p(max(.01,float(source[category+'_subscription'])))
    sensitivity['p_source_checked']=model.predict_proba(search.feature_frame(sensitivity,config))[:,1]
    sensitivity_result=summarize(sensitivity.target_binary,sensitivity.p_source_checked,selected['threshold'])
    paths=[ROOT/'ml/research_module1_financial_fix.py',ROOT/'ml/research_module1_downside_fix.py',
        ROOT/'ml/collect_module1_public_financials.py',ROOT/'output/module1_financial_fix_protocol_20261024.md',
        FINANCIALS,selection_path,artifact,ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv',
        ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv',
        ROOT/'data/audit/module1_later_bid_crosscheck_20261024.csv']
    report=dict(selected=selected,config=config,features=search.FEATURES,training_n=len(train),test_n=len(test),
        training_ids=sorted(map(int,train.ipo_id)),test_ids=sorted(map(int,test.ipo_id)),results=result,baselines=baselines,
        paired_gains=gains,by_year=yearly,coverage=coverage,source_sensitivity=sensitivity_result,
        corrected_original_36=int(test.corrected_original_false_positive.sum()),
        newly_missed_positive=int((test.target_binary.eq(1)&test.p_combined.ge(.5)&test.financial_fix_call.eq(0)).sum()),
        live_promoted=False,input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        artifact=str(artifact.relative_to(ROOT)),limitations=[
            'Later-year errors motivated this extension, so the test is reused exploratory evidence, not independent confirmation.',
            'All model, preprocessing and cutoff fitting/selection uses 2021–2024 only.',
            'Financial periods precede decisions, but public historical pages do not certify original fact publication times.',
            'Primary test bids and all 155 IPOs are unchanged; source substitutions are sensitivity only.',
            'A trained cutoff changes the call policy, not probability calibration. No live or saved forecast was overwritten.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    lines=['# Module 1: financial extension and downside errors','',
        f'Training: {len(train)} IPOs, 2021–2024 only. Test: the same {len(test)} IPOs, 2025–2026. No overlap.',
        f'Selected using only earlier-year chronological validation: {config}. Cutoff: {selected["threshold"]}.',
        '', '| Model | Correct / 155 | Accuracy | Flat/negative caught | Positive caught | AUC | Brier |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,item in [*baselines.items(),('financial_extension',result)]:
        lines.append(f'| {name} | {item["tn"]+item["tp"]}/155 | {item["classifier_accuracy_pct"]:.1f}% | {item["tn"]}/49 | {item["tp"]}/106 | {item["roc_auc"]} | {item["brier_score"]} |')
    lines += ['',f'Corrected {report["corrected_original_36"]} of the original 36 false positives; newly missed {report["newly_missed_positive"]} previously correct positive openings.',
        '',f'Paired comparisons: {gains}', '',f'Yearly results: {yearly}', '',f'Financial coverage: {coverage}',
        '',f'Source-cross-checked sensitivity: {sensitivity_result}', '', '## Limits','',*[f'- {v}' for v in report['limitations']],
        '',f'Every original and revised call, actual opening and correctness flag: `{ledger.name}`.']
    PREFIX.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(config=config,selected=selected,results=result,coverage=coverage,
                         corrected_original_36=report['corrected_original_36'],newly_missed_positive=report['newly_missed_positive']),indent=2),flush=True)


if __name__=='__main__':
    run()
