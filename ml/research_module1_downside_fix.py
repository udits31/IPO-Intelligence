"""Investigate false positives; select fixes on earlier-year validation only."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.collect_module1_later_holdout import ROOT
from ml.evaluate_module1_later_holdout import validate
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain

PREFIX=ROOT/'output/module1_downside_fix_20261024'
FEATURES=['log_qib','log_retail','log_hni','log_offer_cap_price','log_issue_size','gmp_pct']
THRESHOLDS=np.round(np.arange(.35,.851,.025),3)


def candidates():
    configs=[]
    for transform in ('raw','signed_log'):
        for c in (.05,.5,5.):
            for weight in (None,'balanced'):
                configs.append(dict(family='logistic',transform=transform,C=c,class_weight=weight))
    for depth in (1,2):
        for trees in (80,160):
            configs.append(dict(family='gradient_boosting',transform='raw',max_depth=depth,n_estimators=trees))
    for depth in (3,5):
        for weight in (None,'balanced'):
            configs.append(dict(family='random_forest',transform='raw',max_depth=depth,class_weight=weight))
    for leaves in (3,7):
        for weight in (None,'balanced'):
            configs.append(dict(family='histogram_boosting',transform='raw',max_leaf_nodes=leaves,class_weight=weight))
    for i,config in enumerate(configs):
        config['name']=f'candidate_{i:02d}'
    return configs


def feature_frame(rows,config):
    x=rows[FEATURES].copy()
    if config['transform']=='signed_log':
        x['gmp_pct']=np.sign(x.gmp_pct)*np.log1p(np.abs(x.gmp_pct))
    return x


def estimator(config):
    family=config['family']
    if family=='logistic':
        model=LogisticRegression(C=config['C'],class_weight=config['class_weight'],max_iter=3000,random_state=42)
    elif family=='gradient_boosting':
        model=GradientBoostingClassifier(n_estimators=config['n_estimators'],max_depth=config['max_depth'],
                min_samples_leaf=10,learning_rate=.04,random_state=42)
    elif family=='random_forest':
        model=RandomForestClassifier(n_estimators=250,max_depth=config['max_depth'],min_samples_leaf=8,
                class_weight=config['class_weight'],random_state=42,n_jobs=2)
    elif family=='histogram_boosting':
        model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=config['max_leaf_nodes'],
                min_samples_leaf=15,l2_regularization=5,class_weight=config['class_weight'],random_state=42,
                early_stopping=False)
    else:
        raise ValueError('Unknown candidate')
    return Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                     ('scale',StandardScaler()),('model',model)])


def choose(configs,validation):
    """This selector accepts only pre-2025 validation rows."""
    if not validation.year.isin([2022,2023,2024]).all():
        raise ValueError('Later-year labels cannot select model or cutoff')
    results=[]
    for config in configs:
        p=validation[config['name']].to_numpy(); y=validation.target_binary.to_numpy()
        for threshold in THRESHOLDS:
            call=p>=threshold
            results.append(dict(name=config['name'],threshold=float(threshold),
                correct=int((call==y).sum()),n=len(y),accuracy=float(accuracy_score(y,call)),
                balanced_accuracy=float(balanced_accuracy_score(y,call)),brier=float(brier_score_loss(y,p)),
                simplicity=0 if config['family']=='logistic' else 1 if config['family']=='gradient_boosting' else 2))
    ranked=sorted(results,key=lambda r:(-r['accuracy'],-r['balanced_accuracy'],r['brier'],r['simplicity'],
                                      abs(r['threshold']-.5),r['name'],r['threshold']))
    return ranked[0],ranked


def profile(rows):
    result={}
    for label,group in rows.groupby(pd.cut(rows.gmp_pct,[-np.inf,0,5,10,20,np.inf]),observed=True):
        result[str(label)]=dict(n=len(group),nonpositive=int(group.target_binary.eq(0).sum()),
                               wrong_positive=int((group.target_binary.eq(0)&group.p_combined.ge(.5)).sum()))
    return result


def run():
    train=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv')
    test=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    for rows in (train,test):
        for field in ('listing_date','decision_market_date','issue_close_date'):
            rows[field]=pd.to_datetime(rows[field])
    validate(train,test,canonical)
    configs=candidates();trace=[];folds=[]
    for year in (2022,2023,2024):
        held=train[train.year.eq(year)].copy(); first=held.decision_market_date.min()
        earlier=train[train.year.lt(year)&train.listing_date.lt(first)].copy()
        if set(earlier.ipo_id)&set(held.ipo_id) or earlier.listing_date.max()>=first:
            raise ValueError('Chronological validation leak')
        trace.append(dict(year=year,training_ids=sorted(map(int,earlier.ipo_id)),validation_ids=sorted(map(int,held.ipo_id)),
                          latest_training_outcome=str(earlier.listing_date.max().date()),first_validation_decision=str(first.date())))
        for config in configs:
            model=estimator(config).fit(feature_frame(earlier,config),earlier.target_binary)
            held[config['name']]=model.predict_proba(feature_frame(held,config))[:,1]
        folds.append(held)
        print(f'finished earlier-year validation {year}: train={len(earlier)} validation={len(held)}',flush=True)
    validation=pd.concat(folds,ignore_index=True)
    selected,ranked=choose(configs,validation)
    config=next(c for c in configs if c['name']==selected['name'])
    # Save the early-year selection before any new test-model probabilities.
    selection=dict(selected=selected,config=config,validation_trace=trace,ranked=ranked,
                   test_used_for_selection=False,training_years=[2021,2022,2023,2024])
    selection_path=PREFIX.with_name(PREFIX.name+'_selection.json')
    selection_path.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf8')
    validation.to_csv(PREFIX.with_name(PREFIX.name+'_validation.csv'),index=False)
    model=estimator(config).fit(feature_frame(train,config),train.target_binary)
    artifact=ROOT/'ml/artifacts/module1_downside_fix_research_20261024.joblib'
    joblib.dump(dict(model=model,config=config,threshold=selected['threshold'],training_ids=sorted(map(int,train.ipo_id)),
                     features=FEATURES,research_only=True,selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest()),artifact)
    test['p_selected_fix']=model.predict_proba(feature_frame(test,config))[:,1]
    test['fixed_call']=test.p_selected_fix.ge(selected['threshold']).astype(int)
    results=dict(previous=summarize(test.target_binary,test.p_combined,.5),
                 gmp_sign=summarize(test.target_binary,test.p_gmp_sign,.5),
                 gmp_fitted=summarize(test.target_binary,test.p_gmp_fitted,.5),
                 selected_fix=summarize(test.target_binary,test.p_selected_fix,selected['threshold']),
                 selected_fix_at_point5=summarize(test.target_binary,test.p_selected_fix,.5))
    gain=paired_gain(test.fixed_call.eq(test.target_binary),test.p_combined.ge(.5).eq(test.target_binary))
    test['previous_false_positive']=test.target_binary.eq(0)&test.p_combined.ge(.5)
    test['corrected_previous_false_positive']=test.previous_false_positive&test.fixed_call.eq(0)
    test['newly_missed_positive']=test.target_binary.eq(1)&test.p_combined.ge(.5)&test.fixed_call.eq(0)
    previous=joblib.load(ROOT/'ml/artifacts/module1_combined_train_2021_2024_20261024.joblib')['model']
    names=previous.named_steps['missing'].get_feature_names_out(FEATURES)
    scaled=previous.named_steps['scale'].transform(previous.named_steps['missing'].transform(test[FEATURES]))
    contribution=scaled*previous.named_steps['model'].coef_[0]
    for i,name in enumerate(names):
        test['old_logodds_'+name]=contribution[:,i]
    old_logit=contribution.sum(axis=1)+previous.named_steps['model'].intercept_[0]
    if not np.allclose(1/(1+np.exp(-old_logit)),test.p_combined,atol=1e-12):
        raise ValueError('Previous model attribution did not reconstruct predictions')
    test.to_csv(PREFIX.with_name(PREFIX.name+'_predictions.csv'),index=False)
    test[test.previous_false_positive].to_csv(PREFIX.with_name(PREFIX.name+'_36_false_positives.csv'),index=False)
    # Secondary source check never chooses model settings or changes the primary ledger.
    cross_path=ROOT/'data/audit/module1_later_bid_crosscheck_20261024.csv'
    cross_result=None
    if cross_path.exists():
        cross=pd.read_csv(cross_path).set_index('ipo_id')
        if set(cross.index)!=set(test.ipo_id):
            raise ValueError('Incomplete whole-cohort source audit')
        sensitivity=test.copy(); changed=[]
        for index,row in sensitivity.iterrows():
            source=cross.loc[row.ipo_id]
            if source.status not in ('corroborated','conflicting_sources'):
                continue
            for category in ('qib','hni','retail'):
                value=float(source[category+'_subscription'])
                sensitivity.loc[index,category+'_subscription']=value
                sensitivity.loc[index,'log_'+category]=np.log1p(max(.01,value))
            if source.status=='conflicting_sources':
                changed.append(int(row.ipo_id))
        sensitivity['p_source_checked_fix']=model.predict_proba(feature_frame(sensitivity,config))[:,1]
        sensitivity['p_source_checked_previous']=previous.predict_proba(sensitivity[FEATURES])[:,1]
        sensitivity.to_csv(PREFIX.with_name(PREFIX.name+'_source_sensitivity.csv'),index=False)
        cross_result=dict(status_counts=cross.status.value_counts().to_dict(),conflicting_issuer_ids=changed,
            fixed_model=summarize(sensitivity.target_binary,sensitivity.p_source_checked_fix,selected['threshold']),
            previous_model=summarize(sensitivity.target_binary,sensitivity.p_source_checked_previous,.5),
            interpretation='IPOWatch preferred consistently where verified; unresolved provider conflicts remain uncertainty, not proven corrections.')
    # Predeclared diagnostics only: do not choose a later-year winner.
    diagnostics={}
    for candidate in configs:
        best=next(r for r in ranked if r['name']==candidate['name'])
        fitted=model if candidate['name']==config['name'] else estimator(candidate).fit(feature_frame(train,candidate),train.target_binary)
        p=fitted.predict_proba(feature_frame(test,candidate))[:,1]
        diagnostics[candidate['name']]=dict(config=candidate,validation=best,test=summarize(test.target_binary,p,best['threshold']))
    sources=[ROOT/'ml/research_module1_downside_fix.py',ROOT/'ml/audit_module1_later_bids.py',
             ROOT/'output/module1_downside_fix_protocol_20261024.md',
             ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv',
             ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv',selection_path,artifact]
    if cross_path.exists():
        sources.append(cross_path)
    report=dict(selection=selection,results=results,paired_change_vs_previous=gain,
                training_n=len(train),test_n=len(test),test_ids=sorted(map(int,test.ipo_id)),training_ids=sorted(map(int,train.ipo_id)),
                original_36_corrected=int(test.corrected_previous_false_positive.sum()),
                newly_missed_positive=int(test.newly_missed_positive.sum()),
                class_shift=dict(training_positive_rate=float(train.target_binary.mean()),test_positive_rate=float(test.target_binary.mean())),
                previous_logistic_intercept=float(previous.named_steps['model'].intercept_[0]),
                false_positive_gmp_profile=profile(test),source_sensitivity=cross_result,
                diagnostic_candidates_not_selected_on_test=diagnostics,artifact=str(artifact.relative_to(ROOT)),
                input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                limitations=['This is retrospective development after later-year errors were inspected; independent prospective confirmation is still required.',
                    'No later-year labels enter fit, model selection or cutoff selection; the selected model need not be the best later-year diagnostic.',
                    'Changing cutoff changes decisions, not underlying probability calibration. Report probability metrics separately.',
                    'Same 155 test issuers retained, including all losses and flat opens. No live promotion, paper or Module 2 changes.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    render(report)
    print(json.dumps(dict(config=config,threshold=selected['threshold'],results=results,
        paired_gain=gain,original_36_corrected=report['original_36_corrected'],
        newly_missed_positive=report['newly_missed_positive'],source_sensitivity=cross_result),indent=2),flush=True)


def render(report):
    rows=report['results']; old=rows['previous']; new=rows['selected_fix']
    lines=['# Module 1 downside investigation: fixed 2025–2026 test IPOs','',
        f'Trained on {report["training_n"]} eligible 2021–2024 IPOs. Evaluated on exactly the same {report["test_n"]} later-year IPOs.',
        f'Selected on earlier-year chronological validation: {report["selection"]["config"]}; decision cutoff {report["selection"]["selected"]["threshold"]}.','',
        '| Approach | Accuracy | Flat/negative caught | Positive caught | AUC | Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for key,value in rows.items():
        lines.append(f'| {key} | {value["classifier_accuracy_pct"]:.1f}% | {value["tn"]}/{value["nonpositive_listings"]} | {value["tp"]}/{value["positive_listings"]} | {value["roc_auc"]} | {value["brier_score"]} |')
    lines += ['', '## Why the previous calls failed','',
        f'The training positive rate was {report["class_shift"]["training_positive_rate"]*100:.1f}%, compared with {report["class_shift"]["test_positive_rate"]*100:.1f}% in this test. The previous logistic intercept was {report["previous_logistic_intercept"]:.3f}; positive-demand inputs often outweighed weak GMP. This is an observed model behaviour, not proof of a causal market mechanism.',
        f'The training-selected fix corrects {report["original_36_corrected"]} of the original 36 false positives. It newly misses {report["newly_missed_positive"]} previously correct positive openings. Net accuracy change: {report["paired_change_vs_previous"]}.',
        'Each of the 36 wrong positive calls has a row containing its inputs, actual opening return, old feature contributions, new probability and new call in the companion CSV.',
        '', '## Independent source check','',json.dumps(report['source_sensitivity'],indent=2),
        '', '## Limits','',*[f'- {v}' for v in report['limitations']],
        '', 'All 24 candidate diagnostics and the complete earlier-year selection trace are retained in the JSON. Later-year results do not choose the winner.']
    PREFIX.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    run()
