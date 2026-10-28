"""New-source/temporal challenger and complete error audit; no live promotion."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ml.complete_module1_benchmark import (ROOT,prepare,attach_quotes,TRACKS,SectorRelativeFacts,
    return_metrics,rankings,paired_mean)
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain
from ml.recover_module1_other_sources import OUT as QUOTES,AUDIT
from ml.recover_module1_bid_history import OUT as BIDS

OUT=ROOT/'output/module1_other_source_benchmark_20261024.json'
TEMPORAL=['gmp_quote_age_days','gmp_slope_pp_per_day','log_qib_growth','log_hni_growth','log_retail_growth']
VARIANTS={**TRACKS,'temporal_bids_gmp':TRACKS['bids_offer_gmp']+TEMPORAL}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal_features(frame,quotes,growth):
    result=frame.copy()
    history=quotes.copy();history['gmp_date']=pd.to_datetime(history.gmp_date,errors='coerce')
    for key in TEMPORAL:
        result[key]=np.nan
    result['bid_growth_status']='missing'
    growth=growth.set_index('ipo_id')
    for index,row in result.iterrows():
        if pd.notna(row.gmp_pct):
            date=pd.Timestamp(row.gmp_date);decision=pd.Timestamp(row.decision_market_date)
            if not date<decision:
                raise ValueError('Future quote in temporal features')
            result.loc[index,'gmp_quote_age_days']=(decision-date).days
            earlier=history[history.ipo_id.eq(row.ipo_id)&history.source_url.eq(row.gmp_source_url)&
                history.gmp_date.lt(date)&history.gmp_date.ge(date-pd.Timedelta(days=7))].copy()
            conflicts=earlier.groupby('gmp_date').gmp_rs.nunique()
            earlier=earlier[~earlier.gmp_date.isin(conflicts[conflicts.gt(1)].index)]
            if len(earlier):
                prior=earlier.sort_values('gmp_date').iloc[-1]
                previous_pct=100*float(prior.gmp_rs)/np.expm1(row.log_offer_cap_price)
                result.loc[index,'gmp_slope_pp_per_day']=(row.gmp_pct-previous_pct)/(date-prior.gmp_date).days
        if row.ipo_id in growth.index:
            observed=growth.loc[row.ipo_id]
            result.loc[index,'bid_growth_status']=observed.status
            if observed.status=='matched_penultimate_table':
                for key in TEMPORAL[2:]:
                    result.loc[index,key]=observed[key]
    return result


def fitted(track,kind):
    steps=[('relative',SectorRelativeFacts())] if track=='sector_relative_gmp' else []
    steps += [('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]
    model=(LogisticRegression(C=.5,max_iter=2000,random_state=42) if kind=='direction' else
        Ridge(alpha=20) if kind=='ridge' else GradientBoostingRegressor(loss='huber',n_estimators=100,
        max_depth=2,min_samples_leaf=10,learning_rate=.04,random_state=42))
    return Pipeline(steps+[('model',model)])


def run():
    audit=json.loads(AUDIT.read_text())
    if audit['completed']!=audit['attempted'] or audit['attempted']!=63:
        raise ValueError('Other-source audit not complete for all remaining 63')
    base,canonical,old_quotes=prepare()
    growth=pd.read_csv(BIDS)
    if set(growth.ipo_id)!=set(base.ipo_id):
        raise ValueError('Bid-history recovery incomplete')
    additional=pd.read_csv(QUOTES)
    quotes=pd.concat([old_quotes,additional],ignore_index=True)
    before=base.set_index('ipo_id')
    frame=attach_quotes(base,quotes,canonical)
    frame['original_gmp_covered']=frame.ipo_id.map(before.original_gmp_covered)
    frame['previous_gmp_covered']=frame.ipo_id.map(before.gmp_pct.notna())
    frame=temporal_features(frame,quotes,growth)
    frame['gmp_low_pct']=frame.gmp_pct;frame['gmp_high_pct']=frame.gmp_pct
    frame['gmp_quote_basis']=np.where(frame.gmp_pct.notna(),'published_single_value','missing')
    for index,row in frame[frame.gmp_pct.notna()].iterrows():
        selected=quotes[quotes.ipo_id.eq(row.ipo_id)&quotes.source_url.eq(row.gmp_source_url)&
            pd.to_datetime(quotes.gmp_date).eq(pd.Timestamp(row.gmp_date))]
        if 'gmp_basis' in selected and selected.gmp_basis.eq('published_range_midpoint').any():
            quote=selected[selected.gmp_basis.eq('published_range_midpoint')].iloc[-1]
            cap=np.expm1(row.log_offer_cap_price)
            frame.loc[index,'gmp_low_pct']=100*float(quote.gmp_low_rs)/cap
            frame.loc[index,'gmp_high_pct']=100*float(quote.gmp_high_rs)/cap
            frame.loc[index,'gmp_quote_basis']='derived_published_range_midpoint'
    if frame.loc[frame.previous_gmp_covered,'gmp_pct'].isna().any():
        raise ValueError('New source conflicts invalidated an existing comparison; must be adjudicated visibly')
    paired=frame[frame.gmp_pct.notna()].copy();folds=[];trace=[];models={}
    for year in (2022,2023,2024):
        test=paired[paired.year.eq(year)].copy();first=test.decision_market_date.min()
        train=paired[paired.year.lt(year)&paired.listing_date.lt(first)]
        if set(train.ipo_id)&set(test.ipo_id):
            raise ValueError('Train/test issuer overlap')
        for track,features in VARIANTS.items():
            classifier=fitted(track,'direction').fit(train[features],train.target_binary)
            test['p_'+track]=classifier.predict_proba(test[features])[:,1]
            models[(year,track,'direction')]=classifier
            for kind in ('ridge','huber'):
                model=fitted(track,kind).fit(train[features],train.target_return-train.gmp_pct)
                test['r_'+track+'_'+kind]=np.maximum(-100,test.gmp_pct+model.predict(test[features]))
                models[(year,track,kind)]=model
        folds.append(test)
        trace.append(dict(test_year=year,train_ids=sorted(map(int,train.ipo_id)),test_ids=sorted(map(int,test.ipo_id)),
            latest_training_outcome=str(train.listing_date.max().date()),first_test_decision=str(first.date())))
    ledger=pd.concat(folds,ignore_index=True)
    dev=ledger[ledger.year.isin([2022,2023])];test=ledger[ledger.year.eq(2024)].copy()
    development_direction={t:summarize(dev.target_binary,dev['p_'+t],.5) for t in VARIANTS}
    return_names=['r_'+t+'_'+kind for t in VARIANTS for kind in ('ridge','huber')]
    development_returns={n:return_metrics(dev,n) for n in return_names}
    selected_direction=min([t for t in VARIANTS if t!='gmp_fitted'],key=lambda t:(development_direction[t]['brier_score'],t))
    selected_return=min([n for n in return_names if 'gmp_fitted' not in n],key=lambda n:(development_returns[n]['mae_pp'],n))
    selected_gmp_return=min([n for n in return_names if 'gmp_fitted' in n],key=lambda n:(development_returns[n]['mae_pp'],n))
    development_ranking={n:rankings(dev,n) for n in ['gmp_pct',*return_names]}
    def rank_key(name):
        item=development_ranking[name]
        complexity=0 if name=='gmp_pct' else len(VARIANTS[name[2:].rsplit('_',1)[0]])
        return (-(item['pairwise_accuracy_pct'] or 0),complexity,name)
    eligible_rank=[n for n,item in development_ranking.items() if item['panels']>=5 and item['pair_count']>=10]
    selected_ranking=min(eligible_rank,key=rank_key) if eligible_rank else 'gmp_pct'
    results={}
    for population,rows in [('original_39',test[test.original_gmp_covered]),
        ('previous_51',test[test.previous_gmp_covered]),('expanded',test)]:
        y=rows.target_binary.to_numpy();base_correct=rows.gmp_pct.gt(0).to_numpy()==y
        direction={t:dict(metrics=summarize(y,rows['p_'+t],.5),
            vs_gmp=paired_gain(rows['p_'+t].ge(.5).to_numpy()==y,base_correct)) for t in VARIANTS}
        direction['gmp_sign']=dict(metrics=summarize(y,rows.gmp_pct.gt(0).astype(float),.5))
        returns={n:return_metrics(rows,n) for n in ['gmp_pct','gmp_low_pct','gmp_high_pct',*return_names]}
        for n in returns:
            model_error=np.abs(rows.target_return-rows[n])
            calibrated_error=np.abs(rows.target_return-rows[selected_gmp_return])
            returns[n]['mae_improvement_vs_fitted_gmp']=paired_mean(calibrated_error-model_error)
        results[population]=dict(n=len(rows),ipo_ids=sorted(map(int,rows.ipo_id)),direction=direction,returns=returns,
            ranking={n:rankings(rows,n) for n in ['gmp_pct',*return_names]})
    if results['original_39']['n']!=39 or results['previous_51']['n']!=51:
        raise ValueError('Existing test denominators changed')
    # Error audit: later final demand is only an explanation field, never in X.
    training=paired[paired.year.lt(2024)&paired.listing_date.lt(test.decision_market_date.min())]
    features=VARIANTS[selected_direction]
    direction_model=models[(2024,selected_direction,'direction')]
    x=test[features]
    if selected_direction=='sector_relative_gmp':
        transformed=direction_model.named_steps['relative'].transform(x)
    else:
        transformed=x
    imputed=direction_model.named_steps['missing'].transform(transformed)
    names=direction_model.named_steps['missing'].get_feature_names_out(features)
    contributions=direction_model.named_steps['scale'].transform(imputed)*direction_model.named_steps['model'].coef_[0]
    logits=contributions.sum(axis=1)+direction_model.named_steps['model'].intercept_[0]
    if not np.allclose(1/(1+np.exp(-logits)),test['p_'+selected_direction]):
        raise ValueError('Error explanations do not reconstruct saved probabilities')
    errors=[]
    for pos,row in enumerate(test.to_dict('records')):
        effects=sorted(zip(names,contributions[pos]),key=lambda v:v[1])
        observed=training[features]
        extrapolated=[key for key in features if pd.notna(row[key]) and observed[key].notna().any() and
            (row[key]<observed[key].min() or row[key]>observed[key].max())]
        item={key:row[key] for key in ('ipo_id','company_name','decision_market_date','gmp_date','gmp_source_url','gmp_pct',
            'target_binary','target_return','financial_observed','offer_observed','gmp_quote_age_days','gmp_slope_pp_per_day','bid_growth_status',
            'gmp_low_pct','gmp_high_pct','gmp_quote_basis')}
        item.update(direction_probability=row['p_'+selected_direction],direction_correct=bool((row['p_'+selected_direction]>=.5)==row['target_binary']),
            gmp_correct=bool((row['gmp_pct']>0)==row['target_binary']),return_prediction=row[selected_return],
            return_error_pp=abs(row[selected_return]-row['target_return']),gmp_error_pp=abs(row['gmp_pct']-row['target_return']),
            extrapolated_features='; '.join(extrapolated),
            strongest_negative_effects='; '.join(f'{k}: {v:+.3f}' for k,v in effects[:3]),
            strongest_positive_effects='; '.join(f'{k}: {v:+.3f}' for k,v in effects[-3:]))
        if row['ipo_id'] in set(growth.ipo_id):
            later=growth.set_index('ipo_id').loc[row['ipo_id']]
            for category in ('qib','hni','retail'):
                item['postcutoff_final_'+category+'_diagnostic_only']=later.get('postcutoff_final_'+category)
                item['interim_'+category]=row[category+'_subscription']
        errors.append(item)
    all_coverage=frame[['ipo_id','company_name','year','decision_market_date','gmp_date','gmp_pct','gmp_source_url',
        'original_gmp_covered','previous_gmp_covered','gmp_conflict_days','financial_observed','offer_observed',*TEMPORAL,'bid_growth_status',
        'gmp_low_pct','gmp_high_pct','gmp_quote_basis']]
    all_coverage.to_csv(OUT.with_name(OUT.stem+'_coverage.csv'),index=False)
    ledger.to_csv(OUT.with_name(OUT.stem+'_predictions.csv'),index=False)
    pd.DataFrame(errors).to_csv(OUT.with_name(OUT.stem+'_errors.csv'),index=False)
    source_counts={}
    for item in audit['sources']:
        for provider in item['providers']:
            key=provider['provider']+': '+provider['status'];source_counts[key]=source_counts.get(key,0)+1
    gap_reasons=[];gap_reason_by_id={}
    quote_dates=quotes.copy();quote_dates['gmp_date']=pd.to_datetime(quote_dates.gmp_date,errors='coerce')
    for row in frame[frame.gmp_pct.isna()].itertuples():
        observed=quote_dates[quote_dates.ipo_id.eq(row.ipo_id)]
        prior=observed[observed.gmp_date.lt(row.decision_market_date)]
        reason=('conflicting_predecision_quotes' if row.gmp_conflict_days else
            'only_stale_quotes' if len(prior) else 'only_same_day_or_later_quotes' if len(observed) else 'no_verified_numeric_history')
        gap_reasons.append(reason);gap_reason_by_id[row.ipo_id]=reason
    all_coverage['gmp_gap_reason']=all_coverage.ipo_id.map(gap_reason_by_id).fillna('available')
    all_coverage['newly_recovered']=all_coverage.gmp_pct.notna()&~all_coverage.previous_gmp_covered
    all_coverage.to_csv(OUT.with_name(OUT.stem+'_coverage.csv'),index=False)
    all_coverage[~all_coverage.previous_gmp_covered].to_csv(OUT.with_name(OUT.stem+'_63_gap_results.csv'),index=False)
    report=dict(live_promoted=False,module2_changed=False,paper_changed=False,
        protocol='output/module1_other_source_protocol_20261024.md',selected_direction=selected_direction,
        selected_return=selected_return,selected_gmp_return=selected_gmp_return,
        selected_ranking=selected_ranking,development_ranking=development_ranking,
        ranking_selection_rule='2022/2023 pairwise ordering accuracy, minimum 5 panels and 10 pairs; ties prefer fewer inputs then name; insufficient evidence falls back to raw GMP',
        development_direction=development_direction,development_returns=development_returns,trace=trace,results=results,
        coverage=dict(previous_paired_n=int(frame.previous_gmp_covered.sum()),paired_n=len(paired),
            recovered_n=int((frame.gmp_pct.notna()&~frame.previous_gmp_covered).sum()),remaining_n=int(frame.gmp_pct.isna().sum()),
            other_source_history_issuers=audit['history_issuers'],source_status_counts=source_counts,
            bid_growth_usable_n=int(frame.bid_growth_status.eq('matched_penultimate_table').sum()),
            gmp_slope_usable_n=int(frame.gmp_slope_pp_per_day.notna().sum()),
            selected_published_range_n=int(frame.gmp_quote_basis.eq('derived_published_range_midpoint').sum()),
            remaining_gap_reasons=pd.Series(gap_reasons).value_counts().to_dict(),
            unmatched_issuer_ids=sorted(map(int,frame.loc[frame.gmp_pct.isna(),'ipo_id']))),
        limitations=['All 2024 outcomes have been inspected; this is exploratory retrospective research, not independent prospective proof.',
            'Different publishers can disagree. Conflicting same-date values are withheld, not averaged or selected by outcome.',
            'Only pre-decision public dated rows enter features. Paid history and access blocks are not bypassed.',
            'Final demand in the error CSV is post-cutoff diagnostic context only; never a prediction feature.',
            'Conditional opening returns are not executable/allotment-weighted portfolio profit.'])
    paths=[ROOT/'ml/module1_other_source_benchmark.py',ROOT/'ml/recover_module1_other_sources.py',ROOT/'ml/recover_module1_bid_history.py',
        ROOT/'ml/complete_module1_benchmark.py',ROOT/'ml/model_pipeline.py',ROOT/'ml/research_monitor_daily_learning.py',
        ROOT/report['protocol'],QUOTES,AUDIT,BIDS,ROOT/'output/module1_completion_20261024.json',
        ROOT/'output/module1_completion_inputs_20261024.json',ROOT/'data/audit/module1_completion_early_20261024.csv',ROOT/'ml/model_bundle.json']
    report['input_sha256']={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    artifact=ROOT/'ml/artifacts/module1_other_sources_research_20261024.joblib'
    artifact.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(research_only=True,validated_for_live=False,models=models,features=VARIANTS,
        fold_trace=trace,selected_direction=selected_direction,selected_return=selected_return,
        selected_ranking=selected_ranking,input_sha256=report['input_sha256']),artifact)
    report['research_artifact']=artifact.relative_to(ROOT).as_posix()
    report['research_artifact_sha256']=digest(artifact)
    OUT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    render(report,errors)
    print(json.dumps(dict(coverage=report['coverage'],selected_direction=selected_direction,selected_return=selected_return,
        summary={name:dict(n=value['n'],gmp_accuracy=value['direction']['gmp_sign']['metrics']['classifier_accuracy_pct'],
            model_accuracy=value['direction'][selected_direction]['metrics']['classifier_accuracy_pct'],
            gmp_mae=value['returns']['gmp_pct']['mae_pp'],model_mae=value['returns'][selected_return]['mae_pp'],
            fitted_gmp_mae=value['returns'][selected_gmp_return]['mae_pp']) for name,value in results.items()}),indent=2))
    return report


def render(report,errors):
    sd=report['selected_direction'];sr=report['selected_return'];sk=report['selected_ranking'];c=report['coverage']
    lines=['# Module 1: other-source recovery, temporal inputs and error diagnosis','',
        f"Usable GMP coverage increased from {c['previous_paired_n']} to {c['paired_n']}; {c['recovered_n']} additional eligible histories, "
        f"{c['remaining_n']} unresolved. All 63 prior gaps were attempted. Source failures and restrictions are retained, not silently dropped.",
        '',f"Remaining gap reasons: {json.dumps(c['remaining_gap_reasons'],sort_keys=True)}. "
        'A source with different numeric quotes on the same day is not arbitrarily preferred because it improves the score.',
        '', f"Matched early bid-growth histories: {c['bid_growth_usable_n']}; same-publisher pre-decision GMP slopes: {c['gmp_slope_usable_n']}.",
        '', f"{c['selected_published_range_n']} selected GMP inputs are explicitly published ranges. Both endpoints are retained; "
        'the labelled midpoint is a derived feature, not a claimed single observed quote. Lower/upper endpoint return-error scenarios are in the JSON.',
        '', f"Selected on 2022/2023 only: direction **{sd}**, returns **{sr}**. Fixed thresholds and model parameters. "
        '2024 has been inspected before; this is exploratory research, not blind confirmation.',
        '',f"Separate ranking selector: **{sk}**. {report['ranking_selection_rule']}. "
        'This corrects the mismatch between choosing a model for average price error and choosing one for ordering IPOs; '
        'it is not a claim that ranking has improved.',
        '', '| Test set | IPOs | GMP-sign accuracy | Combined accuracy | GMP return error | Combined return error | Fitted GMP-only return error |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,result in report['results'].items():
        lines.append(f"| {name} | {result['n']} | {result['direction']['gmp_sign']['metrics']['classifier_accuracy_pct']:.1f}% | "
            f"{result['direction'][sd]['metrics']['classifier_accuracy_pct']:.1f}% | {result['returns']['gmp_pct']['mae_pp']:.2f} pp | "
            f"{result['returns'][sr]['mae_pp']:.2f} pp | {result['returns'][report['selected_gmp_return']]['mae_pp']:.2f} pp |")
    lines+=['', 'Return error means average absolute error in percentage points; lower is better. The original denominators remain separate.',
        '', '## Every fixed direction candidate', '', '| Test set | Inputs | Accuracy | Balanced accuracy | AUC |', '|---|---|---:|---:|---:|']
    for population,result in report['results'].items():
        for track,item in result['direction'].items():
            m=item['metrics'];lines.append(f"| {population} | {track} | {m['classifier_accuracy_pct']:.1f}% | {m['balanced_accuracy_pct']:.1f}% | {m['roc_auc']:.3f} |")
    lines+=['','## Paired uncertainty and ranking']
    for name,result in report['results'].items():
        d=result['direction'][sd]['vs_gmp'];r=result['returns'][sr];rank=result['ranking'][sr]
        task_rank=result['ranking'][sk]
        gain=r['mae_improvement_vs_gmp'];cal=r['mae_improvement_vs_fitted_gmp'];ci=d['bootstrap_95_pp']
        lines+=['',f'### {name}','',
            f"Direction difference: {d['change_pp']:+.2f} points (95% interval {ci[0]:+.2f} to {ci[1]:+.2f}); "
            f"{d['corrected_errors']} GMP mistakes corrected; {d['introduced_errors']} new mistakes introduced.",
            '', f"Return-error improvement versus raw GMP: {gain['mean']:+.2f} pp "
            f"(95% interval {gain['bootstrap_95'][0]:+.2f} to {gain['bootstrap_95'][1]:+.2f}). "
            f"Versus fitted GMP-only: {cal['mean']:+.2f} pp "
            f"(95% interval {cal['bootstrap_95'][0]:+.2f} to {cal['bootstrap_95'][1]:+.2f}).",
            '', f"Ranking: {rank['panels']} identical-date choice sets. Reliable edge established: {rank['established_edge']}.",
            '',f"Dedicated ranking selection: {sk}; pairwise ordering {task_rank['pairwise_accuracy_pct']}% "
            f"versus GMP {task_rank['gmp_pairwise_accuracy_pct']}%; selection difference "
            f"{task_rank['paired_selection_difference_pp']['mean']} pp. "
            f"Reliable dedicated-ranking edge established: {task_rank['established_edge']}.",
            '', '| Decision date | IPOs | Combined choice | GMP choice | Actual combined opening return | Actual GMP opening return |',
            '|---|---:|---|---|---:|---:|']
        for choice in rank['choices']:
            lines.append(f"| {choice['decision_date']} | {choice['n']} | {choice['model_choice']} | {choice['gmp_choice']} | "
                f"{choice['model_opening_return_pct']:+.2f}% | {choice['gmp_opening_return_pct']:+.2f}% |")
    lines+=['','## Where the selected direction model is wrong','',
        '| IPO | Dated GMP | Predicted positive probability | Actual opening return | Missing earnings / OFS | Out-of-training-range inputs |',
        '|---|---:|---:|---:|---|---|']
    for e in errors:
        if not e['direction_correct']:
            lines.append(f"| {e['company_name']} | {e['gmp_pct']:.1f}% | {e['direction_probability']*100:.1f}% | {e['target_return']:+.2f}% | "
                f"{not e['financial_observed']} / {not e['offer_observed']} | {e['extrapolated_features'] or 'None'} |")
    lines+=['','## Largest return misses','', '| IPO | Predicted return | Actual opening return | Dated GMP | Absolute error |', '|---|---:|---:|---:|---:|']
    for e in sorted(errors,key=lambda r:-r['return_error_pp'])[:10]:
        lines.append(f"| {e['company_name']} | {e['return_prediction']:+.2f}% | {e['target_return']:+.2f}% | "
            f"{e['gmp_pct']:+.2f}% | {e['return_error_pp']:.2f} pp |")
    lines+=['','The errors CSV contains every case, source clock, exact linear-model feature effects, missingness and training-range flags. '
        'Later final-demand fields are labelled diagnostic-only and never enter model training. These associations do not establish causation.',
        '', 'No production promotion or Module 2/paper changes. A numerical improvement is not called reliable when its interval includes no improvement.',
        '',*['- '+item for item in report['limitations']]]
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    run()
