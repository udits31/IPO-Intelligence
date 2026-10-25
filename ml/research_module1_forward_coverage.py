"""Retain all available early books; distinguish fixed and updated training."""
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor,as_completed

import joblib
import numpy as np
import pandas as pd

import ml.research_module1_forward_update as core
from ml.complete_module1_benchmark import prepare,attach_quotes
from ml.recover_module1_other_sources import OUT as EXTRA_QUOTES
from ml.research_module1_trajectory_fix import add_temporal
from ml.recover_module1_2025_training_gaps import CACHE as RECOVERY_CACHE,OUT as RECOVERY
from ml.collect_module1_public_financials import collect as financial_collect,extract as financial_extract
from ml.collect_module1_later_holdout import index_urls
from ml.fetch_daywise_subscriptions import table_days
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain

ROOT=core.ROOT
PREFIX=ROOT/'output/module1_forward_coverage_20261024'


def recovery_record(ipo_id):
    path=RECOVERY_CACHE/f'{ipo_id}_v2.json.gz'
    if not path.exists():
        path=RECOVERY_CACHE/f'{ipo_id}.json.gz'
    with gzip.open(path,'rt',encoding='utf8') as stream:
        return json.load(stream)


def extra_context(rows,records):
    rows=rows.copy()
    for field in core.TRAJECTORY:
        rows[field]=np.nan
    for index,row in rows.iterrows():
        record=records[int(row.ipo_id)]
        prior=None;first=None
        # Prefer the provider that generated this decision row, not a favourable discrepancy.
        if row.bid_source_url.startswith('https://ipowatch.in/'):
            for response in record.get('ipowatch_responses',[]):
                try:
                    detail=json.loads(response['raw_response_text'])
                except ValueError:
                    continue
                if not isinstance(detail,dict):
                    continue
                days=table_days(detail.get('content',{}).get('rendered',''))
                if days and len(days)>=3:
                    candidate=days[max(days)-1]
                    if all(abs(candidate[c]-row[c+'_subscription'])<=.02 for c in ('qib','hni','retail')):
                        first={c+'_subscription':days[min(days)][c] for c in ('qib','hni','retail')}
                        prior={c+'_subscription':candidate[c] for c in ('qib','hni','retail')}
                        break
        else:
            history=[h for h in record.get('bid_history',[]) if pd.Timestamp(h['date'])<=row.decision_market_date]
            if len(history)>=2 and pd.Timestamp(history[-1]['date'])==row.decision_market_date:
                if all(abs(history[-1][c+'_subscription']-row[c+'_subscription'])<=.02 for c in ('qib','hni','retail')):
                    first,prior=history[0],history[-1]
        if first is not None:
            for category in ('qib','hni','retail'):
                rows.loc[index,'log_'+category+'_growth']=np.log1p(prior[category+'_subscription'])-np.log1p(first[category+'_subscription'])
        if pd.notna(row.get('gmp_date')):
            selected=pd.Timestamp(row.gmp_date)
            rows.loc[index,'gmp_quote_age_days']=(row.decision_market_date-selected).days
            earlier=[]
            for source in record['sources']:
                if source.get('url')!=row.gmp_source_url:
                    continue
                earlier += [q for q in source.get('quotes',[]) if selected-pd.Timedelta(days=7)<=pd.Timestamp(q['gmp_date'])<selected]
            clean=[q for q in earlier if len({v['gmp_rs'] for v in earlier if v['gmp_date']==q['gmp_date']})==1]
            if clean:
                older=sorted(clean,key=lambda q:q['gmp_date'])[-1]
                previous_pct=100*older['gmp_rs']/np.expm1(row.log_offer_cap_price)
                rows.loc[index,'gmp_slope_pp_per_day']=(row.gmp_pct-previous_pct)/(selected-pd.Timestamp(older['gmp_date'])).days
    return rows


def frame():
    old,canonical,quotes=prepare()
    quotes=pd.concat([quotes,pd.read_csv(EXTRA_QUOTES)],ignore_index=True)
    old=attach_quotes(old,quotes,canonical)
    for field in ('listing_date','decision_market_date','issue_close_date','issue_open_date'):
        old[field]=pd.to_datetime(old[field])
    existing=pd.read_csv(ROOT/'output/module1_forward_update_20261024_features.csv')
    old_ids=set(existing[existing.year.le(2024)].ipo_id)
    new_old=old[~old.ipo_id.isin(old_ids)]
    urls=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
    financial=pd.read_csv(ROOT/'data/audit/module1_public_financials_20261024.csv')
    recovered=pd.read_csv(RECOVERY)
    if set(recovered.status)-{'paired_training_recovered','early_book_missing_gmp'}:
        raise ValueError('Not every missing 2025 issuer has a usable interim book')
    records={int(row.ipo_id):recovery_record(int(row.ipo_id)) for row in recovered.itertuples()}
    financial_extra=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[]
        for row in new_old.itertuples():
            futures.append(pool.submit(financial_collect,dict(ipo_id=int(row.ipo_id),company_name=row.company_name,
                issue_open_date=row.issue_open_date,issue_close_date=row.issue_close_date,
                issue_price=float(canonical.loc[row.ipo_id].price_band_upper),decision_market_date=row.decision_market_date),urls))
        for future in as_completed(futures):
            value=future.result();value.pop('raw_response_text',None)
            value['financial_sector']=float(canonical.loc[value['ipo_id']].sector=='Financial Services')
            financial_extra.append(value)
    old=add_temporal(old,True,quotes)
    for field in ('listing_date','decision_market_date','issue_close_date','issue_open_date'):
        recovered[field]=pd.to_datetime(recovered[field])
    recovered=extra_context(recovered,records)
    for row in recovered.itertuples():
        record=records[int(row.ipo_id)]
        source=next(s for s in record['sources'] if s['provider']=='ipoji' and s['status']=='verified_offer')
        value=financial_extract(source['raw_response_text'],row.decision_market_date,float(canonical.loc[row.ipo_id].price_band_upper))
        value.update(ipo_id=int(row.ipo_id),source_url=source['url'],source_sha256=source['source_sha256'],
            financial_sector=float(canonical.loc[row.ipo_id].sector=='Financial Services'))
        financial_extra.append(value)
    public=pd.concat([financial,pd.DataFrame(financial_extra)],ignore_index=True)
    public.to_csv(PREFIX.with_name(PREFIX.name+'_financial_inputs.csv'),index=False)
    public=public.rename(columns={c:'pub_'+c for c in public if c!='ipo_id'})
    original_test=pd.read_csv(ROOT/'output/module1_trajectory_fix_20261024_predictions.csv')
    combined=pd.concat([old,original_test,recovered],ignore_index=True)
    if combined.ipo_id.duplicated().any():
        raise ValueError('Duplicate expanded issuer')
    combined=combined.merge(public[['ipo_id','pub_annual_period',*core.FINANCIAL]],on='ipo_id',how='left',validate='one_to_one')
    for field in ('listing_date','decision_market_date','issue_close_date','issue_open_date'):
        combined[field]=pd.to_datetime(combined[field])
    observed=combined.gmp_date.notna()
    if not (pd.to_datetime(combined.loc[observed,'gmp_date'])<combined.loc[observed,'decision_market_date']).all():
        raise ValueError('Future GMP')
    observed=combined.pub_annual_period.notna()
    if not (pd.to_datetime(combined.loc[observed,'pub_annual_period'])<combined.loc[observed,'decision_market_date']).all():
        raise ValueError('Future financial period')
    core.PREFIX=PREFIX
    combined=core.market_features(combined)
    if not (pd.to_datetime(combined.macro_source_date)<combined.decision_market_date).all():
        raise ValueError('Future market close')
    combined.to_csv(PREFIX.with_name(PREFIX.name+'_features.csv'),index=False)
    return combined,set(original_test.ipo_id)


def run():
    data,test_ids=frame()
    configs=core.candidates()
    core.SETS['gmp_only']=['gmp_pct']
    configs.append(dict(name='gmp_only_control',family='logistic',transform='raw',C=.5,class_weight=None,feature_set='gmp_only'))
    paths=[ROOT/'ml/research_module1_forward_coverage.py',ROOT/'ml/research_module1_forward_update.py',
        ROOT/'ml/recover_module1_2025_training_gaps.py',ROOT/'output/module1_forward_coverage_protocol_20261024.md',
        PREFIX.with_name(PREFIX.name+'_features.csv'),PREFIX.with_name(PREFIX.name+'_financial_inputs.csv'),RECOVERY]
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):core.sha(p) for p in paths}
    folds=[];trace=[]
    for year in (2022,2023,2024,2025):
        held=data[data.year.eq(year)].copy();first=held.decision_market_date.min()
        earlier=data[data.year.lt(year)&data.listing_date.lt(first)]
        core.membership_guard(earlier,held,year-1)
        trace.append(dict(year=year,training_ids=sorted(map(int,earlier.ipo_id)),validation_ids=sorted(map(int,held.ipo_id)),
            latest_training_outcome=str(earlier.listing_date.max().date()),first_validation_decision=str(first.date())))
        probabilities={}
        for config in configs:
            model=core.estimator(config).fit(core.features(earlier,config),earlier.target_binary)
            probabilities[config['name']]=model.predict_proba(core.features(held,config))[:,1]
        folds.append(pd.concat([held,pd.DataFrame(probabilities,index=held.index)],axis=1))
        print(f'coverage validation {year}: train={len(earlier)} validation={len(held)}',flush=True)
    validation=pd.concat(folds,ignore_index=True)
    validation.to_csv(PREFIX.with_name(PREFIX.name+'_validation.csv'),index=False)
    selections={};models={};splits={}
    for end in (2024,2025):
        selected,ranked=core.select(configs,validation[validation.year.le(end)],end)
        config=next(c for c in configs if c['name']==selected['name'])
        held=data[data.ipo_id.isin(test_ids)&data.year.gt(end)].copy()
        earlier=data[data.year.le(end)&data.listing_date.lt(held.decision_market_date.min())].copy()
        core.membership_guard(earlier,held,end)
        gmp_config=next(c for c in configs if c['name']=='gmp_only_control')
        gmp_choice,_=core.select([gmp_config],validation[validation.year.le(end)],end)
        selection=dict(chosen=selected,config=config,ranked=ranked,gmp_control_cutoff=gmp_choice,
            training_ids=sorted(map(int,earlier.ipo_id)),test_ids=sorted(map(int,held.ipo_id)),test_used_for_selection=False,
            latest_training_outcome=str(earlier.listing_date.max().date()),first_test_decision=str(held.decision_market_date.min().date()))
        path=PREFIX.with_name(PREFIX.name+f'_through{end}_selection.json')
        path.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf8')
        model=core.estimator(config).fit(core.features(earlier,config),earlier.target_binary)
        gmp_model=core.estimator(gmp_config).fit(core.features(earlier,gmp_config),earlier.target_binary)
        artifact=ROOT/f'ml/artifacts/module1_forward_coverage_through{end}_research_20261024.joblib'
        joblib.dump(dict(model=model,gmp_control=gmp_model,config=config,features=core.SETS[config['feature_set']],
            threshold=selected['threshold'],training_ids=selection['training_ids'],training_end_year=end,research_only=True,
            selection_sha256=core.sha(path),input_sha256=hashes),artifact)
        selections[str(end)]=selection;models[end]=(model,gmp_model,gmp_config);splits[end]=(earlier,held)
    if any(core.sha(ROOT/p)!=h for p,h in hashes.items()):
        raise ValueError('Inputs changed during training')
    results={};ledgers={}
    original=pd.read_csv(ROOT/'output/module1_blended_fix_20261024_predictions.csv').set_index('ipo_id')
    for end in (2024,2025):
        earlier,held=splits[end];selection=selections[str(end)];config=selection['config'];model,gmp_model,gmp_config=models[end]
        held['p_coverage']=model.predict_proba(core.features(held,config))[:,1]
        held['p_gmp_control']=gmp_model.predict_proba(core.features(held,gmp_config))[:,1]
        held['direction_cutoff']=selection['chosen']['threshold'];held['trained_through']=end
        held['coverage_call']=held.p_coverage.ge(held.direction_cutoff).astype(int)
        held['coverage_correct']=held.coverage_call.eq(held.target_binary)
        held['previous_blended_correct']=held.ipo_id.map(original.blended_correct)
        ledgers[end]=held
        held.to_csv(PREFIX.with_name(PREFIX.name+f'_through{end}_predictions.csv'),index=False)
        results[str(end)]=dict(training_n=len(earlier),test_n=len(held),metrics=summarize(held.target_binary,held.p_coverage,selection['chosen']['threshold']),
            gmp_control=summarize(held.target_binary,held.p_gmp_control,selection['gmp_control_cutoff']['threshold']),
            by_year={str(int(y)):summarize(g.target_binary,g.p_coverage,selection['chosen']['threshold']) for y,g in held.groupby('year')},
            paired_change_vs_previous_blend=paired_gain(held.coverage_correct,held.previous_blended_correct))
    annual=pd.concat([ledgers[2024][ledgers[2024].year.eq(2025)],ledgers[2025]],ignore_index=True)
    if set(annual.ipo_id)!=test_ids or annual.ipo_id.duplicated().any():
        raise ValueError('Changed annual test population')
    annual.to_csv(PREFIX.with_name(PREFIX.name+'_annual_replay_predictions.csv'),index=False)
    missed=original[original.target_binary.eq(1)&original.blended_call.eq(0)].index
    review=annual[annual.ipo_id.isin(missed)].copy()
    review.to_csv(PREFIX.with_name(PREFIX.name+'_20_missed_positive_review.csv'),index=False)
    result=dict(selections=selections,validation_trace=trace,results=results,input_sha256=hashes,
        annual_replay=dict(n=len(annual),correct=int(annual.coverage_correct.sum()),accuracy_pct=float(annual.coverage_correct.mean()*100),
            tn=int((annual.coverage_call.eq(0)&annual.target_binary.eq(0)).sum()),fp=int((annual.coverage_call.eq(1)&annual.target_binary.eq(0)).sum()),
            fn=int((annual.coverage_call.eq(0)&annual.target_binary.eq(1)).sum()),tp=int((annual.coverage_call.eq(1)&annual.target_binary.eq(1)).sum())),
        previous_20_corrected=int(review.coverage_correct.sum()),coverage=dict(training_2021_2024=int(data.year.le(2024).sum()),
            training_2025=int(data.year.eq(2025).sum()),missing_gmp_earlier=int(data[data.year.le(2024)].gmp_pct.isna().sum()),
            missing_gmp_2025=int(data[data.year.eq(2025)].gmp_pct.isna().sum())),live_promoted=False,
        limitations=['The fixed 155-row test and all 69 2026 rows are preserved; no later-year winner was selected.',
            'No 2026 labels enter fitting or selection. Through-2025 training can be evaluated only on 2026.',
            'These reused historical outcomes are exploratory, not an untouched prospective test.',
            'Missing GMP is imputed only inside training, with a missing indicator; it is never recorded as observed zero.',
            'Financial period dates precede decisions; original fact publication times remain uncertified.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(selected={y:s['config']|{'cutoff':s['chosen']['threshold']} for y,s in selections.items()},
        results=results,annual=result['annual_replay'],coverage=result['coverage'],previous20corrected=result['previous_20_corrected']),indent=2),flush=True)


if __name__=='__main__':
    run()
