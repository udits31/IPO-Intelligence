"""Prespecified temporal extension, train/tune only through 2024."""
import gzip
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

import ml.research_module1_downside_fix as search
from ml.collect_module1_later_holdout import CACHE as LATER_CACHE
from ml.collect_module1_public_financials import CACHE as FIN_CACHE
from ml.collect_module1_later_holdout import bid_history
from ml.recover_module1_other_sources import ipoji_payload,OUT as OTHER_QUOTES
from ml.complete_module1_benchmark import prepare
from ml.evaluate_module1_later_holdout import validate
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain

ROOT=search.ROOT
PREFIX=ROOT/'output/module1_trajectory_fix_20261024'
EXTRA=['log_qib_growth','log_hni_growth','log_retail_growth','gmp_quote_age_days','gmp_slope_pp_per_day']
FEATURES=search.FEATURES+EXTRA


def add_temporal(rows,training,quotes):
    rows=rows.copy()
    for field in EXTRA:
        rows[field]=np.nan
    old_growth=pd.read_csv(ROOT/'data/audit/module1_bid_growth_20261024.csv').set_index('ipo_id')
    quote_history=quotes.copy();quote_history['gmp_date']=pd.to_datetime(quote_history.gmp_date)
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    for index,row in rows.iterrows():
        history=[]
        if training:
            if row.ipo_id in old_growth.index and old_growth.loc[row.ipo_id].status=='matched_penultimate_table':
                for field in EXTRA[:3]:
                    rows.loc[index,field]=old_growth.loc[row.ipo_id,field]
            path=FIN_CACHE/f'{row.ipo_id}.json.gz'
            if path.exists():
                with gzip.open(path,'rt',encoding='utf8') as stream:
                    cached=json.load(stream)
                if cached.get('raw_response_text'):
                    item=dict(row,issue_price=float(canonical.loc[row.ipo_id].price_band_upper))
                    payload=ipoji_payload(item,BeautifulSoup(cached['raw_response_text'],'html.parser'))
                    if payload:
                        history=bid_history(payload['bids'],row.issue_open_date,row.issue_close_date)
        else:
            with gzip.open(LATER_CACHE/f'{row.ipo_id}.json.gz','rt',encoding='utf8') as stream:
                history=json.load(stream).get('bid_history',[])
        past=[v for v in history if pd.Timestamp(v['date'])<=row.decision_market_date]
        if len(past)>=2 and pd.Timestamp(past[-1]['date'])==row.decision_market_date and all(
                abs(past[-1][c+'_subscription']-row[c+'_subscription'])<=.02 for c in ('qib','hni','retail')):
            for category in ('qib','hni','retail'):
                field='log_'+category+'_growth'
                if pd.isna(rows.loc[index,field]):
                    rows.loc[index,field]=np.log1p(past[-1][category+'_subscription'])-np.log1p(past[0][category+'_subscription'])
        selected=pd.Timestamp(row.gmp_date)
        rows.loc[index,'gmp_quote_age_days']=(row.decision_market_date-selected).days
        prior=quote_history[quote_history.ipo_id.eq(row.ipo_id)&quote_history.source_url.eq(row.gmp_source_url)&
            quote_history.gmp_date.lt(selected)&quote_history.gmp_date.ge(selected-pd.Timedelta(days=7))].copy()
        conflicts=prior.groupby('gmp_date').gmp_rs.nunique()
        prior=prior[~prior.gmp_date.isin(conflicts[conflicts.gt(1)].index)]
        if len(prior):
            older=prior.sort_values('gmp_date').iloc[-1]
            earlier_pct=100*older.gmp_rs/float(canonical.loc[row.ipo_id].price_band_upper)
            rows.loc[index,'gmp_slope_pp_per_day']=(row.gmp_pct-earlier_pct)/(selected-older.gmp_date).days
    return rows


def run():
    train=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv')
    test=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv')
    _,_,old_quotes=prepare();quotes=pd.concat([old_quotes,pd.read_csv(OTHER_QUOTES)],ignore_index=True)
    new_quotes=[]
    for row in test.itertuples():
        with gzip.open(LATER_CACHE/f'{row.ipo_id}.json.gz','rt',encoding='utf8') as stream:
            record=json.load(stream)
        for source in record['sources']:
            for quote in source.get('quotes',[]):
                new_quotes.append(dict(quote,ipo_id=row.ipo_id,source_url=source['url']))
    quotes=pd.concat([quotes,pd.DataFrame(new_quotes)],ignore_index=True)
    for rows in (train,test):
        for field in ('listing_date','decision_market_date','issue_close_date','issue_open_date'):
            rows[field]=pd.to_datetime(rows[field])
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    validate(train,test,canonical)
    train=add_temporal(train,True,quotes);test=add_temporal(test,False,quotes)
    training_path=PREFIX.with_name(PREFIX.name+'_training.csv')
    train.to_csv(training_path,index=False)
    quotes_path=PREFIX.with_name(PREFIX.name+'_quote_inputs.csv')
    quotes.to_csv(quotes_path,index=False)
    search.FEATURES=FEATURES
    configs=search.candidates();folds=[];trace=[]
    for year in (2022,2023,2024):
        held=train[train.year.eq(year)].copy();first=held.decision_market_date.min()
        earlier=train[train.year.lt(year)&train.listing_date.lt(first)]
        if set(earlier.ipo_id)&set(held.ipo_id) or earlier.listing_date.max()>=first:
            raise ValueError('Validation leakage')
        trace.append(dict(year=year,train_ids=sorted(map(int,earlier.ipo_id)),validation_ids=sorted(map(int,held.ipo_id)),
            latest_training_outcome=str(earlier.listing_date.max().date()),first_validation_decision=str(first.date())))
        for config in configs:
            model=search.estimator(config).fit(search.feature_frame(earlier,config),earlier.target_binary)
            held[config['name']]=model.predict_proba(search.feature_frame(held,config))[:,1]
        folds.append(held)
        print(f'temporal validation {year} complete',flush=True)
    validation=pd.concat(folds,ignore_index=True)
    _,ranked=search.choose(configs,validation)
    eligible=[]
    for setting in ranked:
        p=validation[setting['name']].ge(setting['threshold'])
        recall=float((~p[validation.target_binary.eq(0)]).mean())
        if recall>=.5:
            eligible.append(dict(setting,negative_recall=recall))
    if not eligible:
        raise ValueError('No earlier-year setting met downside-recall requirement')
    selected=eligible[0];config=next(c for c in configs if c['name']==selected['name'])
    selection_path=PREFIX.with_name(PREFIX.name+'_selection.json')
    selection_path.write_text(json.dumps(dict(selected=selected,config=config,eligible_ranked=eligible,trace=trace,
        test_used_for_selection=False),indent=2)+'\n',encoding='utf8')
    validation.to_csv(PREFIX.with_name(PREFIX.name+'_validation.csv'),index=False)
    model=search.estimator(config).fit(search.feature_frame(train,config),train.target_binary)
    artifact=ROOT/'ml/artifacts/module1_trajectory_fix_research_20261024.joblib'
    joblib.dump(dict(model=model,config=config,features=FEATURES,threshold=selected['threshold'],
        training_ids=sorted(map(int,train.ipo_id)),research_only=True),artifact)
    test['p_trajectory_fix']=model.predict_proba(search.feature_frame(test,config))[:,1]
    test['trajectory_call']=test.p_trajectory_fix.ge(selected['threshold']).astype(int)
    test.to_csv(PREFIX.with_name(PREFIX.name+'_predictions.csv'),index=False)
    result=summarize(test.target_binary,test.p_trajectory_fix,selected['threshold'])
    gain=paired_gain(test.trajectory_call.eq(test.target_binary),test.p_combined.ge(.5).eq(test.target_binary))
    fp=test.target_binary.eq(0)&test.p_combined.ge(.5)
    coverage={name:{key:int(rows[key].notna().sum()) for key in EXTRA} for name,rows in [('train',train),('test',test)]}
    paths=[ROOT/'ml/research_module1_trajectory_fix.py',ROOT/'ml/research_module1_downside_fix.py',
        ROOT/'output/module1_trajectory_fix_protocol_20261024.md',selection_path,artifact,
        ROOT/'output/module1_2025_2026_paired_holdout_20261024_training.csv',
        ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv',
        ROOT/'data/audit/module1_bid_growth_20261024.csv',OTHER_QUOTES,training_path,quotes_path]
    report=dict(config=config,selected=selected,results=result,paired_change=gain,coverage=coverage,
        training_n=len(train),test_n=len(test),training_ids=sorted(map(int,train.ipo_id)),test_ids=sorted(map(int,test.ipo_id)),
        corrected_original_36=int((fp&test.trajectory_call.eq(0)).sum()),
        newly_missed_positive=int((test.target_binary.eq(1)&test.p_combined.ge(.5)&test.trajectory_call.eq(0)).sum()),
        by_year={str(int(y)):summarize(g.target_binary,g.p_trajectory_fix,selected['threshold']) for y,g in test.groupby('year')},
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        artifact=str(artifact.relative_to(ROOT)),live_promoted=False,
        limitations=['Reused later-year errors make this exploratory retrospective evidence.',
            'No test labels used in training, selection or threshold choice.',
            'GMP and bid trajectories have incomplete, unequal coverage; missing is not zero.',
            'Historical public tables do not certify original publication time.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    run()
