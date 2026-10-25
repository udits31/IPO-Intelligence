"""Strict 2024 fit versus forward update through 2025, on fixed issuer IDs."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import ml.research_module1_downside_fix as prior
from ml.model_pipeline import summarize
from ml.research_monitor_daily_learning import paired_gain
from services.prediction_results_service import identity_name

ROOT=prior.ROOT
PREFIX=ROOT/'output/module1_forward_update_20261024'
BASE=['log_qib','log_retail','log_hni','log_offer_cap_price','log_issue_size','gmp_pct']
TRAJECTORY=['log_qib_growth','log_hni_growth','log_retail_growth','gmp_quote_age_days','gmp_slope_pp_per_day']
MACRO=['macro_nifty_5d','macro_nifty_20d','macro_nifty_vol20','macro_vix','macro_vix_change5']
FINANCIAL=['pub_profit_margin_pct','pub_return_on_assets_pct','pub_revenue_growth_asinh',
           'pub_annual_loss','pub_financial_sector','pub_ipo_earnings_yield_pct','pub_log_ipo_price_book']
SETS={'base':BASE,'trajectory':BASE+TRAJECTORY,'base_macro':BASE+MACRO,
      'trajectory_macro':BASE+TRAJECTORY+MACRO,'financial_macro':BASE+FINANCIAL+MACRO,
      'full':BASE+TRAJECTORY+FINANCIAL+MACRO}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def market_features(rows):
    old=pd.read_csv(ROOT/'data/raw_daily_market_time_series.csv')
    old=old[old.observation_type.eq('CLOSE')].pivot_table(index='observed_at',columns='instrument',values='value',aggfunc='first')
    old.index=pd.to_datetime(old.index)
    new=pd.read_csv(ROOT/'data/audit/module1_macro_download_20261024.csv',header=[0,1],index_col=0)
    new.index=pd.to_datetime(new.index,errors='coerce')
    new=new[new.index.notna()]
    additions=pd.DataFrame({'NIFTY50':new[('Close','^NSEI')],'INDIAVIX':new[('Close','^INDIAVIX')]})
    series=pd.concat([old[['NIFTY50','INDIAVIX']],additions[additions.index>=pd.Timestamp('2025-01-01')]]).sort_index()
    if series.index.has_duplicates:
        raise ValueError('Duplicate market dates')
    series.to_csv(PREFIX.with_name(PREFIX.name+'_market.csv'),index_label='date')
    nifty=series.NIFTY50.dropna();vix=series.INDIAVIX.dropna()
    nr=nifty.pct_change(fill_method=None)*100
    context=pd.DataFrame({'macro_nifty_5d':nifty.pct_change(5,fill_method=None)*100,
        'macro_nifty_20d':nifty.pct_change(20,fill_method=None)*100,
        'macro_nifty_vol20':nr.rolling(20,min_periods=20).std(),
        'macro_vix':vix,'macro_vix_change5':vix.pct_change(5,fill_method=None)*100})
    result=rows.copy()
    for field in MACRO:
        result[field]=np.nan
    result['macro_source_date']=None
    for index,row in result.iterrows():
        available=context[context.index<row.decision_market_date]
        if available.empty:
            continue
        date=available.index[-1]
        if (row.decision_market_date-date).days>5:
            continue
        result.loc[index,MACRO]=available.iloc[-1][MACRO].to_numpy()
        result.loc[index,'macro_source_date']=str(date.date())
    return result


def candidates():
    core=prior.candidates()
    for depth in (1,2):
        core.append(dict(name=f'xgb_depth{depth}',family='xgboost',transform='raw',max_depth=depth))
    return [dict(c,feature_set=s,name=s+'__'+c['name']) for s in SETS for c in core]


def features(rows,config):
    x=rows[SETS[config['feature_set']]].copy()
    if config['transform']=='signed_log':
        x['gmp_pct']=np.sign(x.gmp_pct)*np.log1p(np.abs(x.gmp_pct))
    return x


def estimator(config):
    if config['family']!='xgboost':
        return prior.estimator(config)
    return Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('scale',StandardScaler()),('model',XGBClassifier(n_estimators=120,max_depth=config['max_depth'],
            learning_rate=.04,min_child_weight=5,reg_lambda=10,subsample=.8,colsample_bytree=.8,
            random_state=42,n_jobs=2,eval_metric='logloss'))])


def membership_guard(train,test,cutoff_year):
    if train.empty or test.empty or not train.year.between(2021,cutoff_year).all() or not test.year.gt(cutoff_year).all():
        raise ValueError('Wrong forward split')
    if set(train.ipo_id)&set(test.ipo_id) or train.ipo_id.duplicated().any() or test.ipo_id.duplicated().any():
        raise ValueError('Issuer leakage')
    if train.listing_date.max()>=test.decision_market_date.min():
        raise ValueError('Training outcome after first test decision')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    for field in ('company_name','nse_symbol'):
        transform=identity_name if field=='company_name' else lambda x:str(x).strip().upper()
        a=set(canonical.loc[train.ipo_id,field].dropna().map(transform))-set([''])
        b=set(canonical.loc[test.ipo_id,field].dropna().map(transform))-set([''])
        if a&b:
            raise ValueError('Issuer name/symbol overlap')


def select(configs,validation,end_year):
    if not validation.year.between(2022,end_year).all():
        raise ValueError('Later labels cannot select forward model')
    recent=validation[validation.year.isin([end_year-1,end_year])]
    if set(recent.year)!={end_year-1,end_year}:
        raise ValueError('Missing recent chronological validation year')
    y=recent.target_binary.to_numpy();ranked=[]
    for config in configs:
        p=recent[config['name']].to_numpy()
        for threshold in prior.THRESHOLDS:
            call=p>=threshold;negative_recall=float((~call[y==0]).mean());positive_recall=float(call[y==1].mean())
            ranked.append(dict(name=config['name'],threshold=float(threshold),n=len(y),
                validation_years=[end_year-1,end_year],accuracy=float((call==y).mean()),
                balanced_accuracy=(negative_recall+positive_recall)/2,brier=float(np.mean((p-y)**2)),
                feature_count=len(SETS[config['feature_set']]),
                simplicity=0 if config['family']=='logistic' else 1 if config['family']=='gradient_boosting' else 2))
    ranked.sort(key=lambda r:(-r['accuracy'],-r['balanced_accuracy'],r['brier'],r['feature_count'],
        r['simplicity'],abs(r['threshold']-.5),r['name']))
    return ranked[0],ranked


def run():
    train_path=ROOT/'output/module1_trajectory_fix_20261024_training.csv'
    test_path=ROOT/'output/module1_trajectory_fix_20261024_predictions.csv'
    train=pd.read_csv(train_path); test=pd.read_csv(test_path)
    frame=pd.concat([train,test],ignore_index=True)
    public=pd.read_csv(ROOT/'data/audit/module1_public_financials_20261024.csv')
    public=public.rename(columns={c:'pub_'+c for c in public.columns if c!='ipo_id'})
    frame=frame.merge(public[['ipo_id','pub_annual_period',*FINANCIAL]],on='ipo_id',validate='one_to_one')
    for field in ('listing_date','decision_market_date','issue_close_date','issue_open_date'):
        frame[field]=pd.to_datetime(frame[field])
    if not (pd.to_datetime(frame.gmp_date)<frame.decision_market_date).all():
        raise ValueError('Future/same-day GMP')
    observed=frame.pub_annual_period.notna()
    if not (pd.to_datetime(frame.loc[observed,'pub_annual_period'])<frame.loc[observed,'decision_market_date']).all():
        raise ValueError('Future financial period')
    frame=market_features(frame)
    if not (pd.to_datetime(frame.macro_source_date)<frame.decision_market_date).all():
        raise ValueError('Future market observations')
    frame_path=PREFIX.with_name(PREFIX.name+'_features.csv');frame.to_csv(frame_path,index=False)
    frozen_inputs=[train_path,test_path,ROOT/'data/audit/module1_public_financials_20261024.csv',
        ROOT/'data/raw_daily_market_time_series.csv',ROOT/'data/audit/module1_macro_download_20261024.csv',
        ROOT/'ml/research_module1_forward_update.py',ROOT/'output/module1_forward_update_protocol_20261024.md']
    input_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in frozen_inputs}
    configs=candidates();folds=[];trace=[]
    # All development predictions stop in 2025. Nothing from 2026 fits a model.
    for year in (2022,2023,2024,2025):
        held=frame[frame.year.eq(year)].copy();first=held.decision_market_date.min()
        earlier=frame[frame.year.lt(year)&frame.listing_date.lt(first)]
        membership_guard(earlier,held,year-1)
        trace.append(dict(year=year,train_ids=sorted(map(int,earlier.ipo_id)),validation_ids=sorted(map(int,held.ipo_id)),
            latest_training_outcome=str(earlier.listing_date.max().date()),first_validation_decision=str(first.date())))
        probabilities={}
        for config in configs:
            model=estimator(config).fit(features(earlier,config),earlier.target_binary)
            probabilities[config['name']]=model.predict_proba(features(held,config))[:,1]
        held=pd.concat([held,pd.DataFrame(probabilities,index=held.index)],axis=1)
        folds.append(held)
        print(f'forward validation {year}: train={len(earlier)} validation={len(held)} configs={len(configs)}',flush=True)
    validation=pd.concat(folds,ignore_index=True)
    validation.to_csv(PREFIX.with_name(PREFIX.name+'_validation.csv'),index=False)
    selections={};models={};splits={}
    for end_year in (2024,2025):
        allowed_validation=validation[validation.year.le(end_year)]
        chosen,ranked=select(configs,allowed_validation,end_year)
        config=next(c for c in configs if c['name']==chosen['name'])
        held=frame[frame.year.gt(end_year)].copy();first=held.decision_market_date.min()
        fit=frame[frame.year.le(end_year)&frame.listing_date.lt(first)].copy()
        membership_guard(fit,held,end_year)
        selection=dict(chosen=chosen,config=config,ranked=ranked,test_used_for_selection=False,
            training_ids=sorted(map(int,fit.ipo_id)),test_ids=sorted(map(int,held.ipo_id)),
            latest_training_outcome=str(fit.listing_date.max().date()),first_test_decision=str(first.date()))
        selections[str(end_year)]=selection
        selection_path=PREFIX.with_name(PREFIX.name+f'_through{end_year}_selection.json')
        selection_path.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf8')
        model=estimator(config).fit(features(fit,config),fit.target_binary)
        models[end_year]=model;splits[end_year]=(fit,held)
        artifact=ROOT/f'ml/artifacts/module1_forward_through{end_year}_research_20261024.joblib'
        joblib.dump(dict(model=model,config=config,features=SETS[config['feature_set']],threshold=chosen['threshold'],
            training_ids=selection['training_ids'],training_end_year=end_year,research_only=True,
            selection_sha256=sha(selection_path),input_sha256=input_hashes),artifact)
    # Both selections and model fits are frozen before any test score is computed.
    if any(sha(ROOT/p)!=h for p,h in input_hashes.items()):
        raise ValueError('Inputs changed during fitting')
    results={};ledgers={}
    for end_year in (2024,2025):
        fit,held=splits[end_year];selection=selections[str(end_year)];config=selection['config']
        held['p_forward']=models[end_year].predict_proba(features(held,config))[:,1]
        threshold=selection['chosen']['threshold']
        held['direction_cutoff']=threshold
        held['trained_through']=end_year
        held['forward_call']=held.p_forward.ge(threshold).astype(int)
        held['forward_correct']=held.forward_call.eq(held.target_binary)
        ledgers[end_year]=held
        held.to_csv(PREFIX.with_name(PREFIX.name+f'_through{end_year}_predictions.csv'),index=False)
        baselines={name:summarize(held.target_binary,held[name],.5) for name in ('p_combined','p_gmp_sign','p_gmp_fitted')}
        results[str(end_year)]=dict(training_n=len(fit),test_n=len(held),metrics=summarize(held.target_binary,held.p_forward,threshold),
            baselines=baselines,by_year={str(int(y)):summarize(g.target_binary,g.p_forward,threshold) for y,g in held.groupby('year')},
            vs_original_combined=paired_gain(held.forward_correct,held.p_combined.ge(.5).eq(held.target_binary)))
    rolling=pd.concat([ledgers[2024][ledgers[2024].year.eq(2025)],ledgers[2025]],ignore_index=True)
    if set(rolling.ipo_id)!=set(test.ipo_id) or rolling.ipo_id.duplicated().any():
        raise ValueError('Annual replay changed the fixed 155 issuers')
    rolling.to_csv(PREFIX.with_name(PREFIX.name+'_annual_replay_predictions.csv'),index=False)
    # summarize threshold=.5 on the actual frozen binary calls, keeping mixed cutoffs explicit.
    rolling_accuracy=float(rolling.forward_correct.mean()*100)
    original_blend=pd.read_csv(ROOT/'output/module1_blended_fix_20261024_predictions.csv').set_index('ipo_id')
    previous_fn=original_blend[original_blend.target_binary.eq(1)&original_blend.blended_call.eq(0)].index
    missed=rolling[rolling.ipo_id.isin(previous_fn)].copy()
    missed['previous_blended_score']=missed.ipo_id.map(original_blend.p_blended_fix)
    missed.to_csv(PREFIX.with_name(PREFIX.name+'_20_missed_positive_review.csv'),index=False)
    annual=dict(n=len(rolling),correct=int(rolling.forward_correct.sum()),accuracy_pct=rolling_accuracy,
        tn=int((rolling.target_binary.eq(0)&rolling.forward_call.eq(0)).sum()),
        fp=int((rolling.target_binary.eq(0)&rolling.forward_call.eq(1)).sum()),
        fn=int((rolling.target_binary.eq(1)&rolling.forward_call.eq(0)).sum()),
        tp=int((rolling.target_binary.eq(1)&rolling.forward_call.eq(1)).sum()),
        interpretation='2025 uses model trained through 2024; 2026 uses model trained through 2025, with year-specific earlier-only selection.')
    result=dict(selections=selections,validation_trace=trace,results=results,annual_replay=annual,
        previous_20_corrected_in_annual_replay=int(missed.forward_correct.sum()),
        market_coverage={field:int(frame[field].notna().sum()) for field in MACRO},
        feature_input_sha256=sha(frame_path),input_sha256=input_hashes,live_promoted=False,
        limitations=['Reused test outcomes make this exploratory retrospective development, not independent proof.',
            'Training/selection through 2025 is legitimate only for the 2026 test; 2025 rows cannot be counted as its holdout.',
            'Market closes strictly precede each decision; financial period gates do not certify original publication dates.',
            'Same issuer IDs, original interim bids and original opening labels are retained. No test winner is substituted.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    render(result)
    print(json.dumps(dict(selected={y:s['config']|{'cutoff':s['chosen']['threshold']} for y,s in selections.items()},
        results=results,annual_replay=annual,previous_20_corrected=result['previous_20_corrected_in_annual_replay']),indent=2),flush=True)


def render(result):
    lines=['# Module 1: overall accuracy and forward-year training update','',
        '| Experiment | Training IPOs | Test IPOs | Correct | Accuracy | Loss/flat caught | Gains caught |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for year,value in result['results'].items():
        m=value['metrics']
        lines.append(f'| Fit through {year} | {value["training_n"]} | {value["test_n"]} | {m["tn"]+m["tp"]} | {m["classifier_accuracy_pct"]:.1f}% | {m["tn"]}/{m["nonpositive_listings"]} | {m["tp"]}/{m["positive_listings"]} |')
    annual=result['annual_replay']
    lines.append(f'| Annual chronological replay | 170, then 256 | 155 | {annual["correct"]} | {annual["accuracy_pct"]:.1f}% | {annual["tn"]}/49 | {annual["tp"]}/106 |')
    lines += ['', '**The through-2025 model is tested only on 2026. Its training IPOs are never counted as test predictions.**',
        '', '## What changed','',
        '- The objective is overall accuracy. Each missed positive and missed negative counts once; there is no forced negative-recall requirement.',
        '- Selection uses the latest two earlier validation years. The strict model uses 2023/2024; the updated model uses 2024/2025.',
        '- Historical market conditions are joined strictly before the decision date, alongside prior bid/GMP trajectories and available IPO-context financial facts.',
        f'- Corrected {result["previous_20_corrected_in_annual_replay"]} of the previous 20 missed positive openings in the annual replay.',
        '', '## Selected settings','']
    for year,selection in result['selections'].items():
        lines.append(f'- Through {year}: {selection["config"]}; cutoff {selection["chosen"]["threshold"]}; earlier validation accuracy {selection["chosen"]["accuracy"]*100:.1f}%.')
    lines += ['', '## Full evidence','',
        '- Original/current scores, opening prices, dates, cutoffs and correctness flags are stored in the per-IPO CSVs.',
        '- The complete candidate ranking and chronological training memberships were saved before computing later-year scores.',
        '', '## Limits','',*[f'- {v}' for v in result['limitations']]]
    PREFIX.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    run()
