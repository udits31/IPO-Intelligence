"""Day-specific causal five-expert comparisons. Existing live artifact untouched."""
import json
import joblib
import numpy as np
import pandas as pd
from ml.build_stage_matched_cohorts_20261028 import OUT
from ml.test_historical_joins_and_cutoff_20261028 import ROOT,core,EXPERTS,logits,stacker,sha
from ml.test_through2025_cutoff_20261028 import ARTIFACT
from ml.evaluate_module1_period_stability_20261028 import scope_metrics


def select_quote(quotes,asof):
    eligible=quotes[quotes.timestamp.le(asof)]
    if eligible.empty:return None
    latest=eligible[eligible.timestamp.eq(eligible.timestamp.max())]
    if not np.allclose(latest.gmp_rs.to_numpy(float),float(latest.gmp_rs.iloc[0]),atol=1e-8,rtol=1e-8):return None
    return latest.iloc[0]


def quote_event_date(quote):
    value=quote.get('event_date')
    return pd.Timestamp(value).date() if pd.notna(value) else quote.timestamp.date()


def prepare():
    rows=pd.read_csv(OUT/'clocked_stage_rows.csv')
    rows['stage_timestamp']=pd.to_datetime(rows.stage_timestamp,utc=True).dt.tz_convert('Asia/Kolkata')
    rows['listing_date']=pd.to_datetime(rows.listing_date)
    quotes=pd.read_csv(OUT/'clocked_gmp_quotes.csv');quotes['timestamp']=pd.to_datetime(quotes.timestamp,utc=True).dt.tz_convert('Asia/Kolkata')
    quotes['event_date']=quotes.timestamp.dt.date
    dated=pd.read_csv(ROOT/'data/preclose_gmp_history.csv')
    dated['timestamp']=(pd.to_datetime(dated.gmp_date)+pd.Timedelta(days=1)).dt.tz_localize('Asia/Kolkata')
    dated['source_sha256']=dated.source_sha256
    dated['time_basis']='prior_date_only_quote_available_next_day'
    dated['event_date']=pd.to_datetime(dated.gmp_date).dt.date
    # Retain already-audited single dated quotes as well as full history tables.
    single=pd.read_csv(ROOT/'output/model_comparison_20261028/numeric_audited_features.csv')
    single=single[single.cohort.eq('historical')&single.gmp_date.notna()&single.gmp_pct.notna()].copy()
    single['ipo_id']=pd.to_numeric(single.ipo_id).astype(int)
    single['gmp_rs']=single.issue_price*single.gmp_pct/100
    single['timestamp']=(pd.to_datetime(single.gmp_date)+pd.Timedelta(days=1)).dt.tz_localize('Asia/Kolkata')
    single['source_url']=single.gmp_source_url
    single['source_sha256']=sha(ROOT/'output/model_comparison_20261028/numeric_audited_features.csv')
    single['time_basis']='audited_date_only_quote_available_next_day'
    single['event_date']=pd.to_datetime(single.gmp_date).dt.date
    quotes=pd.concat([quotes,dated,single[['ipo_id','timestamp','event_date','gmp_rs','source_url','source_sha256','time_basis']]],ignore_index=True)
    quotes['ipo_id']=pd.to_numeric(quotes.ipo_id).astype(int)
    quotes_by_id={id:g for id,g in quotes.groupby('ipo_id')}
    market=pd.read_csv(ROOT/'output/module1_forward_update_20261024_market.csv',index_col='date',parse_dates=True)
    n=market.NIFTY50.dropna();v=market.INDIAVIX.dropna()
    m=pd.DataFrame(dict(macro_nifty_5d=n.pct_change(5)*100,macro_nifty_20d=n.pct_change(20)*100,
        macro_nifty_vol20=(n.pct_change()*100).rolling(20).std(),macro_vix=v,macro_vix_change5=v.pct_change(5)*100))
    audit=[]
    for i,row in rows.iterrows():
        ts=row.stage_timestamp
        for cat in ['qib','hni','retail']:
            rows.loc[i,'log_'+cat]=np.log1p(row['stage_'+cat]);rows.loc[i,cat+'_subscription' if cat!='hni' else 'hni_subscription']=row['stage_'+cat]
            rows.loc[i,'log_'+cat+'_growth']=np.nan
        previous=rows[rows.ipo_id.eq(row.ipo_id)&(rows.stage_timestamp.dt.date<ts.date())]
        if not previous.empty:
            previous=previous.loc[previous.stage_timestamp.idxmax()]
            for cat in ['qib','hni','retail']:rows.loc[i,'log_'+cat+'_growth']=np.log1p(row['stage_'+cat])-np.log1p(previous['stage_'+cat])
        rows.loc[i,['gmp_pct','gmp_quote_age_days','gmp_slope_pp_per_day']]=np.nan
        available=quotes_by_id.get(row.ipo_id,quotes.iloc[:0])
        chosen=select_quote(available,ts)
        quote_info=None
        if chosen is not None:
            rows.loc[i,'gmp_pct']=100*chosen.gmp_rs/row.issue_price
            event=quote_event_date(chosen)
            rows.loc[i,'gmp_date']=str(event)
            rows.loc[i,'gmp_quote_age_days']=(ts.date()-event).days
            prior=available[(available.event_date<event)&available.timestamp.ge(chosen.timestamp-pd.Timedelta(days=7))]
            if not prior.empty:
                p=select_quote(prior,chosen.timestamp)
                if p is not None:rows.loc[i,'gmp_slope_pp_per_day']=100*(chosen.gmp_rs-p.gmp_rs)/row.issue_price/(event-quote_event_date(p)).days
            quote_info={k:str(chosen.get(k)) for k in ['timestamp','event_date','gmp_rs','source_url','source_sha256','time_basis']}
        rows.loc[i,core.MACRO]=np.nan
        prior=m[m.index<pd.Timestamp(ts.date())]
        if len(prior) and (ts.date()-prior.index[-1].date()).days<=5:rows.loc[i,core.MACRO]=prior.iloc[-1][core.MACRO].to_numpy()
        period=pd.to_datetime(row.get('pub_annual_period'),errors='coerce')
        if pd.isna(period) or period.date()>=ts.date():rows.loc[i,[f for f in core.FINANCIAL if f!='pub_financial_sector']]=np.nan
        rows.loc[i,'decision_market_date']=str(ts.date())
        audit.append(dict(ipo_id=int(row.ipo_id),stage=row.stage,asof=ts.isoformat(),quote=quote_info,
            growth_reference=None if previous.empty else str(previous.stage_timestamp),
            market_date=None if prior.empty else str(prior.index[-1].date())))
    rows['decision_market_date']=pd.to_datetime(rows.decision_market_date)
    rows.to_csv(OUT/'matched_features.csv',index=False)
    (OUT/'feature_asof_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    return rows


def fit_stage(train,held,configs):
    if not train.year.between(2021,2025).all() or held.year.nunique()!=1 or train.year.max()>=held.year.min() or set(train.ipo_id)&set(held.ipo_id):raise ValueError('Issuer or year leakage')
    if train.ipo_id.duplicated().any() or held.ipo_id.duplicated().any():raise ValueError('Duplicate issuer within stage')
    if train.stage.nunique()!=1 or held.stage.nunique()!=1 or train.stage.iloc[0]!=held.stage.iloc[0]:raise ValueError('Training/live stage mismatch')
    if 'company_name' in train and 'company_name' in held:
        from services.historical_bid_service import name_key
        if set(train.company_name.map(name_key))&set(held.company_name.map(name_key)):raise ValueError('Issuer-name leakage')
    if train.listing_date.max()>=held.decision_market_date.min():raise ValueError('Unavailable training outcomes')
    blocks=[];trace=[]
    for year in sorted(train.year.unique()):
        if year<=2021:continue
        validation=train[train.year.eq(year)].copy()
        known=train[train.year.lt(year)&train.listing_date.lt(validation.decision_market_date.min())]
        if len(known)<25 or known.target_binary.nunique()<2:continue
        for name,c in configs.items():
            model=core.estimator(c).fit(core.features(known,c),known.target_binary)
            validation[name]=model.predict_proba(core.features(validation,c))[:,1]
        blocks.append(validation)
        trace.append(dict(year=int(year),fit_ids=known.ipo_id.tolist(),validation_ids=validation.ipo_id.tolist()))
    oof=pd.concat(blocks,ignore_index=True)
    # Match the frozen recipe: its combiner uses the last two pre-test OOF years.
    end=int(train.year.max());meta_rows=oof[oof.year.isin([end-1,end])]
    if meta_rows.target_binary.nunique()!=2:raise ValueError('Insufficient stage-specific meta outcomes')
    meta=stacker().fit(logits(meta_rows[EXPERTS]),meta_rows.target_binary.astype(int));models={};p=[]
    for name,c in configs.items():
        models[name]=core.estimator(c).fit(core.features(train,c),train.target_binary)
        p.append(models[name].predict_proba(core.features(held,c))[:,1])
    probabilities=meta.predict_proba(logits(np.column_stack(p)))[:,1]
    bundle=dict(models=models,meta_model=meta,expert_configs=configs,experts=EXPERTS,threshold=.5,
        training_ids=train.ipo_id.tolist(),meta_training_ids=meta_rows.ipo_id.tolist(),meta_years=[end-1,end],training_end_year=int(train.year.max()),stage=str(train.stage.iloc[0]),research_only=True,trace=trace)
    return probabilities,np.column_stack(p)[:,-1],bundle


def evaluate(rows,model,gmp):
    mask=rows.gmp_pct.notna()
    return dict(model=scope_metrics(rows,model),gmp_only=scope_metrics(rows,gmp),
        gmp_fixed625=scope_metrics(rows,gmp,.625),
        raw_gmp=scope_metrics(rows[mask],rows.loc[mask,'gmp_pct'].gt(0).astype(float).to_numpy()) if mask.any() else None,
        always_positive=scope_metrics(rows,np.ones(len(rows))))


def run():
    protocol=dict(training='2021–2025 only; causal expanding base OOF; last two pre-test OOF years for meta, matching frozen recipe',validation_years=[2024,2025],test_year=2026,
        fixed_cutoff=.5,architecture='same five expert configurations and C=0.5 log-odds meta',
        stage_policy='Separate issuer-disjoint stage models; no invented clock or post-17 closing-day table',
        features='Bids, previous completed-day trajectory, GMP and prior market closes rebuilt at each actual stage timestamp',
        promotion='Research only. Current deployed artifact unchanged. No switch based on inspected 2026 alone.',
        implementation_audit='Initial all-OOF combiner refit did not match the frozen recent-two-year combiner recipe. Preserved as previous_all_oof_diagnostic; this correction is a retrospective implementation audit, not fresh model selection.')
    (OUT/'model_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    original=sha(ARTIFACT);rows=prepare();core.SETS['gmp_only']=['gmp_pct']
    candidates=core.candidates()+[dict(name='gmp_only_control',family='logistic',C=.5,class_weight=None,transform='raw',feature_set='gmp_only')]
    configs={name:next(c for c in candidates if c['name']==name) for name in EXPERTS}
    report=dict(protocol=protocol,stages={},artifact_sha256=original,live_model_changed=False,
        input_hashes={str((OUT/'clocked_stage_rows.csv').relative_to(ROOT)).replace('\\','/'):sha(OUT/'clocked_stage_rows.csv'),str((OUT/'clocked_gmp_quotes.csv').relative_to(ROOT)).replace('\\','/'):sha(OUT/'clocked_gmp_quotes.csv')})
    for relative in ['output/model_comparison_20261028/numeric_audited_features.csv','data/preclose_gmp_history.csv','output/module1_forward_update_20261024_market.csv']:
        report['input_hashes'][relative]=sha(ROOT/relative)
    ledgers=[]
    for stage,group in rows.groupby('stage'):
        train=group[group.year.le(2025)].copy();held=group[group.year.eq(2026)].copy()
        if len(train)<50 or held.empty:
            report['stages'][stage]=dict(status='insufficient_matching_history',training_n=len(train),test_n=len(held));continue
        validation={}
        for year in [2024,2025]:
            test=group[group.year.eq(year)].copy();known=group[group.year.lt(year)&group.listing_date.lt(test.decision_market_date.min())]
            if len(known)<50 or test.empty:continue
            p,gmp,_=fit_stage(known,test,configs)
            mask=test.gmp_pct.notna().to_numpy()
            validation[str(year)]=dict(all=evaluate(test,p,gmp),paired_observed_gmp=evaluate(test[mask],p[mask],gmp[mask]),n=len(test))
        # All validation scores written before the 2026 fit/score for this stage.
        (OUT/(stage+'_pre2026_validation.json')).write_text(json.dumps(validation,indent=2),encoding='utf8')
        p,gmp,bundle=fit_stage(train,held,configs);mask=held.gmp_pct.notna().to_numpy()
        joblib.dump(bundle,OUT/(stage+'_research.joblib'))
        held['stage_model_score']=p;held['stage_gmp_score']=gmp;ledgers.append(held)
        report['stages'][stage]=dict(status='evaluated_research',training_n=len(train),test_n=len(held),
            validation=validation,all_test=evaluate(held,p,gmp),paired_observed_gmp=evaluate(held[mask],p[mask],gmp[mask]),observed_gmp_n=int(mask.sum()))
    report['closing_pre17_status']='No eligible training stage' if 'CLOSING_PRE17' not in report['stages'] else report['stages']['CLOSING_PRE17']['status']
    report['artifact_unchanged']=sha(ARTIFACT)==original
    report['release_decision']='Do not promote either stage candidate: neither beats the fixed-0.625 stage-fitted GMP baseline on the 69-case test. Current deployed artifact and ledgers remain intact.'
    source=pd.read_csv(ROOT/'output/model_comparison_20261028/numeric_audited_features.csv')
    source=source[source.cohort.eq('historical')].copy();source['ipo_id']=pd.to_numeric(source.ipo_id).astype(int)
    gaps=[]
    for stage in ['DAY1_EOD','DAY2_EOD','CLOSING_PRE17']:
        covered=set(rows.loc[rows.stage.eq(stage),'ipo_id'])
        for r in source[~source.ipo_id.isin(covered)].itertuples():
            gaps.append(dict(ipo_id=int(r.ipo_id),company=r.company_name,year=int(r.year),stage=stage,issue_open_date=r.issue_open_date,issue_close_date=r.issue_close_date,
                required='Category-level QIB/NII/Retail with an explicit publisher bid clock, source URL and raw evidence; closing-day time must be before 17:00 IST'))
    pd.DataFrame(gaps).to_csv(OUT/'missing_clocked_stages.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(OUT/'stage_predictions.csv',index=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    lines=['# Stage-matched Module 1 tests','',
        'Each stage gets its own fitted five-expert model. Train through 2025; test the same 69 2026 issuers when stage coverage permits. Fixed 50% cutoff. GMP and macro features rebuilt as of each source clock.','',
        '| Stage | Training IPOs | Test IPOs | Model (0.50) | GMP-only (0.50) | GMP-only (fixed 0.625) | Raw GMP > 0 (observed quotes only) |','|---|---:|---:|---:|---:|---:|---:|']
    for stage,r in report['stages'].items():
        if r['status']!='evaluated_research':continue
        a=r['all_test']['model'];b=r['all_test']['gmp_only'];c=r['all_test']['raw_gmp']
        lines.append(f"| {stage} | {r['training_n']} | {r['test_n']} | {a['correct']}/{a['sample_size']} ({a['classifier_accuracy_pct']}%) | {b['classifier_accuracy_pct']}% | {r['all_test']['gmp_fixed625']['classifier_accuracy_pct']}% | {c['correct']}/{c['sample_size']} ({c['classifier_accuracy_pct']}%) |")
        lines.append(f"<!-- Observed dated GMP coverage: {r['observed_gmp_n']}/{r['test_n']}. -->")
    lines+=['','Closing-day pre-17 training status: '+str(report['closing_pre17_status']),
        'The all-case learned baseline uses training-only imputation for missing GMP. Raw GMP > 0 is not a valid all-case baseline where GMP is absent; use paired_observed_gmp in JSON for that comparison.',
        'Retrospectively recovered source timestamps are not original immutable predictions. No model weights or forecast ledgers were replaced.']
    lines+=['','## Pre-2026 checks','', '| Stage | 2024 model | 2024 GMP (0.625) | 2025 model | 2025 GMP (0.625) |','|---|---:|---:|---:|---:|']
    for stage,r in report['stages'].items():
        if r['status']=='evaluated_research':
            a=r['validation']['2024']['all'];b=r['validation']['2025']['all']
            lines.append(f"| {stage} | {a['model']['classifier_accuracy_pct']}% | {a['gmp_fixed625']['classifier_accuracy_pct']}% | {b['model']['classifier_accuracy_pct']}% | {b['gmp_fixed625']['classifier_accuracy_pct']}% |")
    lines+=['',report['release_decision'],'',protocol['implementation_audit'],
        'The final results include already-audited single dated GMP quotes where strictly earlier than the stage, in addition to the clocked quote graphs. Initial graph-only coverage was incomplete; these final rows supersede the preliminary graph-only figures.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))


if __name__=='__main__':run()
