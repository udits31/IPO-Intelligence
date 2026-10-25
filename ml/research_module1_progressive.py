"""Preserve five expert views; fixed-ablation research, never rewrite V0."""
import copy,hashlib,json
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from services.saved_preclose_service import ROOT
from services.ipo_regime_service import rolling_regime,adjusted_gmp
from core.features import availability
from ml.research_module1_stacking import logits,stacker
from ml.research_module1_forward_update import SETS
from ml.run_full_enrichment_audit import metrics,integrity

PREFIX=ROOT/'output/module1_progressive_ablation_20261026'
STATIC=['ofs_ratio','qib_quota_pct','nii_quota_pct','retail_quota_pct','anchor_domestic_mf_pct','lot_size_shares']
VIEWS=['base','base_macro','financial_macro','trajectory']


def time_weights(rows,reference,half_life=2.5):
    age=(pd.Timestamp(reference)-pd.to_datetime(rows.listing_date)).dt.days/365.25
    if age.lt(0).any():raise ValueError('Training listing occurs after weight reference')
    return np.power(.5,age.to_numpy()/half_life)


def weighted_fit(model,x,y,weights=None):
    options={} if weights is None else {'model__sample_weight':weights,'scale__sample_weight':weights}
    return model.fit(x,y,**options)


def feature_matrix(rows,config,components):
    columns=list(SETS.get(config['feature_set'],['gmp_pct']))
    if 'quota' in components and config['feature_set'] in VIEWS:columns+=['quota_weighted_book_mult','log_min_category_mult']
    if 'ofs' in components and config['feature_set'] in VIEWS:columns+=['ofs_ratio','day2_to_cutoff_conviction']
    if 'regime' in components and config['feature_set'] in VIEWS:columns+=['rolling_10_hit_rate']
    x=rows.reindex(columns=columns).copy()
    if 'regime' in components and config['feature_set'] in ('base','gmp_only'):x['gmp_pct']=rows.regime_adjusted_gmp_pct
    if config['transform']=='signed_log':x['gmp_pct']=np.sign(x.gmp_pct)*np.log1p(np.abs(x.gmp_pct))
    return x


def infer(artifact,rows,components=()):
    p=np.column_stack([artifact['models'][n].predict_proba(feature_matrix(rows,artifact['expert_configs'][n],components))[:,1] for n in artifact['experts']])
    return artifact['meta_model'].predict_proba(logits(p))[:,1],p


def build_inputs():
    frame=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv')
    deals=pd.read_csv(ROOT/'data/deal_structure_enriched.csv',dtype={'ipo_id':str}).set_index('ipo_id')
    recent=json.loads((ROOT/'output/listed_closing_day_backtest.json').read_text())['records']
    # Archived native extraction only: no OCR or document downloads.
    native=json.loads((ROOT/'data/audit/ipo_dated_fundamentals_all_20261024_before_offer_placeholder_fix.json').read_text())['records']
    native={str(r['ipo_id']):r for r in native if r.get('facts')}
    rows=[]
    for r in frame.to_dict('records'):
        r.update(issuer_key=str(int(r['ipo_id'])),decision_date=str(r['decision_market_date']),cutoff_at_ist=str(r['decision_market_date'])+'T17:00:00+05:30',cohort='historical')
        rows.append(r)
    canonical_cache=__import__('scrapers.live_ipo_scraper',fromlist=['load_scraped_cache']).load_scraped_cache()
    items={r['company_name']:r for r in canonical_cache.values()}
    for r in recent:
        item=items[r['company_name']];key='recent_'+hashlib.sha256(r['company_name'].encode()).hexdigest()[:10]
        from services.prospective_module1 import calendar
        opened,_=calendar(item,pd.Timestamp(r['cutoff_at_ist']).to_pydatetime())
        o=r.get('outcome');bid=r['bid']
        rows.append(dict(**r['features'],ipo_id=key,issuer_key=key,company_name=r['company_name'],cohort='recent',year=2026,
            issue_open_date=str(opened.date()) if opened else None,decision_date=r['cutoff_at_ist'][:10],cutoff_at_ist=r['cutoff_at_ist'],
            listing_date=item.get('listing_date'),target_binary=int(o['positive']) if o else np.nan,
            target_return=o['return_pct'] if o else np.nan,gmp_preclose_verified=bool(r.get('sources',{}).get('gmp')),
            bid_history=r.get('sources',{}).get('growth_history',[]),bid_source=bid.get('source')))
    # Deduplicate legal issuers across the canonical and separately scraped cohorts.
    from services.historical_bid_service import name_key
    seen=set();history=[]
    for r in rows:
        name=name_key(r['company_name'])
        if name in seen or pd.isna(r.get('target_return')):continue
        seen.add(name)
        verified=r.get('gmp_preclose_verified',pd.notna(r.get('gmp_date')) and pd.Timestamp(r['gmp_date'])<pd.Timestamp(r['decision_date']))
        history.append(dict(issuer_key=r['issuer_key'],listing_date=r['listing_date'],actual_return_pct=r['target_return'],gmp_pct=r.get('gmp_pct'),gmp_preclose_verified=bool(verified)))
    history=pd.DataFrame(history);provenance=[]
    from ml.recover_progressive_totals import recover
    recovered_totals=recover(rows)
    for r in rows:
        key=r['issuer_key'];deal=deals.loc[key].to_dict() if key in deals.index else {}
        # Never reuse an undated OFS column from the old frame without classifying it.
        r['ofs_ratio']=np.nan
        n=native.get(key)
        if n and n.get('document_date') and r.get('issue_open_date') and pd.Timestamp(n['document_date'])<=pd.Timestamp(r['issue_open_date']):
            path=ROOT/n['source_path']
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==n['source_sha256']:
                ratio=n['facts'].get('offer_for_sale_ratio')
                if ratio is not None:deal['ofs_ratio']=ratio
        for field in STATIC:
            value=deal.get(field)
            stamp=availability(kind='PROSPECTUS_STATIC',decision_timestamp=r['cutoff_at_ist'],source_timestamp=deal.get(field+'_source_timestamp') if deal.get(field+'_pit_verified')==True else None,
                fetch_timestamp=deal.get(field+'_fetch_timestamp'),issue_open_date=r.get('issue_open_date'),allow_historical_assumptions=True)
            # A late cache observation does not disprove earlier publication;
            # use the explicit static-offer assumption only in this research path.
            if not stamp['eligible']:
                stamp=availability(kind='PROSPECTUS_STATIC',decision_timestamp=r['cutoff_at_ist'],issue_open_date=r.get('issue_open_date'),fetch_timestamp=deal.get(field+'_fetch_timestamp'),allow_historical_assumptions=True)
            r[field]=value if stamp['eligible'] and pd.notna(value) else np.nan
            if pd.notna(r[field]):provenance.append(dict(issuer_key=key,field=field,**stamp))
        r['log_min_category_mult']=min(r.get(k,np.nan) for k in ('log_qib','log_hni','log_retail'))
        weights=[r.get(k+'_quota_pct',np.nan) for k in ('qib','nii','retail')]
        exact=deal.get('quota_weights_exact')==True
        r['quota_weighted_book_mult']=sum(w/100*np.expm1(r[k]) for w,k in zip(weights,('log_qib','log_hni','log_retail'))) if exact and all(pd.notna(w) for w in weights) and abs(sum(weights)-100)<.5 else np.nan
        # Totals require their own dated observations. Category-growth proxies
        # cannot be relabeled as a Day-2 total or a later closing-day total.
        recovered=recovered_totals.get(key,{})
        r['day2_to_cutoff_conviction']=recovered.get('day2_to_cutoff_conviction',np.nan)
        if 'quota_weighted_book_mult' in recovered:r['quota_weighted_book_mult']=recovered['quota_weighted_book_mult']
        if recovered:provenance.append(dict(issuer_key=key,field='dated_total_and_reservation_reconciliation',**recovered))
        regime=rolling_regime(history,r['decision_date'],exclude_issuer=key)
        r.update(regime);r['regime_adjusted_gmp_pct']=adjusted_gmp(r.get('gmp_pct'),regime['rolling_10_gmp_slippage'])
    result=pd.DataFrame(rows)
    return result,provenance


def fit_candidate(frozen,training,components):
    configs=frozen['expert_configs'];experts=frozen['experts'];oof=[];trace=[]
    for year in (2024,2025):
        held=training[training.year.eq(year)];first=pd.to_datetime(held.decision_date).min()
        fit=training[training.year.lt(year)&pd.to_datetime(training.listing_date).lt(first)]
        if set(fit.issuer_key)&set(held.issuer_key):raise ValueError('Base OOF issuer overlap')
        w=time_weights(fit,pd.to_datetime(fit.listing_date).max()) if 'decay' in components else None
        probabilities=[]
        for n in experts:
            model=weighted_fit(clone(frozen['models'][n]),feature_matrix(fit,configs[n],components),fit.target_binary.astype(int),w)
            probabilities.append(model.predict_proba(feature_matrix(held,configs[n],components))[:,1])
        part=held[['issuer_key','listing_date','target_binary']].copy();part[experts]=np.column_stack(probabilities);oof.append(part)
        trace.append(dict(year=year,fit_ids=fit.issuer_key.tolist(),held_ids=held.issuer_key.tolist(),latest_fit_listing=str(pd.to_datetime(fit.listing_date).max().date()),first_held_decision=str(first.date())))
    meta_rows=pd.concat(oof,ignore_index=True);w=time_weights(training,pd.to_datetime(training.listing_date).max()) if 'decay' in components else None
    models={n:weighted_fit(clone(frozen['models'][n]),feature_matrix(training,configs[n],components),training.target_binary.astype(int),w) for n in experts}
    mw=time_weights(meta_rows,pd.to_datetime(training.listing_date).max()) if 'decay' in components else None
    meta=weighted_fit(stacker(),logits(meta_rows[experts]),meta_rows.target_binary.astype(int),mw)
    return dict(models=models,meta_model=meta,experts=experts,expert_configs=configs,threshold=.5,components=list(components),
        training_ids=training.issuer_key.tolist(),training_end_year=2025,reference_train_end_date=str(pd.to_datetime(training.listing_date).max().date()),
        oof_trace=trace,research_only=True,posthoc_selection_on_inspected_cohorts=True)


def passes(older,recent):return older['correct']>=56 and older['wrong_positive_calls']<=5 and recent['correct']>15


def run():
    original=integrity()
    if not original['all_preserved']:raise ValueError('Frozen baseline integrity check failed')
    frozen=joblib.load(ROOT/'ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib')
    data,provenance=build_inputs();train=data[data.cohort.eq('historical')&data.year.le(2025)].copy()
    test=data[data.cohort.eq('historical')&data.year.eq(2026)].copy();recent=data[data.cohort.eq('recent')&data.target_binary.notna()].copy()
    if (len(train),len(test),len(recent))!=(290,69,23):raise ValueError('Fixed cohorts changed')
    if set(train.issuer_key)&(set(test.issuer_key)|set(recent.issuer_key)):raise ValueError('Training/test issuer overlap')
    fields=STATIC+['quota_weighted_book_mult','log_min_category_mult','day2_to_cutoff_conviction','rolling_10_hit_rate','rolling_10_gmp_slippage']
    preflight={name:{f:int(g[f].notna().sum()) for f in fields} for name,g in [('training_290',train),('test_69',test),('recent_23',recent)]}
    print('ELIGIBLE ROW COUNTS (static terms use explicit historical-offer assumption)',flush=True);print(pd.DataFrame(preflight).to_string(),flush=True)
    predictions=[];results={};accepted=[]
    variants=[('V0_FROZEN',()),('V1_TIME_DECAY',('decay',)),('V2_FAST_REGIME_GMP',('regime',)),('V3_QUOTA_MIN_CAT',('quota',)),('V4_OFS_CONVICTION',('ofs',))]
    for name,components in variants:
        artifact=frozen if not components else fit_candidate(frozen,train,components)
        p,_=infer(artifact,test,components);pr,_=infer(artifact,recent,components)
        older=metrics(test.target_binary,p);now=metrics(recent.target_binary,pr)
        if name=='V0_FROZEN' and (older['correct'],older['wrong_positive_calls'],now['correct'])!=(56,5,15):raise ValueError('Exact V0 replay failed')
        gate=passes(older,now);results[name]=dict(cohort_69=older,recent_23=now,passes_user_gate=gate,components=list(components))
        if components:
            path=ROOT/'ml/artifacts'/('module1_'+name.lower()+'_20261026.joblib');joblib.dump(artifact,path)
            results[name]['artifact']=str(path.relative_to(ROOT))
            if gate:accepted+=list(components)
        for cohort,rows,prob in [('2026_69',test,p),('recent_23',recent,pr)]:
            predictions.extend(dict(variant=name,cohort=cohort,issuer_key=r.issuer_key,company_name=r.company_name,actual_positive=int(r.target_binary),probability=float(q),correct=(q>=.5)==bool(r.target_binary)) for r,q in zip(rows.itertuples(),prob))
        print(json.dumps(dict(variant=name,correct_69=older['correct'],false_positives=older['wrong_positive_calls'],correct_recent=now['correct'],gate=gate)),flush=True)
    combined=tuple(sorted(set(accepted)));champion=fit_candidate(frozen,train,combined) if combined else frozen
    p,_=infer(champion,test,combined);pr,_=infer(champion,recent,combined)
    results['V5_CHAMPION']=dict(cohort_69=metrics(test.target_binary,p),recent_23=metrics(recent.target_binary,pr),components=list(combined),passes_user_gate=passes(metrics(test.target_binary,p),metrics(recent.target_binary,pr)))
    if combined and results['V5_CHAMPION']['passes_user_gate']:joblib.dump(champion,ROOT/'ml/artifacts/module1_progressive_champion_20261026.joblib')
    pd.DataFrame(predictions).to_csv(PREFIX.with_name(PREFIX.name+'_predictions.csv'),index=False)
    data.drop(columns=['bid_history'],errors='ignore').to_csv(PREFIX.with_name(PREFIX.name+'_features.csv'),index=False)
    PREFIX.with_name(PREFIX.name+'_provenance.jsonl').write_text(''.join(json.dumps(r,default=str)+'\n' for r in provenance),encoding='utf8')
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),preflight=preflight,results=results,accepted_components=list(combined),
        training_n=290,training_years=[2021,2022,2023,2024,2025],training_ids=train.issuer_key.tolist(),training_test_overlap=0,
        research_selected_on_inspected_test_sets=True,live_promoted=False,integrity_preserved=integrity()['all_preserved'],
        limitations=['Static cache terms are explicitly presumed to be original pre-open offer terms. This is not proof of historical publication or absence of revisions.','Missing Day-2/cutoff totals remain missing, never synthesized from later final bids.','Fast regime uses only earlier verified listings; retrospective archive-vintage assumptions remain.','Choosing components using the 69 and 23 inspected cases is research selection, not independent test evidence.','Quarterly retraining requires a versioned artifact; it never mutates a frozen forecast.'])
    PREFIX.with_suffix('.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False),encoding='utf8')
    return report


if __name__=='__main__':run()
