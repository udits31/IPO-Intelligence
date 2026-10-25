"""Chronological enrichment challenger; frozen live models remain unchanged."""
import json,math
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss,mean_absolute_error
from services.saved_preclose_service import ROOT
from services.prospective_module1 import predict
from services.point_in_time import eligible
from ml.research_module1_forward_update import BASE,MACRO,FINANCIAL,TRAJECTORY

EXTRA=['ofs_ratio','ofs_x_pe_interaction','anchor_domestic_mf_pct','brlm_prior_listing_win_rate',
       'quota_weighted_book_mult','log_min_category_mult','bnii_to_snii_ratio','day3_late_surge_ratio',
       'is_missing_financials','is_missing_gmp']
COLUMNS=BASE+MACRO+FINANCIAL+TRAJECTORY+EXTRA


def classifier():
    return make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.25,max_iter=4000,random_state=42))


def enrich(features,deal,cutoff):
    # Old research frames may already contain undated offer fields. They must
    # not bypass the new per-field timestamp gate merely by sharing a name.
    x={k:v for k,v in features.items() if k not in EXTRA};x['is_missing_gmp']=float('gmp_pct' not in x or not np.isfinite(x['gmp_pct']))
    x['is_missing_financials']=float(not any(np.isfinite(x.get(k,np.nan)) for k in FINANCIAL))
    x['log_min_category_mult']=min(x.get('log_qib',np.nan),x.get('log_hni',np.nan),x.get('log_retail',np.nan))
    admitted={}
    for key in ('ofs_ratio','anchor_domestic_mf_pct','brlm_prior_listing_win_rate','qib_quota_pct','nii_quota_pct','retail_quota_pct','snii_subscription_mult','bnii_subscription_mult'):
        if deal and pd.notna(deal.get(key)) and bool(deal.get(key+'_pit_verified',False)) and deal.get(key+'_source_timestamp'):
            record=dict(source_timestamp=deal[key+'_source_timestamp'],source_timestamp_verified=True)
            if eligible(record,cutoff):admitted[key]=float(deal[key])
    x.update({k:v for k,v in admitted.items() if k in EXTRA})
    earnings=x.get('pub_ipo_earnings_yield_pct')
    if 'ofs_ratio' in admitted and earnings and earnings>0:x['ofs_x_pe_interaction']=admitted['ofs_ratio']*100/earnings
    weights=[admitted.get(k) for k in ('qib_quota_pct','nii_quota_pct','retail_quota_pct')]
    if deal.get('quota_weights_exact') is True and all(w is not None for w in weights) and abs(sum(weights)-100)<.5:
        x['quota_weighted_book_mult']=sum(w/100*np.expm1(x[k]) for w,k in zip(weights,('log_qib','log_hni','log_retail')))
    if all(k in admitted for k in ('snii_subscription_mult','bnii_subscription_mult')):
        x['bnii_to_snii_ratio']=float(np.log1p(admitted['bnii_subscription_mult'])-np.log1p(admitted['snii_subscription_mult']))
    # No same-provider dated Day-2 total means the surge remains missing.
    return x


def scores(rows,probabilities,threshold):
    y=np.asarray(rows.target_binary,dtype=int);p=np.asarray(probabilities)
    call=p>=.5;apply=p>=threshold;loser=y==0;skip=p<.5
    return dict(n=len(y),correct=int((call==y).sum()),accuracy_pct=100*float((call==y).mean()),brier=float(brier_score_loss(y,p)),
        apply_count=int(apply.sum()),apply_precision_pct=100*float(y[apply].mean()) if apply.any() else None,
        skip_recall_on_losers_pct=100*float(skip[loser].mean()) if loser.any() else None)


def run():
    data=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv')
    deals=pd.read_csv(ROOT/'data/deal_structure_enriched.csv',dtype={'ipo_id':str}).set_index('ipo_id')
    training=data[data.year.le(2025)].copy();test=data[data.year.eq(2026)].copy()
    assert not set(training.ipo_id)&set(test.ipo_id)
    def matrix(rows):
        result=[]
        for row in rows.to_dict('records'):
            f={k:float(row[k]) for k in COLUMNS if k in row and pd.notna(row[k])}
            deal=deals.loc[str(int(row['ipo_id']))].to_dict() if str(int(row['ipo_id'])) in deals.index else {}
            result.append(enrich(f,deal,str(row['decision_market_date'])+'T17:00:00+05:30'))
        return pd.DataFrame(result).reindex(columns=COLUMNS)
    x=matrix(training);xtest=matrix(test);oof=np.full(len(training),np.nan);oof_quantiles=np.full((len(training),3),np.nan)
    years=training.year.to_numpy();trace=[]
    for year in (2022,2023,2024,2025):
        fitting=np.where(years<year)[0];held=np.where(years==year)[0]
        if not len(held):continue
        assert pd.to_datetime(training.iloc[fitting].listing_date).max()<pd.to_datetime(training.iloc[held].decision_market_date).min()
        c=classifier().fit(x.iloc[fitting],training.iloc[fitting].target_binary)
        oof[held]=c.predict_proba(x.iloc[held])[:,1]
        for j,q in enumerate((.2,.5,.8)):
            reg=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=100,max_leaf_nodes=4,min_samples_leaf=15,l2_regularization=5,random_state=42))
            reg.fit(x.iloc[fitting],training.iloc[fitting].target_return);oof_quantiles[held,j]=reg.predict(x.iloc[held])
        trace.append(dict(test_year=year,fit_ids=list(map(int,training.iloc[fitting].ipo_id)),validation_ids=list(map(int,training.iloc[held].ipo_id))))
    older=np.isfinite(oof)&(years<=2024);validation=np.isfinite(oof)&(years==2025)
    calibration=IsotonicRegression(out_of_bounds='clip').fit(oof[older],training.loc[older,'target_binary'])
    valid_p=calibration.predict(oof[validation]);valid_y=training.loc[validation,'target_binary'].to_numpy()
    viable=[]
    for threshold in np.linspace(.5,.99,50):
        calls=valid_p>=threshold
        if calls.sum()>=10 and valid_y[calls].mean()>=.85:viable.append((int(calls.sum()),float(threshold)))
    threshold=max(viable,key=lambda a:(a[0],-a[1]))[1] if viable else 1.01
    model=classifier().fit(x,training.target_binary);raw=model.predict_proba(xtest)[:,1];p=calibration.predict(raw)
    # Split CQR: fit through 2024, calibrate 2025; do not refit the return heads on calibration targets.
    fit=years<=2024;cal=years==2025;quantile_models=[];cal_preds=[];test_preds=[]
    for q in (.2,.5,.8):
        reg=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=100,max_leaf_nodes=4,min_samples_leaf=15,l2_regularization=5,random_state=42))
        reg.fit(x.loc[fit],training.loc[fit,'target_return']);quantile_models.append(reg);cal_preds.append(reg.predict(x.loc[cal]));test_preds.append(reg.predict(xtest))
    cq=np.sort(np.column_stack(cal_preds),axis=1);q=np.sort(np.column_stack(test_preds),axis=1);ycal=training.loc[cal,'target_return'].to_numpy()
    conformity=np.maximum(cq[:,0]-ycal,ycal-cq[:,2]);level=min(1.,np.ceil((len(conformity)+1)*.6)/len(conformity));radius=float(np.quantile(conformity,level,method='higher'))
    q[:,0]-=radius;q[:,2]+=radius
    means=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),HistGradientBoostingRegressor(loss='squared_error',max_iter=100,max_leaf_nodes=4,min_samples_leaf=15,l2_regularization=5,random_state=42)).fit(x.loc[fit],training.loc[fit,'target_return'])
    result=dict(training_n=len(training),training_end_year=2025,training_test_overlap=0,calibration='isotonic fitted only on earlier-year out-of-fold probabilities; 2025 selects selective precision hurdle',
        selective_probability_threshold=threshold,validation_precision_target=.85,walk_forward_trace=trace,
        test_69=scores(test,p,threshold),raw_test_69=scores(test,raw,threshold),
        return_cqr=dict(nominal_central_coverage=.6,calibration_year=2025,fit_through=2024,correction_pp=radius,observed_2026_coverage=float(((test.target_return>=q[:,0])&(test.target_return<=q[:,2])).mean()),median_mae_pp=float(mean_absolute_error(test.target_return,q[:,1]))),
        observed_extra_training={k:int(x[k].notna().sum()) for k in EXTRA},live_promoted=False,
        limitations=['OOF isotonic is not a guarantee of future calibration under stage shift.','CQR nominal coverage relies on calibration/test assumptions; chronological distribution shift can break coverage.','Unavailable publication times exclude static enrichment features.','No exact allotment-adjusted portfolio return is claimed without valid category application counts.'])
    artifact=dict(model=model,calibration=calibration,features=COLUMNS,threshold=threshold,quantile_models=quantile_models,mean_return_model=means,cqr_radius=radius,research_only=True,training_ids=list(map(int,training.ipo_id)))
    joblib.dump(artifact,ROOT/'ml/artifacts/module1_full_enrichment_research_20261025.joblib')
    test['enriched_probability']=p;test['enriched_correct']=(p>=.5)==test.target_binary;test['q20']=q[:,0];test['q50']=q[:,1];test['q80']=q[:,2];test['mean_return_estimate']=means.predict(xtest)
    test.to_csv(ROOT/'output/module1_enriched_2026_predictions.csv',index=False)
    # Same recent inputs; do not use actual outcomes in enrichment or inference.
    recent=json.loads((ROOT/'output/listed_closing_day_backtest.json').read_text())['records'];recent_rows=[]
    for row in recent:
        key='recent_'+__import__('hashlib').sha256(row['company_name'].encode()).hexdigest()[:10]
        deal=deals.loc[key].to_dict() if key in deals.index else {}
        f=enrich(row['features'],deal,row['cutoff_at_ist']);m=pd.DataFrame([f]).reindex(columns=COLUMNS)
        probability=float(calibration.predict(model.predict_proba(m)[:,1])[0]);o=row.get('outcome')
        recent_rows.append(dict(company_name=row['company_name'],probability=probability,mean_return_estimate=float(means.predict(m)[0]),correct=((probability>=.5)==o['positive']) if o else None,actual_positive=o['positive'] if o else None))
    resolved=[r for r in recent_rows if r['correct'] is not None]
    result['recent']=dict(records=recent_rows,n=len(resolved),correct=sum(r['correct'] for r in resolved),accuracy_pct=100*sum(r['correct'] for r in resolved)/len(resolved))
    (ROOT/'output/module1_full_enrichment_benchmark_20261025.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('walk_forward_trace','recent','limitations')}),flush=True)
    return result

if __name__=='__main__':run()
