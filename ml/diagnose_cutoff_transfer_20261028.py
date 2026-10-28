"""Read-only paired diagnosis of existing research artifacts. No fitting or selection."""
import json
import joblib
import numpy as np
import pandas as pd
from ml.test_historical_joins_and_cutoff_20261028 import ROOT, SOURCE, core, EXPERTS, logits, threshold_metrics


def run():
    olddir=ROOT/'output/historical_join_cutoff_test_20261028'
    newdir=ROOT/'output/through2025_cutoff_test_20261028'
    old=joblib.load(olddir/'baseline_research.joblib')
    new=joblib.load(newdir/'baseline_research.joblib')
    frame=pd.read_csv(SOURCE);frame=frame[frame.cohort.eq('historical')].copy()
    test=frame[frame.year.eq(2026)&frame.verified_stage.eq('day2')].copy().reset_index(drop=True)
    core.SETS['gmp_only']=['gmp_pct']
    candidates=core.candidates()+[dict(name='gmp_only_control',family='logistic',C=.5,class_weight=None,transform='raw',feature_set='gmp_only')]
    configs={name:next(c for c in candidates if c['name']==name) for name in EXPERTS}
    out=dict(protocol='Read-only diagnosis on the identical 64 IPOs; no threshold optimization or model promotion.',models={},year_profiles={},validation_profiles={},counterfactuals={})
    probabilities={}
    for label,bundle in [('through2024',old),('through2025',new)]:
        raw=np.column_stack([bundle['models'][name].predict_proba(core.features(test,configs[name]))[:,1] for name in EXPERTS])
        p=bundle['meta'].predict_proba(logits(raw))[:,1];probabilities[label]=(p,raw)
        out['models'][label]=dict(at50=threshold_metrics(test,p,.5),at625=threshold_metrics(test,p,.625),
            score_by_actual={str(int(y)):dict(n=len(g),median=float(np.median(p[g.index])),q25=float(np.quantile(p[g.index],.25)),q75=float(np.quantile(p[g.index],.75))) for y,g in test.groupby('target_binary')},
            meta_coefficients=dict(zip(EXPERTS,bundle['meta'].named_steps['model'].coef_[0].tolist())),
            expert_auc={name:threshold_metrics(test,raw[:,i],.5)['roc_auc'] for i,name in enumerate(EXPERTS)})
    for base_label,meta_label in [('through2024','through2025'),('through2025','through2024')]:
        meta={'through2024':old,'through2025':new}[meta_label]['meta']
        p=meta.predict_proba(logits(probabilities[base_label][1]))[:,1]
        out['counterfactuals'][base_label+'_experts_'+meta_label+'_meta']=dict(at50=threshold_metrics(test,p,.5),at625=threshold_metrics(test,p,.625))
    for year,g in frame.groupby('year'):
        if year>2025:continue
        out['year_profiles'][str(int(year))]=dict(n=len(g),positive=int(g.target_binary.sum()),gmp_median=float(g.gmp_pct.median()),qib_median=float(g.qib_subscription.median()),
            earnings_observed=int(g.pub_ipo_earnings_yield_pct.notna().sum()),pricebook_observed=int(g.pub_log_ipo_price_book.notna().sum()))
    for label,path in [('through2024',olddir/'baseline_pre2025_validation.csv'),('through2025',newdir/'baseline_pre2026_validation.csv')]:
        val=pd.read_csv(path)
        out['validation_profiles'][label]={str(int(y)):dict(n=len(g),at50=threshold_metrics(g,g.p.to_numpy(),.5),at625=threshold_metrics(g,g.p.to_numpy(),.625),
            positive_median=float(g[g.target_binary.eq(1)].p.median()),negative_median=float(g[g.target_binary.eq(0)].p.median())) for y,g in val.groupby('year')}
    oldp=probabilities['through2024'][0];newp=probabilities['through2025'][0]
    test['through2024_score']=oldp;test['through2025_score']=newp
    test['old625_correct']=(oldp>=.625)==test.target_binary.to_numpy()
    test['new625_correct']=(newp>=.625)==test.target_binary.to_numpy()
    test['new50_correct']=(newp>=.5)==test.target_binary.to_numpy()
    out['cutoff_transfer']=dict(previously_correct_now_wrong=int((test.old625_correct&~test.new625_correct).sum()),previously_wrong_now_correct=int((~test.old625_correct&test.new625_correct).sum()),
        actual_positive_between50_and625=int(((newp>=.5)&(newp<.625)&test.target_binary.eq(1)).sum()),
        actual_negative_between50_and625=int(((newp>=.5)&(newp<.625)&test.target_binary.eq(0)).sum()))
    test[['ipo_id','company_name','target_binary','through2024_score','through2025_score','old625_correct','new625_correct','new50_correct']].to_csv(newdir/'cutoff_transfer_cases.csv',index=False)
    (newdir/'cutoff_transfer_diagnosis.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(out,indent=2))


if __name__=='__main__':run()
