"""Period-by-period causal tests with fixed decisions and paired GMP controls."""
import json
import numpy as np
import pandas as pd
from ml.finalize_module1_stability_20261028 import OUT as INPUT, fit_policy
from ml.test_historical_joins_and_cutoff_20261028 import ROOT, threshold_metrics
from ml.compare_model_generations_20261028 import paired
from ml.test_historical_joins_and_cutoff_20261028 import sha,SOURCE

OUT=ROOT/'output/module1_period_stability_20261028'


def scope_metrics(rows,p,threshold=.5):
    if rows.target_binary.nunique()>1:return threshold_metrics(rows,np.asarray(p),threshold)
    y=rows.target_binary.astype(int).to_numpy();p=np.asarray(p);call=p>=threshold
    return dict(sample_size=len(rows),positive_listings=int(y.sum()),nonpositive_listings=int((y==0).sum()),
        correct=int((call==y).sum()),classifier_accuracy_pct=round(100*float((call==y).mean()),1),
        brier_score=round(float(np.mean((p-y)**2)),4),roc_auc=None,balanced_accuracy_pct=None,
        tn=int(((~call)&(y==0)).sum()),fp=int((call&(y==0)).sum()),fn=int(((~call)&(y==1)).sum()),tp=int((call&(y==1)).sum()))


def compare(rows):
    result={}
    for label,key in [('model','model_score'),('gmp_only','gmp_score'),('always_positive','always_positive')]:
        result[label]=scope_metrics(rows,rows[key].to_numpy())
    result['paired_model_vs_gmp']=paired(rows,rows.model_score.to_numpy(),rows.gmp_score.to_numpy())
    result['raw_gmp_positive_rule']=scope_metrics(rows,(rows.gmp_pct>0).astype(float).to_numpy())
    return result


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    protocol=dict(model_cutoff=.5,gmp_cutoff=.5,raw_gmp_rule='positive if dated GMP > 0%',
        periods='2023–2025 expanding causal base/meta fits; 2026 same actual frozen release and fitted GMP expert',
        primary='Only identical stage-verified Day-2 issuer rows with observed dated GMP in both arms',
        all69='Separate sensitivity includes five unresolved-stage rows and GMP imputation when applicable',
        missing='Reported explicitly, never zero-filled; exclusions identical across arms',
        posthoc_selection=False,warning='Previously examined retrospective data; small quarters are descriptive, not independent validation')
    protocol['architecture_selection_caveat']='The expert architecture was chosen during earlier development. Earlier-year folds refit weights causally but are retrospective diagnostics of that fixed design, not untouched architecture-selection holdouts.'
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    oof=pd.read_csv(INPUT/'base_expert_causal_oof.csv')
    for key in ['decision_market_date','listing_date']:oof[key]=pd.to_datetime(oof[key])
    records=[]
    for year in [2023,2024,2025]:
        held=oof[oof.year.eq(year)].copy();fit=oof[oof.year.lt(year)]
        if fit.listing_date.max()>=held.decision_market_date.min():raise ValueError('Future meta labels')
        held['model_score'],_=fit_policy('incumbent_C05',fit,held)
        held['gmp_score']=held.gmp_only_control
        held['fit_end_year']=year-1
        records.append(held)
    predictions=pd.read_csv(INPUT/'paired_predictions.csv')
    source=pd.read_csv(ROOT/'output/model_comparison_20261028/numeric_audited_features.csv')
    held=source[source.cohort.eq('historical')&source.year.eq(2026)].copy()
    held['ipo_id']=pd.to_numeric(held.ipo_id).astype(int)
    held=held.merge(predictions[['ipo_id','actual_frozen_artifact']],on='ipo_id',validate='one_to_one')
    # GMP expert in this release is the same fixed logistic configuration refitted through 2025.
    import joblib
    from ml.test_through2025_cutoff_20261028 import ARTIFACT
    artifact=joblib.load(ARTIFACT)
    held['gmp_score']=artifact['models']['gmp_only_control'].predict_proba(held[['gmp_pct']])[:,1]
    held['model_score']=held.actual_frozen_artifact;held['fit_end_year']=2025
    records.append(held)
    frame=pd.concat(records,ignore_index=True);frame['always_positive']=1.
    frame['decision_market_date']=pd.to_datetime(frame.decision_market_date)
    frame['period']=frame.decision_market_date.dt.to_period('Q').astype(str)
    quote_date=pd.to_datetime(frame.gmp_date,errors='coerce')
    frame['dated_gmp_eligible']=frame.gmp_pct.notna()&quote_date.notna()&(quote_date<frame.decision_market_date)
    clean=frame[frame.verified_stage.eq('day2')&frame.dated_gmp_eligible].copy()
    report=dict(protocol=protocol,input_hashes={str(SOURCE.relative_to(ROOT)).replace('\\','/'):sha(SOURCE),str(ARTIFACT.relative_to(ROOT)).replace('\\','/'):sha(ARTIFACT)},coverage={},by_year={},by_quarter={},all69_2026=compare(frame[frame.year.eq(2026)]))
    from services.prospective_module1 import PROTOCOL
    live_protocol=json.loads(PROTOCOL.read_text())
    gmp_bundle=joblib.load(ROOT/live_protocol['gmp_baseline']['path'])
    report['input_hashes'][str(PROTOCOL.relative_to(ROOT)).replace('\\','/')]=sha(PROTOCOL)
    report['input_hashes'][live_protocol['gmp_baseline']['path']]=sha(ROOT/live_protocol['gmp_baseline']['path'])
    report['current_app_gmp_comparison_2026']={}
    for name,rows in [('same69',frame[frame.year.eq(2026)]),('paired_primary',clean[clean.year.eq(2026)])]:
        p=gmp_bundle['gmp_control'].predict_proba(rows[['gmp_pct']])[:,1]
        report['current_app_gmp_comparison_2026'][name]=dict(model=scope_metrics(rows,rows.model_score.to_numpy()),gmp_only=scope_metrics(rows,p,live_protocol['gmp_baseline']['cutoff']),gmp_cutoff=live_protocol['gmp_baseline']['cutoff'],paired=paired(rows,rows.model_score.to_numpy(),(p>=live_protocol['gmp_baseline']['cutoff']).astype(float)))
    for year,g in frame.groupby('year'):
        mask=clean.year.eq(year)
        report['coverage'][str(int(year))]=dict(available=len(g),stage_verified=int(g.verified_stage.eq('day2').sum()),dated_gmp_observed=int(g.gmp_pct.notna().sum()),paired_primary=int(mask.sum()))
    for year,g in clean.groupby('year'):report['by_year'][str(int(year))]=compare(g)
    for quarter,g in clean.groupby('period'):report['by_quarter'][quarter]=compare(g)
    report['summary']=dict(years_beating_gmp=sum(r['model']['correct']>r['gmp_only']['correct'] for r in report['by_year'].values()),years_tied_gmp=sum(r['model']['correct']==r['gmp_only']['correct'] for r in report['by_year'].values()),years_losing_gmp=sum(r['model']['correct']<r['gmp_only']['correct'] for r in report['by_year'].values()))
    frame.to_csv(OUT/'all_predictions.csv',index=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    lines=['# Chronological stability versus GMP-only','',protocol['periods'], '',
        'Both direction thresholds fixed at 0.50. Primary rows require observed dated GMP and verified Day-2 stage. All arms use identical issuer IDs and listing outcomes.','',
        '| Test year | IPOs | Five-expert correct | Learned GMP-only correct | Raw GMP > 0 rule | Always positive | Model minus learned GMP |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for year,r in report['by_year'].items():
        a=r['model'];b=r['gmp_only'];c=r['always_positive']
        raw=r['raw_gmp_positive_rule']
        lines.append(f"| {year} | {a['sample_size']} | {a['correct']} ({a['classifier_accuracy_pct']}%) | {b['correct']} ({b['classifier_accuracy_pct']}%) | {raw['classifier_accuracy_pct']}% | {c['classifier_accuracy_pct']}% | {a['classifier_accuracy_pct']-b['classifier_accuracy_pct']:+.1f} pp |")
    lines+=['','## Quarterly checks','', '| Quarter | IPOs | Model accuracy | GMP-only accuracy |','|---|---:|---:|---:|']
    for q,r in report['by_quarter'].items():
        a=r['model'];b=r['gmp_only'];lines.append(f"| {q} | {a['sample_size']} | {a['classifier_accuracy_pct']}% | {b['classifier_accuracy_pct']}% |")
    a=report['all69_2026']['model'];b=report['all69_2026']['gmp_only']
    lines+=['',f"All-69 2026 sensitivity: model {a['correct']}/69 ({a['classifier_accuracy_pct']}%), GMP-only {b['correct']}/69 ({b['classifier_accuracy_pct']}%).",'',
        json.dumps(report['summary']), '', protocol['warning'],
        'Scores for other bidding stages remain provisional. Historical Day-2 results cannot certify intraday or closing-day performance. The learned GMP-only control and the raw GMP-sign rule are both reported in JSON; they are different baselines.']
    lines+=['',protocol['architecture_selection_caveat']]
    app=report['current_app_gmp_comparison_2026']['same69']
    lines+=['',f"Actual app GMP-only artifact at its separately locked {app['gmp_cutoff']:.3f} cutoff: {app['gmp_only']['correct']}/69 ({app['gmp_only']['classifier_accuracy_pct']}%). This additional 2026-only comparison preserves the app baseline; it is not retrospectively applied as a selected policy to earlier validation years."]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))


if __name__=='__main__':run()
