"""Recovered-input five-expert research. Never mutates deployed artifacts/forecasts."""
import copy, hashlib, json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from services.saved_preclose_service import ROOT
from services.retail_hni_economics import capital_plan
from ml.research_module1_forward_update import SETS
from ml.research_module1_stacking import logits, stacker
from ml.research_module1_progressive import time_weights, weighted_fit, passes
from ml.run_full_enrichment_audit import metrics, integrity
from ml.stage_alignment_bridge import CATS, LOGS, GROWTH, run as recover

ARTIFACT=ROOT/'ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib'


def matrix(rows,config,components=()):
    view=config['feature_set'];cols=list(SETS.get(view,['gmp_pct']))
    if 'quota' in components and view!='gmp_only':cols+=['quota_weighted_book_mult','log_min_category_mult']
    if 'ofs' in components and view in ('base','financial_macro'):cols+=['ofs_ratio','is_missing_ofs']
    if 'regime' in components and view!='gmp_only':cols+=['rolling_10_hit_rate']
    x=rows.reindex(columns=cols).copy()
    if 'regime' in components and view in ('base','gmp_only'):x['gmp_pct']=rows.regime_adjusted_gmp_pct
    if config['transform']=='signed_log':x['gmp_pct']=np.sign(x.gmp_pct)*np.log1p(abs(x.gmp_pct))
    return x


def infer(bundle,rows,components=()):
    p=np.column_stack([bundle['models'][n].predict_proba(matrix(rows,bundle['expert_configs'][n],components))[:,1] for n in bundle['experts']])
    return bundle['meta_model'].predict_proba(logits(p))[:,1]


def fit(frozen,training,components=()):
    """Chronological meta folds; no 2026 labels/features fit preprocessing or weights."""
    if training.year.gt(2025).any():raise ValueError('Test IPO passed to fitting')
    pieces=[];trace=[];names=frozen['experts'];configs=frozen['expert_configs']
    for year in (2024,2025):
        held=training[training.year.eq(year)];first=pd.to_datetime(held.decision_date).min()
        known=training[training.year.lt(year)&pd.to_datetime(training.listing_date).lt(first)]
        if held.empty or known.target_binary.nunique()!=2:raise ValueError('Insufficient chronological training fold')
        w=time_weights(known,pd.to_datetime(known.listing_date).max()) if 'decay' in components else None
        p=[]
        for n in names:
            model=weighted_fit(clone(frozen['models'][n]),matrix(known,configs[n],components),known.target_binary.astype(int),w)
            p.append(model.predict_proba(matrix(held,configs[n],components))[:,1])
        block=held[['issuer_key','listing_date','target_binary']].copy();block[names]=np.column_stack(p);pieces.append(block)
        trace.append(dict(fold=year,train_ids=known.issuer_key.tolist(),held_ids=held.issuer_key.tolist(),latest_train_listing=str(pd.to_datetime(known.listing_date).max()),first_held_decision=str(first)))
    meta_rows=pd.concat(pieces);end=pd.to_datetime(training.listing_date).max()
    w=time_weights(training,end) if 'decay' in components else None
    models={n:weighted_fit(clone(frozen['models'][n]),matrix(training,configs[n],components),training.target_binary.astype(int),w) for n in names}
    mw=time_weights(meta_rows,end) if 'decay' in components else None
    meta=weighted_fit(stacker(),logits(meta_rows[names]),meta_rows.target_binary.astype(int),mw)
    return dict(models=models,meta_model=meta,experts=names,expert_configs=configs,training_ids=training.issuer_key.tolist(),oof_trace=trace,
                components=list(components),threshold=.5,research_only=True,posthoc_selection_on_inspected_cohorts=True)


def day2_track(frame):
    out=frame[frame.day2_timestamp.notna() & frame.day2_qib.notna()].copy()
    for i,r in out.iterrows():
        day=pd.Timestamp(r.day2_timestamp)
        out.loc[i,'decision_date']=str(day.date());out.loc[i,'cutoff_at_ist']=day.isoformat()
        for c in CATS:
            out.loc[i,'log_'+c]=np.log1p(r['day2_'+c])
            previous=r['day1_'+c]
            out.loc[i,'log_'+c+'_growth']=np.log1p(r['day2_'+c])-np.log1p(previous) if pd.notna(previous) else np.nan
        # A date-only same-day GMP cannot establish publication by Day2's close.
        if pd.isna(r.gmp_date) or pd.Timestamp(r.gmp_date).date()>=day.date():
            out.loc[i,['gmp_pct','gmp_quote_age_days','gmp_slope_pp_per_day']]=np.nan
        if pd.notna(r.macro_source_date) and pd.Timestamp(r.macro_source_date).date()>day.date():
            out.loc[i,[c for c in out if c.startswith('macro_') and c!='macro_source_date']]=np.nan
    return out


def selective(rows,p):
    led=[]
    for r,q in zip(rows.itertuples(),p):
        values={c:float(np.expm1(getattr(r,'log_'+c))) for c in CATS}
        price=float(np.expm1(r.log_offer_cap_price))
        lot=r.lot_size_shares
        if pd.isna(lot) or lot<=0:
            led.append(dict(issuer_key=r.issuer_key,tier=None,status='missing_original_lot_size'));continue
        plan=capital_plan(score=float(q),regime_adjusted_gmp_pct=float(r.gmp_pct) if pd.notna(r.gmp_pct) else None,
            qib=values['qib'],hni=values['hni'],retail=values['retail'],cap_price=price,lot_size=int(lot),issue_size_cr=r.issue_size_cr)
        realized=None
        if plan.get('tier') in (1,2):
            realized=min(1.,1/max(1.,values['retail']))*price*lot*(r.target_return-.35)/100-9
        led.append(dict(issuer_key=r.issuer_key,tier=plan['tier'],actual_positive=int(r.target_binary),
                        retail_realized_return_scenario_rupees=realized,EU_scenario_rupees=plan.get('retail_EU_scenario_rupees')))
    good=[r for r in led if r['tier'] is not None];apply=[r for r in good if r['tier']<3];losers=[r for r in good if not r['actual_positive']]
    return dict(covered_n=len(good),apply_n=len(apply),apply_precision_pct=100*sum(r['actual_positive'] for r in apply)/len(apply) if apply else None,
                skip_recall_on_losers_pct=100*sum(r['tier']==3 for r in losers)/len(losers) if losers else None,
                retail_allotment_weighted_net_rupee_scenario=sum(r['retail_realized_return_scenario_rupees'] or 0 for r in good),
                snii_net_return=None,bnii_net_return=None,hni_status='eligible_dated_split_books_and_valid_allotment_basis_not_available_for_all_cases',
                exact_allotment_probability=False,records=led)


def route_track(*,has_preclose,has_day2):
    return 'M1_CUTOFF_TRACK' if has_preclose else 'M1_DAY2_TRACK' if has_day2 else 'DATA_PENDING'


def run():
    if not integrity()['all_preserved']:raise ValueError('Frozen baseline changed')
    frozen=joblib.load(ARTIFACT);sha=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    if not sha.startswith('bf07f15c'):raise ValueError('Unexpected frozen artifact')
    recovered,audit,pairs,bridge=recover()
    recovered['is_missing_ofs']=recovered.ofs_ratio.isna().astype(int)
    recovered['log_min_category_mult']=recovered[LOGS].min(axis=1,skipna=False)
    original=pd.read_csv(ROOT/'output/module1_progressive_ablation_20261026_features.csv',dtype={'issuer_key':str})
    train=recovered[recovered.cohort.eq('historical')&recovered.year.le(2025)]
    test=recovered[recovered.cohort.eq('historical')&recovered.year.eq(2026)]
    recent=recovered[recovered.cohort.eq('recent')&recovered.target_binary.notna()]
    assert (len(train),len(test),len(recent))==(290,69,23)
    assert not set(train.issuer_key)&(set(test.issuer_key)|set(recent.issuer_key))
    results={};ledgers=[];accepted=[]
    def score(name,bundle,components,old=False):
        stats={}
        for cohort,rows in [('test_69',test),('recent_23',recent)]:
            data=original.set_index('issuer_key').loc[rows.issuer_key].reset_index() if old else rows
            p=infer(bundle,data,components);stats[cohort]=metrics(rows.target_binary,p)
            stats[cohort]['selective_portfolio']=selective(rows,p)
            ledgers.extend(dict(variant=name,cohort=cohort,issuer_key=r.issuer_key,company_name=r.company_name,probability=float(q),
                                actual_positive=int(r.target_binary),correct=bool((q>=.5)==r.target_binary),actual_open_return_pct=r.target_return) for r,q in zip(rows.itertuples(),p))
        stats['passes_user_gate']=passes(stats['test_69'],stats['recent_23']);stats['components']=list(components)
        extra=recovered[recovered.cohort.eq('recent')&recovered.target_binary.isna()]
        if len(extra)==1 and 'National Stock Exchange' in extra.iloc[0].company_name:
            extra_data=original.set_index('issuer_key').loc[extra.issuer_key].reset_index() if old else extra
            extra_p=float(infer(bundle,extra_data,components)[0])
            base=stats['recent_23'];correct=base['correct']+int(extra_p>=.5)
            stats['recent_24_user_supplied_BSE_sensitivity']=dict(n=24,correct=correct,accuracy_pct=100*correct/24,
                extra_probability=extra_p,issue_price=1785,open_price=1800,primary_archive_verified=False)
        results[name]=stats
        print(json.dumps(dict(variant=name,correct_69=stats['test_69']['correct'],fp=stats['test_69']['wrong_positive_calls'],correct_recent=stats['recent_23']['correct'],gate=stats['passes_user_gate'])),flush=True)
        return stats
    prior=score('M1_V0_FROZEN_PRIOR',frozen,(),True)
    assert (prior['test_69']['correct'],prior['test_69']['wrong_positive_calls'],prior['recent_23']['correct'])==(56,5,15)
    score('M1_V1_FROZEN_WITH_RECOVERED_INPUTS',frozen,())
    results['M1_V2_STAGE_ALIGNED']=dict(status='blocked_no_genuine_training_Day2_preclose_pairs',bridge_training_pair_n=bridge.n_pairs,
        final_deflation='not_identified_without_paired_final_and_preclose_books',passes_user_gate=False)
    for name,components in [('M1_V3_QUOTA_DENOMINATORS',('quota',)),('M1_V4_OFS_SPLIT',('ofs',)),('M1_V5_PROGRESSIVE_REGIME',('decay','regime'))]:
        candidate=fit(frozen,train,components);stats=score(name,candidate,components)
        joblib.dump(candidate,ROOT/'ml/artifacts'/(name.lower()+'_stage_research_20261026.joblib'))
        if stats['passes_user_gate']:accepted.extend(components)
    combined=tuple(sorted(set(accepted)))
    if combined:
        candidate=fit(frozen,train,combined);champion=score('M1_CHAMPION_CANDIDATE',candidate,combined)
        selected='M1_CHAMPION_CANDIDATE' if champion['passes_user_gate'] else 'M1_V0_FROZEN_PRIOR'
    else:selected='M1_V1_FROZEN_WITH_RECOVERED_INPUTS' if results['M1_V1_FROZEN_WITH_RECOVERED_INPUTS']['passes_user_gate'] else 'M1_V0_FROZEN_PRIOR'
    # Separate observed Day2 model: no synthetic Day3 targets or date-only same-day GMP.
    d2=day2_track(recovered);dt=d2[d2.cohort.eq('historical')&d2.year.le(2025)]
    model=fit(frozen,dt)
    day2_results={}
    for name,rows in [('test_69',d2[d2.cohort.eq('historical')&d2.year.eq(2026)]),('recent_23',d2[d2.cohort.eq('recent')&d2.target_binary.notna()])]:
        p=infer(model,rows);day2_results[name]=metrics(rows.target_binary,p)
        ledgers.extend(dict(variant='M1_DAY2_TRACK',cohort=name,issuer_key=r.issuer_key,company_name=r.company_name,probability=float(q),actual_positive=int(r.target_binary),correct=bool((q>=.5)==r.target_binary),actual_open_return_pct=r.target_return) for r,q in zip(rows.itertuples(),p))
    joblib.dump(model,ROOT/'ml/artifacts/module1_day2_stage_track_research_20261026.joblib')
    results['M1_DAY2_TRACK']=dict(training_n=len(dt),training_ids=dt.issuer_key.tolist(),metrics=day2_results,
         source_basis='Publisher-labelled Day2 EOD tables recovered later; date-only same-day GMP withheld.',live_promoted=False)
    day2_test=day2_results['test_69'];day2_recent=day2_results['recent_23']
    results['M1_DAY2_TRACK']['passes_user_gate']=bool(day2_test['n']==69 and day2_recent['n']==23 and passes(day2_test,day2_recent))
    # User-supplied BSE result is a separate sensitivity until a source archive is bound.
    extra=recovered[recovered.cohort.eq('recent')&recovered.target_binary.isna()]
    supplied=[]
    for r in extra.itertuples():
        if 'National Stock Exchange' not in r.company_name:continue
        probability=float(infer(frozen,pd.DataFrame([r._asdict()]))[0])
        supplied.append(dict(company_name=r.company_name,issue_price=1785.,actual_open=1800.,actual_return_pct=100*(1800/1785-1),
            probability=probability,correct=probability>=.5,outcome_evidence='user_supplied_BSE_open_not_bound_to_primary_archive',included_in_primary_23=False))
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256=sha,training_n=290,training_years=[2021,2022,2023,2024,2025],
        training_test_issuer_overlap=0,results=results,selected=selected,accepted_components=list(combined),additional_24th_case_sensitivity=supplied,
        stage_bridge_status=bridge.status,live_promoted=False,integrity_preserved=integrity()['all_preserved'],
        limitations=['The inspected 69 and 23 cohorts are development comparisons, not a fresh independent test.',
          'The 161 total ratios are not 161 verified distinct Day2/preclose category pairs.',
          'No arbitrary final-book discount or unverified statutory reservation fallback is applied.',
          'Allotment-weighted returns are scenarios; no exact HNI return is fabricated from aggregate NII.',
          'Frozen-model recovered-input inference is a changed-input replay, not a saved historical call.'])
    path=ROOT/'output/module1_controlled_stage_ablation_20261026.json';path.write_text(json.dumps(report,indent=2,default=str,allow_nan=False))
    pd.DataFrame(ledgers).to_csv(path.with_suffix('.csv'),index=False)
    return report

if __name__=='__main__':run()
