"""Frozen Module 1 coverage/direction/return/ranking research completion.

No live artifact or paper changes. Recovery is additive and source dated.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.model_pipeline import FEATURE_COLS,summarize
from ml.research_monitor_daily_learning import paired_gain
from ml.train_stage_fundamental_challenger import ROOT,EXTRA
from ml.module1_gap_recovery import OUT as RECOVERY,AUDIT as RECOVERY_AUDIT

OUT=ROOT/'output/module1_completion_20261024.json'
FRAME=ROOT/'data/audit/module1_completion_early_20261024.csv'
MANIFEST=ROOT/'output/module1_completion_inputs_20261024.json'
BASE=FEATURE_COLS+EXTRA[:2]+['gmp_pct']
FULL=FEATURE_COLS+EXTRA+['gmp_pct']
TRACKS={'gmp_fitted':['gmp_pct'],'bids_offer_gmp':BASE,'full_facts_gmp':FULL,'sector_relative_gmp':FULL+['financial_sector']}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SectorRelativeFacts(BaseEstimator,TransformerMixin):
    """Training-only financial-sector-relative robust transformations."""
    def fit(self,x,y=None):
        self.statistics_={}
        for field in ('earnings_yield_pct','eps_growth_asinh'):
            values=pd.to_numeric(x[field],errors='coerce').dropna()
            global_stats=self.stats(values)
            self.statistics_[field]={-1:global_stats}
            for sector in (0.,1.):
                subset=pd.to_numeric(x.loc[x.financial_sector.eq(sector),field],errors='coerce').dropna()
                self.statistics_[field][sector]=self.stats(subset) if len(subset)>=5 else global_stats
        return self

    @staticmethod
    def stats(values):
        if len(values)==0:
            return (0.,0.,0.,1.)
        lo,hi=values.quantile([.05,.95]);median=values.median()
        iqr=float(values.quantile(.75)-values.quantile(.25))
        return (float(lo),float(hi),float(median),iqr if iqr>1e-8 else 1.)

    def transform(self,x):
        result=x.copy()
        for field,groups in self.statistics_.items():
            result[field]=pd.to_numeric(result[field],errors='coerce').astype(float)
            for sector in result.financial_sector.unique():
                mask=result.financial_sector.eq(sector)
                lo,hi,median,scale=groups.get(sector,groups[-1])
                result.loc[mask,field]=(pd.to_numeric(result.loc[mask,field],errors='coerce').clip(lo,hi)-median)/scale
        return result


def estimator(track,kind):
    steps=[]
    if track=='sector_relative_gmp':
        steps.append(('relative',SectorRelativeFacts()))
    steps += [('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]
    model=(LogisticRegression(C=.5,max_iter=2000,random_state=42) if kind=='direction' else
        Ridge(alpha=20) if kind=='ridge' else GradientBoostingRegressor(loss='huber',n_estimators=100,
        max_depth=2,min_samples_leaf=10,learning_rate=.04,random_state=42))
    return Pipeline(steps+[('model',model)])


def attach_quotes(early,quotes,canonical):
    """Conflicting daily values are withheld; no same-day or stale quotes."""
    result=early.copy()
    result['original_gmp_covered']=result.gmp_pct.notna()
    result['gmp_conflict_days']=0
    quotes=quotes.copy();quotes['gmp_date']=pd.to_datetime(quotes.gmp_date,errors='coerce')
    quotes['gmp_rs']=pd.to_numeric(quotes.gmp_rs,errors='coerce')
    for index,row in result.iterrows():
        day=pd.Timestamp(row.decision_market_date)
        eligible=quotes[quotes.ipo_id.eq(row.ipo_id)&quotes.gmp_date.lt(day)&
            quotes.gmp_date.ge(day-pd.Timedelta(days=7))&quotes.gmp_rs.notna()].copy()
        counts=eligible.groupby('gmp_date').gmp_rs.nunique()
        conflict_dates=counts[counts.gt(1)].index
        result.loc[index,'gmp_conflict_days']=len(conflict_dates)
        eligible=eligible[~eligible.gmp_date.isin(conflict_dates)]
        result.loc[index,['gmp_pct','gmp_date','gmp_source_url']]=[np.nan,None,None]
        if eligible.empty:
            continue
        latest=eligible.sort_values(['gmp_date','source_url']).iloc[-1]
        cap=float(canonical.loc[row.ipo_id].price_band_upper)
        if not np.isfinite(cap) or cap<=0:
            raise ValueError('No verified offer cap for GMP normalization')
        result.loc[index,'gmp_pct']=100*float(latest.gmp_rs)/cap
        result.loc[index,'gmp_date']=latest.gmp_date.strftime('%Y-%m-%d')
        result.loc[index,'gmp_source_url']=latest.source_url
    return result


def prepare():
    manifest=json.loads(MANIFEST.read_text())
    for relative,expected in manifest.items():
        if digest(ROOT/relative)!=expected:
            raise ValueError(f'Completion source changed: {relative}')
    recovery=json.loads(RECOVERY_AUDIT.read_text())
    if recovery['completed']!=recovery['attempted']:
        raise ValueError('Gap collection is incomplete')
    early=pd.read_csv(FRAME)
    for field in ('listing_date','decision_market_date','issue_close_date','issue_open_date'):
        early[field]=pd.to_datetime(early[field])
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    quotes=pd.concat([pd.read_csv(ROOT/'data/preclose_gmp_history.csv'),pd.read_csv(RECOVERY)],ignore_index=True)
    early=attach_quotes(early,quotes,canonical)
    early['financial_sector']=early.ipo_id.map(canonical.sector.eq('Financial Services').astype(float))
    if early.financial_sector.isna().any():
        raise ValueError('Unknown sector must not become nonfinancial')
    early['offer_cap_matches_issue_price']=np.isclose(np.expm1(early.log_offer_cap_price),early.issue_price)
    # Preserve the original test denominator even if a newly discovered conflict
    # makes a previous observation unusable: fail visibly rather than drop it.
    if early.loc[early.original_gmp_covered,'gmp_pct'].isna().any():
        raise ValueError('New GMP conflict invalidates an original test observation')
    return early,canonical,quotes


def paired_mean(values):
    values=np.asarray(values,dtype=float)
    if not len(values):
        return dict(n=0,mean=None,bootstrap_95=None)
    rng=np.random.default_rng(42)
    boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    return dict(n=len(values),mean=float(values.mean()),bootstrap_95=np.quantile(boot,[.025,.975]).tolist())


def return_metrics(rows,name):
    error=rows.target_return.to_numpy()-rows[name].to_numpy()
    gmp_error=rows.target_return.to_numpy()-rows.gmp_pct.to_numpy()
    return dict(n=len(rows),mae_pp=float(np.abs(error).mean()),rmse_pp=float(np.sqrt(np.square(error).mean())),
        median_absolute_error_pp=float(np.median(np.abs(error))),
        return_direction_accuracy_pct=float(((rows[name]>0)==rows.target_binary.astype(bool)).mean()*100),
        mae_improvement_vs_gmp=paired_mean(np.abs(gmp_error)-np.abs(error)))


def rankings(rows,name):
    choices=[];pair_correct=0.;gmp_pair_correct=0.;pairs=0
    for date,group in rows.groupby('decision_market_date'):
        if len(group)<2:
            continue
        selected=group.sort_values([name,'ipo_id'],ascending=[False,True]).iloc[0]
        baseline=group.sort_values(['gmp_pct','ipo_id'],ascending=[False,True]).iloc[0]
        model_return=float(selected.target_return) if selected[name]>0 else 0.
        gmp_return=float(baseline.target_return) if baseline.gmp_pct>0 else 0.
        choices.append(dict(decision_date=str(pd.Timestamp(date).date()),n=len(group),
            eligible_ids=sorted(map(int,group.ipo_id)),model_choice=selected.company_name if selected[name]>0 else 'SKIP ALL',
            gmp_choice=baseline.company_name if baseline.gmp_pct>0 else 'SKIP ALL',
            model_opening_return_pct=model_return,gmp_opening_return_pct=gmp_return,difference_pp=model_return-gmp_return))
        records=group.to_dict('records')
        for i,left in enumerate(records):
            for right in records[i+1:]:
                actual=left['target_return']-right['target_return']
                if actual==0:
                    continue
                difference=left[name]-right[name];gmp=left['gmp_pct']-right['gmp_pct']
                pair_correct+=.5 if difference==0 else float(difference*actual>0)
                gmp_pair_correct+=.5 if gmp==0 else float(gmp*actual>0)
                pairs+=1
    delta=paired_mean([item['difference_pp'] for item in choices])
    return dict(panels=len(choices),choices=choices,paired_selection_difference_pp=delta,
        pair_count=pairs,pairwise_accuracy_pct=100*pair_correct/pairs if pairs else None,
        gmp_pairwise_accuracy_pct=100*gmp_pair_correct/pairs if pairs else None,
        established_edge=bool(len(choices)>=20 and delta['bootstrap_95'] and delta['bootstrap_95'][0]>0))


def coverage(early,canonical,quotes):
    payload=json.loads((ROOT/'data/audit/ipo_dated_fundamentals_all_20261024.json').read_text())
    records={r['ipo_id']:r for r in payload['records']}
    early_map=early.set_index('ipo_id')
    dates=pd.read_csv(ROOT/'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    outcomes=pd.read_csv(ROOT/'data/audit/module1_exchange_listing_open_audit.csv').set_index('ipo_id')
    recovery_audit=json.loads(RECOVERY_AUDIT.read_text())
    recovery_status={r['ipo_id']:r['status'] for r in recovery_audit['sources']}
    rows=[]
    for ipo_id,record in records.items():
        item=canonical.loc[ipo_id]
        row=dict(ipo_id=ipo_id,company_name=record['company_name'],listing_year=pd.Timestamp(item.listing_date).year,
            offer_dates_verified=ipo_id in dates.index,source_status=record['status'],source_path=record.get('source_path'),
            document_date=record.get('document_date'),annual_eps_extracted='annual_diluted_eps' in record.get('facts',{}),
            early_book_eligible=ipo_id in early_map.index,exchange_open_record_present=ipo_id in outcomes.index,
            financial_asof=False,offer_asof=False,gmp_asof=False,all_facts_asof=False,
            outside_early_cohort_reason='No eligible penultimate book in frozen 2021–2024 protocol')
        if ipo_id in early_map.index:
            e=early_map.loc[ipo_id]
            available=quotes[quotes.ipo_id.eq(ipo_id)].copy()
            available['gmp_date']=pd.to_datetime(available.gmp_date,errors='coerce')
            prior=available[available.gmp_date.lt(e.decision_market_date)]
            gap_reason=('available' if pd.notna(e.gmp_pct) else
                'conflicting_eligible_quotes' if e.gmp_conflict_days else
                'only_stale_predecision_quotes' if len(prior) else
                'only_same_day_or_later_quotes' if len(available) else
                recovery_status.get(ipo_id,'no_verified_dated_history'))
            row.update(decision_date=str(e.decision_market_date.date()),financial_asof=bool(e.financial_observed),
                offer_asof=bool(e.offer_observed),gmp_asof=bool(pd.notna(e.gmp_pct)),gmp_date=e.gmp_date,
                gmp_source_url=e.gmp_source_url,gmp_conflict_days=int(e.gmp_conflict_days),
                financial_evidence_status=e.evidence_status,offer_evidence_status=e.offer_evidence_status,
                gmp_gap_reason=gap_reason,
                canonical_close_date=str(item.issue_close_date),verified_close_date=str(dates.loc[ipo_id].issue_close_date),
                original_gmp_covered=bool(e.original_gmp_covered),
                gmp_recovered=bool(pd.notna(e.gmp_pct) and not e.original_gmp_covered),
                all_facts_asof=bool(e.financial_observed and e.offer_observed and pd.notna(e.gmp_pct)),
                outside_early_cohort_reason='')
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT.with_name(OUT.stem+'_coverage.csv'),index=False)
    paired=early[early.gmp_pct.notna()]
    summary=dict(universe_n=len(frame),available_pdfs=int(frame.source_path.notna().sum()),
        eps_extracted=int(frame.annual_eps_extracted.sum()),early_book_n=len(early),
        original_paired_n=int(early.original_gmp_covered.sum()),recovered_paired_n=int((early.gmp_pct.notna()&~early.original_gmp_covered).sum()),
        paired_n=len(paired),remaining_gmp_gap_n=int(early.gmp_pct.isna().sum()),
        annual_eps_asof_n=int(early.financial_observed.sum()),offer_asof_n=int(early.offer_observed.sum()),
        all_facts_asof_n=int(frame.all_facts_asof.sum()),
        original_2024_n=int((early.year.eq(2024)&early.original_gmp_covered).sum()),
        expanded_2024_n=int((early.year.eq(2024)&early.gmp_pct.notna()).sum()),
        source_status_counts=frame.source_status.value_counts().to_dict(),
        financial_asof_exceptions=early.loc[~early.financial_observed,'evidence_status'].value_counts().to_dict(),
        offer_asof_exceptions=early.loc[~early.offer_observed,'offer_evidence_status'].value_counts().to_dict(),
        gmp_gap_reasons=frame.loc[frame.early_book_eligible&~frame.gmp_asof,'gmp_gap_reason'].value_counts().to_dict(),
        canonical_date_corrections=int(sum(ipo_id in dates.index and str(canonical.loc[ipo_id].issue_close_date)!=str(dates.loc[ipo_id].issue_close_date) for ipo_id in early.ipo_id)),
        gmp_recovery=recovery_audit,cover_date_repair=payload.get('cover_date_repair',{}))
    summary['by_listing_year']={str(year):dict(issuers=len(group),pdfs=int(group.source_path.notna().sum()),
        eps_extracted=int(group.annual_eps_extracted.sum()),early_books=int(group.early_book_eligible.sum()),
        gmp_asof=int(group.gmp_asof.sum()),financial_asof=int(group.financial_asof.sum()),offer_asof=int(group.offer_asof.sum()))
        for year,group in frame.groupby('listing_year')}
    return summary


def run():
    early,canonical,quotes=prepare()
    paired=early[early.gmp_pct.notna()].copy()
    folds=[];trace=[]
    for year in (2022,2023,2024):
        test=paired[paired.year.eq(year)].copy();first=test.decision_market_date.min()
        train=paired[paired.year.lt(year)&paired.listing_date.lt(first)]
        if set(train.ipo_id)&set(test.ipo_id):
            raise ValueError('Train/test overlap')
        for track,features in TRACKS.items():
            classifier=estimator(track,'direction').fit(train[features],train.target_binary)
            test['p_'+track]=classifier.predict_proba(test[features])[:,1]
            for kind in ('ridge','huber'):
                model=estimator(track,kind).fit(train[features],train.target_return-train.gmp_pct)
                test['r_'+track+'_'+kind]=np.maximum(-100,test.gmp_pct+model.predict(test[features]))
        folds.append(test)
        trace.append(dict(test_year=year,train_n=len(train),test_n=len(test),train_ids=sorted(map(int,train.ipo_id)),
            test_ids=sorted(map(int,test.ipo_id)),latest_training_outcome=str(train.listing_date.max().date()),
            first_test_decision=str(first.date())))
    ledger=pd.concat(folds,ignore_index=True)
    dev=ledger[ledger.year.isin([2022,2023])];confirmation=ledger[ledger.year.eq(2024)].copy()
    direction_dev={track:summarize(dev.target_binary,dev['p_'+track],.5) for track in TRACKS}
    combined=[t for t in TRACKS if t!='gmp_fitted']
    selected_direction=min(combined,key=lambda t:(direction_dev[t]['brier_score'],t))
    return_names=['r_'+track+'_'+kind for track in TRACKS for kind in ('ridge','huber')]
    return_dev={name:return_metrics(dev,name) for name in return_names}
    selected_return=min([name for name in return_names if 'gmp_fitted' not in name],key=lambda n:(return_dev[n]['mae_pp'],n))
    selected_gmp_calibration=min([name for name in return_names if 'gmp_fitted' in name],key=lambda n:(return_dev[n]['mae_pp'],n))
    subsets={}
    for subset,rows in [('original_primary',confirmation[confirmation.original_gmp_covered]),('expanded_secondary',confirmation)]:
        y=rows.target_binary.to_numpy();baseline=(rows.gmp_pct.to_numpy()>0)==y
        directions={}
        for track in TRACKS:
            p=rows['p_'+track].to_numpy()
            directions[track]=dict(metrics=summarize(y,p,.5),vs_gmp=paired_gain((p>=.5)==y,baseline))
        directions['gmp_sign']=dict(metrics=summarize(y,rows.gmp_pct.gt(0).astype(float),.5))
        returns={name:return_metrics(rows,name) for name in ['gmp_pct',*return_names]}
        ranking={name:rankings(rows,name) for name in ['gmp_pct',*return_names]}
        selected_predictions=rows[selected_return].to_numpy();selected_correct=rows['p_'+selected_direction].ge(.5).to_numpy()==y
        subsets[subset]=dict(n=len(rows),ipo_ids=sorted(map(int,rows.ipo_id)),direction=directions,returns=returns,ranking=ranking,
            selected_direction_error_ids=sorted(map(int,rows.loc[~selected_correct,'ipo_id'])),
            selected_return_largest_misses=rows.assign(return_prediction=selected_predictions,
                absolute_error=np.abs(rows.target_return-selected_predictions)).nlargest(8,'absolute_error')[
                ['ipo_id','company_name','gmp_pct','target_return','return_prediction','absolute_error']].to_dict('records'))
    summary=coverage(early,canonical,quotes)
    if subsets['original_primary']['n']!=39:
        raise ValueError('Original 39-case test denominator changed')
    report=dict(protocol_path='output/module1_completion_protocol_20261024.md',live_promoted=False,
        paper_changed=False,module2_changed=False,coverage=summary,trace=trace,
        selected_direction=selected_direction,selected_return=selected_return,selected_gmp_calibration=selected_gmp_calibration,
        direction_development=direction_dev,return_development=return_dev,results=subsets,
        limitations=['Retrospective, previously inspected 2024 outcomes; not blind or prospective proof.',
            'Recovered coverage is non-random. Original and expanded tests are reported separately.',
            'Dates on historical GMP/bid tables do not certify original intraday publication clocks.',
            'Sector labels are static canonical metadata without a verified publication-time history; sector-relative results are exploratory.',
            'Rankings compare only identical snapshot dates, not invented simultaneous access to different-day figures.',
            'Selection returns are gross conditional-on-allotment opening returns, not capital-weighted portfolio profits.',
            'This does not validate arbitrary intraday, first-day, or closing-hour use.'])
    paths=[FRAME,MANIFEST,RECOVERY,RECOVERY_AUDIT,ROOT/'data/audit/ipo_dated_fundamentals_all_20261024.json',
        ROOT/'data/canonical_listed_ipos.csv',ROOT/'output/module1_completion_protocol_20261024.md',
        ROOT/'ml/complete_module1_benchmark.py',ROOT/'ml/module1_gap_recovery.py',ROOT/'ml/model_bundle.json']
    report['input_sha256']={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    ledger.to_csv(OUT.with_name(OUT.stem+'_predictions.csv'),index=False)
    choice_rows=[]
    for subset,result in subsets.items():
        for name,item in result['ranking'].items():
            choice_rows += [dict(subset=subset,variant=name,**choice) for choice in item['choices']]
    pd.DataFrame(choice_rows).to_csv(OUT.with_name(OUT.stem+'_ranking_choices.csv'),index=False)
    # The primary report is a completed, bounded experiment, not a new model
    # auto-promotion. No fitted research artifacts replace production bundles.
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    render(report)
    print(json.dumps(dict(coverage={k:v for k,v in summary.items() if k!='gmp_recovery'},
        selected_direction=selected_direction,selected_return=selected_return,
        results={key:dict(n=value['n'],direction=value['direction'][selected_direction],
            returns=value['returns'][selected_return],ranking=value['ranking'][selected_return]) for key,value in subsets.items()}),indent=2))
    return report


def render(report):
    c=report['coverage'];sd=report['selected_direction'];sr=report['selected_return']
    lines=['# Module 1 completion: data coverage and GMP comparison','',
        'Module 1 only. No Module 2, selected-paper, saved-prediction or deployed-model changes.','',
        '## Data coverage','',
        f"Universe: {c['universe_n']} issuers; {c['available_pdfs']} existing PDFs; {c['eps_extracted']} extracted annual EPS tables. "
        f"The frozen early-book population has {c['early_book_n']} IPOs. Dated GMP coverage increased from "
        f"{c['original_paired_n']} to {c['paired_n']} ({c['recovered_paired_n']} recovered); {c['remaining_gmp_gap_n']} remain unresolved.",
        '',f"At decision time: {c['annual_eps_asof_n']} usable earnings records; {c['offer_asof_n']} usable OFS ratios; "
        f"{c['all_facts_asof_n']} IPOs have all three (earnings, OFS, GMP). "
        f"{c['canonical_date_corrections']} canonical closing dates disagree with the verified issue calendar; the verified dates are used.",
        '', 'The coverage CSV accounts for every issuer, including source absence, later document dates, conflicting offer amounts and uncovered bids. '
        'Missing data is not converted into an observed zero. Sector-relative financial features use retrospective sector metadata and are not publication-time certified.',
        '', '## Fixed experiment and selections','',
        'Earlier-year matured labels only; 2022/2023 predictions select models. All candidates and all thresholds were fixed before this run. '
        '2024 outcomes have been inspected before: these results are exploratory, not a fresh blind test.',
        '',f"Development-selected direction model: **{sd}**. Return model: **{sr}**. "
        f"GMP-only fitted return control: **{report['selected_gmp_calibration']}**.",
        '', '## Direction, return estimates and ranking','',
        '| Test population | IPOs | GMP-sign accuracy | Combined direction accuracy | GMP return MAE | Combined return MAE | Same-date ranking panels | Selection difference |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,result in report['results'].items():
        direction=result['direction'];returns=result['returns'];ranking=result['ranking'][sr]
        delta=ranking['paired_selection_difference_pp']['mean']
        lines.append(f"| {name} | {result['n']} | {direction['gmp_sign']['metrics']['classifier_accuracy_pct']:.1f}% | "
            f"{direction[sd]['metrics']['classifier_accuracy_pct']:.1f}% | {returns['gmp_pct']['mae_pp']:.2f} pp | "
            f"{returns[sr]['mae_pp']:.2f} pp | {ranking['panels']} | {delta:+.2f} pp |" if delta is not None else
            f"| {name} | {result['n']} | See JSON | See JSON | See JSON | See JSON | 0 | Not estimable |")
    lines+=['','MAE is average absolute return error in percentage points; lower is better. Ranking compares the highest positive central estimate '
        'with the highest positive GMP on exactly the same available IPOs, or skips all when no positive estimate exists. '
        'No feature-range filtering removes difficult cases. Ties use issuer ID, never realized results.','', '## Detailed paired results']
    lines+=['', '### All fixed input combinations','',
        'For each input track below, the return algorithm is selected on development MAE only. These are not test-selected winners.',
        '', '| Test population | Input track | Direction accuracy | Balanced accuracy | AUC | Return algorithm | Return MAE |',
        '|---|---|---:|---:|---:|---|---:|']
    for population,result in report['results'].items():
        for track in TRACKS:
            name=min(['r_'+track+'_'+kind for kind in ('ridge','huber')],
                key=lambda n:(report['return_development'][n]['mae_pp'],n))
            d=result['direction'][track]['metrics']
            lines.append(f"| {population} | {track} | {d['classifier_accuracy_pct']:.1f}% | "
                f"{d['balanced_accuracy_pct']:.1f}% | {d['roc_auc']:.3f} | {name.rsplit('_',1)[-1]} | "
                f"{result['returns'][name]['mae_pp']:.2f} pp |")
    for name,result in report['results'].items():
        d=result['direction'][sd];m=result['returns'][sr];rank=result['ranking'][sr]
        ci=d['vs_gmp']['bootstrap_95_pp'];rci=m['mae_improvement_vs_gmp']['bootstrap_95'];rd=rank['paired_selection_difference_pp']
        lines+=['',f'### {name}','',
            f"Direction difference versus GMP-sign: {d['vs_gmp']['change_pp']:+.2f} points "
            f"(paired bootstrap 95% interval {ci[0]:+.2f} to {ci[1]:+.2f}); "
            f"{d['vs_gmp']['corrected_errors']} GMP mistakes corrected and {d['vs_gmp']['introduced_errors']} new mistakes introduced.",
            '',f"Return-MAE improvement versus GMP: {m['mae_improvement_vs_gmp']['mean']:+.2f} pp "
            f"(95% interval {rci[0]:+.2f} to {rci[1]:+.2f}). "
            f"GMP-only calibrated control MAE: {result['returns'][report['selected_gmp_calibration']]['mae_pp']:.2f} pp.",
            '',f"Ranking: {rank['panels']} same-date choice sets, {rank['pair_count']} non-tied IPO pairs. "
            f"Reliable ranking advantage established: **{'yes' if rank['established_edge'] else 'no'}**.",
            '',f"Selection difference, model minus GMP: {rd['mean']:+.2f} pp "
            f"(95% interval {rd['bootstrap_95'][0]:+.2f} to {rd['bootstrap_95'][1]:+.2f}). "
            f"Pairwise ordering: model {rank['pairwise_accuracy_pct']:.1f}% versus GMP {rank['gmp_pairwise_accuracy_pct']:.1f}%.",
            '', '| Date | Eligible IPOs | Combined choice | GMP choice | Combined actual opening return | GMP actual opening return |', '|---|---:|---|---|---:|---:|']
        for item in rank['choices']:
            lines.append(f"| {item['decision_date']} | {item['n']} | {item['model_choice']} | {item['gmp_choice']} | "
                f"{item['model_opening_return_pct']:+.2f}% | {item['gmp_opening_return_pct']:+.2f}% |")
    lines+=['','## Remaining gaps and decision','',
        f"{c['remaining_gmp_gap_n']} early-book issuers still lack a verified, suitably dated GMP quote after the bounded recovery. "
        'Later/undated figures are not a repair. Missing earlier RHP versions and bid histories outside the frozen cohort remain source gaps, '
        'not values that can be inferred from listing outcomes.','',
        f"Remaining GMP gap reasons: {json.dumps(c['gmp_gap_reasons'],sort_keys=True)}. "
        f"The native-cover date audit corrected {c['cover_date_repair'].get('changed_records',0)} document dates; "
        'incorporation dates are no longer accepted as prospectus dates.', '',
        'No automatic production promotion. Improvement is judged against the baselines and uncertainty above, not a requested headline accuracy. '
        'A negative result completes this comparison; it does not justify tuning the inspected test. '
        'The prediction and ranking ledgers retain every evaluated case.','',*['- '+text for text in report['limitations']]]
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    coverage_lines=['# Module 1 data coverage and remaining exceptions','',
        '## What is covered','',
        'Issuer-universe coverage and usable pre-decision feature coverage are different. '
        'The frozen modeling protocol is 2021–2024 penultimate-session books; 2025–2026 rows are inventoried but not quietly substituted into this experiment.',
        '', '| Listing year | Issuers | Existing PDFs | Extracted annual EPS | Eligible early books | Dated GMP at decision | Earnings at decision | OFS at decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for year,item in c['by_listing_year'].items():
        coverage_lines.append(f"| {year} | {item['issuers']} | {item['pdfs']} | {item['eps_extracted']} | {item['early_books']} | "
            f"{item['gmp_asof']} | {item['financial_asof']} | {item['offer_asof']} |")
    coverage_lines+=['', '## Repairs completed','',
        f"- Retried all {c['gmp_recovery']['attempted']} missing-GMP early-book issuers; found {c['gmp_recovery']['matched']} dated histories. "
        f"Only {c['recovered_paired_n']} have a quote strictly before the decision and no more than seven days old. "
        f"Usable GMP coverage: {c['original_paired_n']} to {c['paired_n']}; 2024: {c['original_2024_n']} to {c['expanded_2024_n']}.",
        f"- Corrected {c['cover_date_repair'].get('changed_records',0)} cover dates using the prospectus title block, not incorporation narrative. "
        'Only changed records were re-extracted; this was native text, not OCR.',
        '- Preserved the prior offer-placeholder fix and amount/share-count conflict guard. Conflicting ratios are withheld.',
        f"- Used verified exchange offer dates instead of {c['canonical_date_corrections']} legacy fiscal-date placeholders in this early-book population. "
        'The canonical database, deployed model and saved predictions were not rewritten.',
        '', '## Gaps that remain','',
        f"- GMP: {c['remaining_gmp_gap_n']} unresolved. Reasons: {json.dumps(c['gmp_gap_reasons'],sort_keys=True)}.",
        f"- Earnings at decision: {c['early_book_n']-c['annual_eps_asof_n']} unavailable. "
        f"Reasons: {json.dumps(c['financial_asof_exceptions'],sort_keys=True)}. "
        'A later final Prospectus cannot be backdated into an earlier bidding decision; its earlier RHP is needed.',
        f"- OFS at decision: {c['early_book_n']-c['offer_asof_n']} unavailable. Reasons: {json.dumps(c['offer_asof_exceptions'],sort_keys=True)}.",
        f"- {c['source_status_counts'].get('missing_or_ambiguous_pdf',0)} universe issuers lack a uniquely matched local PDF. "
        'They remain in the coverage inventory; no data was fabricated for them.',
        '- Daily GMP and bid histories were recovered retrospectively. A displayed historical date does not prove the original intraday publication time.',
        '- Sector metadata is retrospectively recorded; sector-relative model results are exploratory.',
        '', '## Deliverables','',
        '- module1_completion_20261024_coverage.csv: every issuer, source/date status and explicit gap reason.',
        '- module1_completion_20261024.md: paired direction, return and ranking comparison.',
        '- module1_completion_20261024_predictions.csv: every evaluated IPO and every fixed model prediction.',
        '- module1_completion_20261024_ranking_choices.csv: exact eligible sets and paired choices.',
        '', 'No Module 2 or conference-paper changes. Remaining source gaps are not claimed solved.']
    (ROOT/'output/module1_data_coverage_20261024.md').write_text('\n'.join(coverage_lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    run()
