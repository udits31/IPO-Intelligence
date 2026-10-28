"""Stage-matched three-bid challenger with strictly dated RHP feature joins.

Same earlier-year folds, fixed logistic family/threshold, paired test IPOs.
Artifacts are research-only: no unsupported stage is automatically routed here.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.research_module1_stage_transfer import datasets
from ml.model_pipeline import FEATURE_COLS,fit_model,score_model,summarize,prepare_dataset
from ml.research_monitor_daily_learning import paired_gain
from services.dated_ipo_fundamentals import load_registry,at_decision,REGISTRY

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/module1_stage_fundamental_challenger_all_20261024.json'
ARTIFACT=ROOT/'ml/artifacts/module1_stage_fundamental_challenger_all_20261024.joblib'
EXTRA=['log_offer_cap_price','log_issue_size','earnings_yield_pct','eps_growth_asinh','loss_making','ofs_ratio']


def attach(frame,registry,canonical,stage,offer_dates=None):
    frame=frame.copy()
    if stage not in ('penultimate_session','final_book'):
        raise ValueError('Untrained bid stage')
    rows=[]
    for row in frame.itertuples():
        ipo_id=int(row.ipo_id)
        item=canonical.loc[ipo_id]
        cap=pd.to_numeric(item.price_band_upper,errors='coerce')
        issue_size=pd.to_numeric(item.issue_size_cr,errors='coerce')
        # Canonical offer dates contain legacy fiscal-year placeholders. Use
        # NSE's identity-checked issue archive, never March-31 placeholders.
        clock=(row.decision_market_date if stage=='penultimate_session' else
               offer_dates.loc[ipo_id].issue_close_date if offer_dates is not None and ipo_id in offer_dates.index else None)
        features=at_decision(registry.get(ipo_id),clock,cap)
        features['log_offer_cap_price']=float(np.log1p(cap)) if pd.notna(cap) and cap>0 else None
        features['log_issue_size']=float(np.log1p(issue_size)) if pd.notna(issue_size) and issue_size>0 else None
        features['fundamental_decision_date']=clock
        rows.append(features)
    joined=pd.concat([frame.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    joined['bid_training_stage']=stage
    return joined


def fit(train,cols):
    model=Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('scale',StandardScaler()),('classifier',LogisticRegression(C=.5,max_iter=2000,random_state=42))])
    model.fit(train[cols],train.target_binary)
    return model


def run():
    live=ROOT/'ml/model_bundle.json';live_hash=hashlib.sha256(live.read_bytes()).hexdigest()
    final,early,coverage=datasets()
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    offer_dates=pd.read_csv(ROOT/'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    registry=load_registry()
    original_labels=prepare_dataset().set_index('id').target_binary
    early=attach(early,registry,canonical,'penultimate_session')
    final=attach(final,registry,canonical,'final_book',offer_dates)
    predictions=[];trace=[]
    for year in (2022,2023,2024):
        test=early[early.year.eq(year)].copy()
        cutoff=pd.to_datetime(test.decision_market_date).min()
        # Labels must have occurred before the first test decision.
        train_early=early[early.year.lt(year)&early.listing_date.lt(cutoff)]
        train_final=final[final.year.lt(year)&final.listing_date.lt(cutoff)]
        if set(train_early.ipo_id)&set(test.ipo_id):
            raise ValueError('Training and test issuers overlap')
        scaler,legacy=fit_model(train_final)
        test['legacy_final_fit_on_interim']=score_model(scaler,legacy,test)
        original_train=train_final.copy()
        original_train['target_binary']=original_labels.loc[original_train.ipo_id].to_numpy()
        original_scaler,original_model=fit_model(original_train)
        test['original_label_pipeline_on_interim']=score_model(original_scaler,original_model,test)
        matched_final=train_final[train_final.ipo_id.isin(train_early.ipo_id)]
        if set(matched_final.ipo_id)!=set(train_early.ipo_id):
            raise ValueError('Matched-stage training control must use identical issuer identities')
        matched_scaler,matched_model=fit_model(matched_final)
        test['matched_id_final_fit_on_interim']=score_model(matched_scaler,matched_model,test)
        for name,cols in [('stage_bids_only',FEATURE_COLS),('stage_plus_offer_terms',FEATURE_COLS+EXTRA[:2]),
                          ('stage_plus_dated_fundamentals',FEATURE_COLS+EXTRA)]:
            model=fit(train_early,cols)
            test[name]=model.predict_proba(test[cols])[:,1]
        predictions.append(test)
        trace.append(dict(test_year=year,train_early_n=len(train_early),train_final_n=len(train_final),
            train_latest_outcome=train_early.listing_date.max().date().isoformat(),
            test_first_decision=cutoff.date().isoformat(),
            test_n=len(test),test_financial_observed=int(test.financial_observed.sum()),
            test_offer_observed=int(test.offer_observed.sum())))
    ledger=pd.concat(predictions)
    confirm=ledger[ledger.year.eq(2024)]
    previous_path=ROOT/'output/module1_stage_fundamental_challenger_20261023_predictions.csv'
    if previous_path.exists():
        previous=pd.read_csv(previous_path)
        previous=previous[previous.year.eq(2024)].set_index('ipo_id')
        if set(confirm.ipo_id)!=set(previous.index) or not np.array_equal(
                confirm.target_binary.to_numpy(),previous.loc[confirm.ipo_id].target_binary.to_numpy()):
            raise ValueError('Extraction improvements cannot change the original test IPOs or outcomes')
    y=confirm.target_binary.to_numpy()
    results={}
    for name in ('original_label_pipeline_on_interim','legacy_final_fit_on_interim','matched_id_final_fit_on_interim','stage_bids_only','stage_plus_offer_terms','stage_plus_dated_fundamentals'):
        p=confirm[name].to_numpy()
        results[name]=dict(metrics=summarize(y,p,.5),
            vs_original_pipeline=paired_gain((p>=.5)==y,(confirm.original_label_pipeline_on_interim.to_numpy()>=.5)==y),
            vs_legacy=paired_gain((p>=.5)==y,(confirm.legacy_final_fit_on_interim.to_numpy()>=.5)==y),
            vs_stage_bids=paired_gain((p>=.5)==y,(confirm.stage_bids_only.to_numpy()>=.5)==y),
            vs_always_positive=paired_gain((p>=.5)==y,y.astype(bool)))
    paired_gmp=confirm[confirm.gmp_pct.notna()]
    gmp_result=None
    if len(paired_gmp):
        gmp_y=paired_gmp.target_binary.to_numpy()
        gmp_calls=paired_gmp.gmp_pct.gt(0).to_numpy()
        gmp_result=dict(n=len(paired_gmp),gmp_rule='Positive if prior dated GMP is above zero; no magnitude forecast',
            gmp_accuracy_pct=float((gmp_calls==gmp_y).mean()*100),
            model_accuracy_pct=float(((paired_gmp.stage_plus_dated_fundamentals.to_numpy()>=.5)==gmp_y).mean()*100),
            paired_change=paired_gain((paired_gmp.stage_plus_dated_fundamentals.to_numpy()>=.5)==gmp_y,gmp_calls==gmp_y))
        gmp_result['tracks']={name:dict(
            accuracy_pct=float(((paired_gmp[name].to_numpy()>=.5)==gmp_y).mean()*100),
            vs_gmp=paired_gain((paired_gmp[name].to_numpy()>=.5)==gmp_y,gmp_calls==gmp_y))
            for name in results}
    # Save trained stage-specific research artifacts. No final coefficients are
    # used for interim inference, and no early coefficients for final inference.
    tracks={}
    for stage,train in [('penultimate_session',early),('final_book',final)]:
        tracks[stage]=dict(model=fit(train,FEATURE_COLS+EXTRA),features=FEATURE_COLS+EXTRA,
            training_n=len(train),training_ipo_ids=sorted(map(int,train.ipo_id)),
            latest_training_outcome=train.listing_date.max().date().isoformat())
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump(dict(schema_version=1,research_only=True,tracks=tracks),ARTIFACT)
    report=dict(protocol='2022/2023 development folds; unchanged 2024 confirmation IPOs. Fixed C=0.5 and direction cutoff=0.5; no test-driven tuning.',
        live_promoted=False,synthetic_rows=0,confirmation_n=len(confirm),results=results,trace=trace,
        confirmation_ids_sha256=hashlib.sha256(','.join(map(str,sorted(confirm.ipo_id))).encode()).hexdigest(),
        paired_gmp=gmp_result,
        coverage=dict(early_n=len(early),registry_issuers=len(registry),
            early_financial_observed=int(early.financial_observed.sum()),
            early_offer_observed=int(early.offer_observed.sum()),
            final_n=len(final),final_financial_observed=int(final.financial_observed.sum()),
            final_offer_observed=int(final.offer_observed.sum()),
            confirmation_financial_observed=int(confirm.financial_observed.sum()),
            confirmation_offer_observed=int(confirm.offer_observed.sum()),cohort_audit=coverage),
        artifact_path=ARTIFACT.relative_to(ROOT).as_posix(),artifact_sha256=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        limitations=['RHP dates and retrospective daywise tables do not certify original publication-time availability.',
            'Missing fundamentals use training-only median imputation and missingness indicators, never observed zero.',
            'All existing cohort documents were attempted; extraction and pre-decision coverage are reported separately.',
            'Penultimate-session training does not validate first-day, closing-hour or arbitrary intraday forecasts.',
            '2024 outcomes have already been inspected in other research; not blind or prospective confirmation.',
            'Offer cap price and issue size come from canonical offer terms, not post-listing prices.'],
        input_sha256={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [ROOT/'data/canonical_listed_ipos.csv',ROOT/'data/subscription_observations.csv',
             ROOT/'data/daywise_subscription_research_cohort.csv',REGISTRY,Path(__file__),
             ROOT/'services/dated_ipo_fundamentals.py',ROOT/'ml/extract_dated_ipo_fundamentals.py',
             ROOT/'ml/extract_all_ipo_fundamentals.py',ROOT/'data/nse_verified_offer_dates.csv']})
    ledger.to_csv(OUT.with_name(OUT.stem+'_predictions.csv'),index=False)
    source_payload=json.loads(REGISTRY.read_text(encoding='utf-8'))
    coverage_rows=[]
    early_map=early.set_index('ipo_id')
    for source in source_payload['records']:
        row=dict(ipo_id=source['ipo_id'],company_name=source.get('company_name'),source_status=source['status'],
            document_date=source.get('document_date'),eps_extracted='annual_diluted_eps' in source.get('facts',{}),
            source_path=source.get('source_path'),used_in_early_cohort=source['ipo_id'] in early_map.index)
        if source['ipo_id'] in early_map.index:
            observed=early_map.loc[source['ipo_id']]
            row.update(decision_date=observed.decision_market_date,
                financial_joined=bool(observed.financial_observed),offer_joined=bool(observed.offer_observed),
                asof_status=observed.evidence_status)
        coverage_rows.append(row)
    pd.DataFrame(coverage_rows).to_csv(OUT.with_name(OUT.stem+'_coverage.csv'),index=False)
    report['document_coverage']=dict(expected=source_payload['expected_issuer_count'],
        processed=source_payload['processed_count'],
        available_pdfs=sum(bool(row.get('source_path')) for row in source_payload['records']),
        extracted_eps=sum('annual_diluted_eps' in row.get('facts',{}) for row in source_payload['records']),
        source_status_counts=pd.Series([row['status'] for row in source_payload['records']]).value_counts().to_dict())
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    labels=dict(original_label_pipeline_on_interim='Current-model method, retrained on earlier years only',
        legacy_final_fit_on_interim='Final-book training, corrected listing-open labels',
        matched_id_final_fit_on_interim='Final-book training, same training issuers as interim model',
        stage_bids_only='Interim-book training, bids only',
        stage_plus_offer_terms='Interim bids + offer price and issue size',
        stage_plus_dated_fundamentals='Interim bids + offer terms + dated earnings, valuation and OFS')
    lines=['# Module 1: do extracted company facts improve prediction?','',report['protocol'],'',
        '## Same-test comparison','',
        'Every row below predicts positive versus flat/negative listing open on the same 75 IPOs. '
        'The test uses penultimate-session bids, not final bids or arbitrary intraday readings. '
        'The current-model baseline is refitted using its original method and labels on earlier years only; '
        'using today\'s deployed bundle on these 2024 IPOs would test on its own training records.','',
        '| Model | Correct / 75 | Accuracy | Balanced accuracy | AUC |',
        '|---|---:|---:|---:|---:|']
    lines += [f"| {labels[name]} | {item['metrics']['tn']+item['metrics']['tp']}/75 | "
        f"{item['metrics']['classifier_accuracy_pct']:.1f}% | {item['metrics']['balanced_accuracy_pct']:.1f}% | "
        f"{item['metrics']['roc_auc']:.3f} |" for name,item in results.items()]
    lines += ['',f"Confirmation: {len(confirm)} IPOs; {int(confirm.financial_observed.sum())} have dated annual EPS facts; {int(confirm.offer_observed.sum())} have a dated OFS ratio (pure or mixed offer).",
        'All cases remain in every comparison. Missing facts are not a reason to remove a losing IPO.',
        '',f"Always predicting positive gets {int(y.sum())}/{len(y)} correct ({y.mean()*100:.1f}%). "
        'Balanced accuracy gives equal weight to positive and nonpositive IPOs; ordinary accuracy is inflated by the positive majority.',
        '', '## What changed and what it means','',
        f"Offer-price/size model accuracy: {results['stage_plus_offer_terms']['metrics']['classifier_accuracy_pct']:.1f}%; "
        f"full-fundamentals accuracy: {results['stage_plus_dated_fundamentals']['metrics']['classifier_accuracy_pct']:.1f}%. "
        'Paired differences below distinguish stage matching from adding company facts. '
        'These retrospective scores do not establish general live accuracy or validate return-magnitude ranking.','']
    for name in ('stage_bids_only','stage_plus_offer_terms','stage_plus_dated_fundamentals'):
        delta=results[name]['vs_original_pipeline']; interval=delta['bootstrap_95_pp']
        lines.append(f"- {labels[name]}: {delta['change_pp']:+.2f} percentage points versus the current-model method "
            f"(paired bootstrap 95% interval {interval[0]:+.2f} to {interval[1]:+.2f}).")
    docs=report['document_coverage']
    lines += ['', '## Source coverage','',
        f"All {docs['processed']} issuer records were accounted for; all {docs['available_pdfs']} existing PDFs were attempted. "
        f"Annual diluted EPS was extracted for {docs['extracted_eps']} documents. "
        f"{docs['source_status_counts'].get('missing_or_ambiguous_pdf',0)} issuer records have no uniquely matched local PDF. "
        'A source dated after the decision is not backdated. Full issuer-level extraction and as-of exceptions are in the coverage CSV.']
    if gmp_result:
        lines += ['', '## GMP-only comparison (separate matched subset)','',
            f"Only {gmp_result['n']} of the 75 test IPOs have the required dated GMP observation. "
            f"The GMP-above-zero rule gets {gmp_result['gmp_accuracy_pct']:.1f}% on these same {gmp_result['n']} IPOs.",
            '', '| Model on matched GMP subset | Accuracy | Change versus GMP |', '|---|---:|---:|']
        for name in ('original_label_pipeline_on_interim','stage_bids_only','stage_plus_offer_terms','stage_plus_dated_fundamentals'):
            item=gmp_result['tracks'][name]
            lines.append(f"| {labels[name]} | {item['accuracy_pct']:.1f}% | {item['vs_gmp']['change_pp']:+.2f} points |")
        lines += ['', 'This subset must not be compared directly with the full-cohort accuracy above.']
    lines += ['', '## Deployment and limitations',
        '', 'Separate penultimate-session and final-book artifacts were saved as research-only. The live model and old saved predictions were not replaced.',
        '',*['- '+value for value in report['limitations']]]
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if hashlib.sha256(live.read_bytes()).hexdigest()!=live_hash:
        raise ValueError('Challenger changed the live model')
    print(json.dumps(dict(results=results,coverage=report['coverage'],trace=trace),indent=2))
    return report


if __name__=='__main__':
    run()
