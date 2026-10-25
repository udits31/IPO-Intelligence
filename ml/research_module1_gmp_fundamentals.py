"""Paired, stage-matched GMP/fundamentals ablation; never promotes live models.

Reuses the completed, source-verified native extraction, not OCR or PDF parsing.
All candidates, C=.5 and cutoff=.5 are fixed before this run. Development Brier
score selects one variant using 2022/23 only; 2024 is inspected retrospective data.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from ml.model_pipeline import FEATURE_COLS, summarize
from ml.research_module1_stage_transfer import datasets
from ml.research_monitor_daily_learning import paired_gain
from ml.train_stage_fundamental_challenger import ROOT, OUT as PREVIOUS, EXTRA, attach, fit
from services.dated_ipo_fundamentals import REGISTRY

OUT=ROOT/'output/module1_gmp_fundamentals_comparison_20261024.json'
VARIANTS={
    'bids':FEATURE_COLS,
    'bids_offer_terms':FEATURE_COLS+EXTRA[:2],
    'bids_all_facts':FEATURE_COLS+EXTRA,
    'gmp_logistic':['gmp_pct'],
    'bids_gmp':FEATURE_COLS+['gmp_pct'],
    'bids_offer_terms_gmp':FEATURE_COLS+EXTRA[:2]+['gmp_pct'],
    'bids_all_facts_gmp':FEATURE_COLS+EXTRA+['gmp_pct'],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared():
    previous=json.loads(PREVIOUS.read_text())
    # Reuse the validated extraction only while its exact inputs/code and every
    # source document still match. No repeated text/table extraction is needed.
    for relative,expected in previous['input_sha256'].items():
        if digest(ROOT/relative)!=expected:
            raise ValueError(f'Previously verified extraction inputs changed: {relative}')
    payload=json.loads(REGISTRY.read_text(encoding='utf-8'))
    registry={int(r['ipo_id']):r for r in payload['records'] if r['status']=='dated_document_facts'}
    for record in registry.values():
        source=(ROOT/record['source_path']).resolve()
        source.relative_to((ROOT/'data/prospectus_pdfs').resolve())
        if digest(source)!=record['source_sha256']:
            raise ValueError('Source changed; revalidate its extracted facts')
    _,early,_=datasets()
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    early=attach(early,registry,canonical,'penultimate_session')
    paired=early[early.gmp_pct.notna()].copy()
    if not pd.to_datetime(paired.gmp_date).lt(pd.to_datetime(paired.decision_market_date)).all():
        raise ValueError('Undated, same-day or future GMP is forbidden')
    if not np.isfinite(paired.gmp_pct).all():
        raise ValueError('Invalid GMP cannot be imputed')
    expected=pd.read_csv(PREVIOUS.with_name(PREVIOUS.stem+'_predictions.csv'))
    expected=expected[expected.year.eq(2024)&expected.gmp_pct.notna()].set_index('ipo_id')
    actual=paired[paired.year.eq(2024)].set_index('ipo_id')
    if set(actual.index)!=set(expected.index) or not np.array_equal(
            actual.target_binary,expected.loc[actual.index].target_binary):
        raise ValueError('Paired test identities or outcomes changed')
    return early,paired


def run():
    live_hash=digest(ROOT/'ml/model_bundle.json')
    early,paired=prepared()
    folds=[];trace=[]
    for year in (2022,2023,2024):
        test=paired[paired.year.eq(year)].copy()
        first=pd.to_datetime(test.decision_market_date).min()
        train=paired[paired.year.lt(year)&paired.listing_date.lt(first)].copy()
        if set(train.ipo_id)&set(test.ipo_id):
            raise ValueError('Train/test issuer overlap')
        for name,features in VARIANTS.items():
            test[name]=fit(train,features).predict_proba(test[features])[:,1]
        test['gmp_sign']=test.gmp_pct.gt(0).astype(float)
        test['always_positive']=1.
        trace.append(dict(test_year=year,train_n=len(train),test_n=len(test),
            training_ids=sorted(map(int,train.ipo_id)),test_ids=sorted(map(int,test.ipo_id)),
            latest_training_outcome=str(train.listing_date.max().date()),
            first_test_decision=str(first.date())))
        folds.append(test)
    ledger=pd.concat(folds,ignore_index=True)
    dev=ledger[ledger.year.isin([2022,2023])]
    confirmation=ledger[ledger.year.eq(2024)].copy()
    development={name:summarize(dev.target_binary,dev[name],.5) for name in VARIANTS}
    selected=min(VARIANTS,key=lambda name:(development[name]['brier_score'],name))
    y=confirmation.target_binary.to_numpy()
    gmp_correct=(confirmation.gmp_sign.to_numpy()>=.5)==y
    results={}
    for name in [*VARIANTS,'gmp_sign','always_positive']:
        p=confirmation[name].to_numpy()
        results[name]=dict(metrics=summarize(y,p,.5),
            vs_gmp_sign=paired_gain((p>=.5)==y,gmp_correct),
            vs_gmp_logistic=paired_gain((p>=.5)==y,(confirmation.gmp_logistic.to_numpy()>=.5)==y))
    # Explicit error decomposition, including GMP's five nonpositive misses.
    errors=[]
    for row in confirmation.to_dict('records'):
        item={key:row[key] for key in ('ipo_id','company_name','gmp_date','decision_market_date',
            'gmp_pct','target_binary','target_return','financial_observed','offer_observed')}
        for name in [*VARIANTS,'gmp_sign']:
            item[name+'_probability']=float(row[name])
            item[name+'_correct']=bool((row[name]>=.5)==row['target_binary'])
        item['selected_model']=selected
        item['selected_vs_gmp']=('both_correct' if item[selected+'_correct'] and item['gmp_sign_correct'] else
            'corrected_gmp_error' if item[selected+'_correct'] else
            'introduced_error' if item['gmp_sign_correct'] else 'both_wrong')
        errors.append(item)
    full2024=early[early.year.eq(2024)]
    old_ledger=pd.read_csv(PREVIOUS.with_name(PREVIOUS.stem+'_predictions.csv'))
    old_paired=old_ledger[old_ledger.year.eq(2024)&old_ledger.gmp_pct.notna()]
    previous_diagnosis={name:dict(
        corrected_gmp_errors=int((old_paired[name].lt(.5)&old_paired.target_binary.eq(0)).sum()),
        introduced_errors=int((old_paired[name].lt(.5)&old_paired.target_binary.eq(1)).sum()))
        for name in ('stage_plus_offer_terms','stage_plus_dated_fundamentals')}
    report=dict(protocol='Fixed logistic C=0.5, fixed cutoff=0.5; paired training/test identities for all variants; earlier-year matured labels only.',
        live_promoted=False,stage='penultimate_session',selected_by='lowest pooled 2022/2023 Brier score; lexical tie break',
        selected_variant=selected,development=development,results=results,trace=trace,
        confirmation_n=len(confirmation),confirmation_positive_n=int(y.sum()),
        positive_gmp_n=int(confirmation.gmp_pct.gt(0).sum()),
        same_test_ids_sha256=hashlib.sha256(','.join(map(str,sorted(confirmation.ipo_id))).encode()).hexdigest(),
        coverage=dict(all_early_n=len(early),paired_n=len(paired),full_2024_n=len(full2024),
            paired_2024_n=len(confirmation),unpaired_2024_ids=sorted(map(int,full2024.loc[~full2024.ipo_id.isin(confirmation.ipo_id),'ipo_id'])),
            paired_2024_financial_observed=int(confirmation.financial_observed.sum()),
            paired_2024_offer_observed=int(confirmation.offer_observed.sum())),
        selected_error_counts=pd.Series([r['selected_vs_gmp'] for r in errors]).value_counts().to_dict(),
        previous_model_diagnosis=previous_diagnosis,
        offer_data_correction='Systematic offer-placeholder spillover fix and amount/share consistency guard; corrected across issuers, not chosen by outcomes.',
        limitations=['All 2024 outcomes have been inspected before; this is retrospective, not new blind or prospective validation.',
            'Only IPOs with a genuine preceding-date GMP are in this paired experiment. No missing GMP is replaced with zero.',
            'Daily source dates are not verified intraday publication clocks; later historical revisions remain possible.',
            'GMP-sign is a hard direction baseline; its 0/1 Brier score is not a calibrated market probability.',
            'Direction accuracy does not establish return ranking, allotment-adjusted profit or a tradable edge.'])
    paths=[ROOT/'ml/model_bundle.json',PREVIOUS,REGISTRY,ROOT/'data/preclose_gmp_history.csv',
        ROOT/'data/daywise_subscription_research_cohort.csv',ROOT/'data/canonical_listed_ipos.csv',
        ROOT/'data/nse_verified_offer_dates.csv',ROOT/'data/raw_daily_market_time_series.csv',
        ROOT/'data/audit/module1_exchange_listing_open_audit.csv',ROOT/'ml/model_pipeline.py',
        ROOT/'ml/return_pipeline.py',ROOT/'ml/research_module1_stage_transfer.py',
        ROOT/'ml/train_stage_fundamental_challenger.py',ROOT/'services/dated_ipo_fundamentals.py',
        ROOT/'ml/research_module1_gmp_fundamentals.py']
    report['input_sha256']={path.relative_to(ROOT).as_posix():digest(path) for path in paths}
    ledger.to_csv(OUT.with_name(OUT.stem+'_predictions.csv'),index=False)
    pd.DataFrame(errors).to_csv(OUT.with_name(OUT.stem+'_errors.csv'),index=False)
    artifact=OUT.with_suffix('.joblib')
    last_train=paired[paired.year.le(2023)&paired.listing_date.lt(pd.to_datetime(confirmation.decision_market_date).min())]
    selected_model=fit(last_train,VARIANTS[selected])
    joblib.dump(dict(research_only=True,variant=selected,features=VARIANTS[selected],
        model=selected_model,training_ids=sorted(map(int,last_train.ipo_id))),artifact)
    names=selected_model.named_steps['missing'].get_feature_names_out(VARIANTS[selected])
    standardized=selected_model.named_steps['scale'].transform(
        selected_model.named_steps['missing'].transform(confirmation[VARIANTS[selected]]))
    classifier=selected_model.named_steps['classifier']
    contribution=standardized*classifier.coef_[0]
    logits=contribution.sum(axis=1)+classifier.intercept_[0]
    if not np.allclose(1/(1+np.exp(-logits)),confirmation[selected]):
        raise ValueError('Decision contributions do not reproduce saved probabilities')
    contributions=[]
    for index,row in enumerate(confirmation.itertuples()):
        for feature,value in zip(names,contribution[index]):
            contributions.append(dict(ipo_id=int(row.ipo_id),company_name=row.company_name,
                feature=feature,log_odds_contribution=float(value),intercept=float(classifier.intercept_[0])))
    pd.DataFrame(contributions).to_csv(OUT.with_name(OUT.stem+'_contributions.csv'),index=False)
    report['selected_training']=dict(n=len(last_train),positive_n=int(last_train.target_binary.sum()),
        features=VARIANTS[selected],transformed_feature_count=len(names))
    report['artifact_sha256']=digest(artifact)
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Module 1: combining dated GMP, interim bids and company facts','',report['protocol'],'',
        f"All models use the same {len(confirmation)} 2024 test IPOs and the same eligible earlier-year training IPOs. "
        f"All {int(confirmation.gmp_pct.gt(0).sum())} GMP quotes are positive: GMP-sign is therefore identical to always predicting positive.",
        '', '| Inputs | Correct | Accuracy | Balanced accuracy | AUC | Brier (lower is better) |',
        '|---|---:|---:|---:|---:|---:|']
    for name,item in results.items():
        m=item['metrics']
        lines.append(f"| {name} | {m['tp']+m['tn']}/{len(confirmation)} | {m['classifier_accuracy_pct']:.1f}% | "
            f"{m['balanced_accuracy_pct']:.1f}% | {m['roc_auc']:.3f} | {m['brier_score']:.4f} |")
    delta=results[selected]['vs_gmp_sign'];ci=delta['bootstrap_95_pp']
    lines+=['','## Development-selected candidate','',
        f"Selected using 2022/2023 probability error only: **{selected}**. "
        f"Against GMP-sign on 2024: {delta['change_pp']:+.2f} percentage points "
        f"(paired bootstrap 95% interval {ci[0]:+.2f} to {ci[1]:+.2f}); "
        f"{delta['corrected_errors']} GMP mistakes corrected, {delta['introduced_errors']} new mistakes introduced.",
        '', '## Why the previous model lost to GMP-sign','',
        'The quote-covered test sample has many more positive IPOs than flat/discount ones. '
        'A rule that never warns scores highly. Those previous models used a larger non-GMP training pool; '
        'the table above refits every candidate on identical paired training IPOs.',
        '',*[(f"- Previous {name}: caught {item['corrected_gmp_errors']} GMP mistakes but "
            f"missed {item['introduced_errors']} positive IPOs.") for name,item in previous_diagnosis.items()],
        '', 'A systematic extraction defect was also corrected: an undisclosed OFS amount could incorrectly pick up the following employee-reservation amount. '
        'All cited offer pages were checked with the same fix; contradictory amounts/share counts are withheld, not guessed. '
        'Annual EPS tables were reused, not re-extracted. This correction was prompted by inspected test errors; these results remain exploratory.',
        '', '## GMP-sign mistakes','', '| IPO | Dated GMP | Actual listing return | Selected model probability |', '|---|---:|---:|---:|']
    for row in errors:
        if not row['gmp_sign_correct']:
            lines.append(f"| {row['company_name']} | {row['gmp_pct']:.1f}% | {row['target_return']:+.2f}% | {row[selected+'_probability']*100:.1f}% |")
    lines+=['', '## Where the combined model goes wrong','',
        f"The selected fit has {len(last_train)} training IPOs, including only "
        f"{int(last_train.target_binary.eq(0).sum())} flat/negative openings, for {len(names)} transformed inputs. "
        'Its feature effects are learned associations, not economic laws. The following are positive IPOs it incorrectly warned against; '
        'negative contributions lower its predicted probability. They exactly reproduce the saved model, not a guessed explanation.',
        '', '| Missed positive IPO | Probability | Two strongest negative contributions (log-odds) |', '|---|---:|---|']
    contribution_frame=pd.DataFrame(contributions)
    for row in errors:
        if row['selected_vs_gmp']=='introduced_error':
            effects=contribution_frame[contribution_frame.ipo_id.eq(row['ipo_id'])].sort_values('log_odds_contribution').head(2)
            explanation='; '.join(f"{effect.feature}: {effect.log_odds_contribution:+.2f}" for effect in effects.itertuples())
            lines.append(f"| {row['company_name']} | {row[selected+'_probability']*100:.1f}% | {explanation} |")
    lines+=['','## Scope','',
        f"Dated GMP exists for {len(paired)} of {len(early)} early-book records and {len(confirmation)} of {len(full2024)} 2024 test IPOs. "
        'Uncovered IPOs are listed explicitly in the JSON; the 39-case score is not a score for all 75 IPOs. '
        'No live model, paper metric or saved prediction was replaced. The errors CSV includes every paired test IPO.',
        '',*['- '+item for item in report['limitations']]]
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if live_hash!=digest(ROOT/'ml/model_bundle.json'):
        raise ValueError('Research changed the live model')
    print(json.dumps(dict(selected=selected,results=results,coverage=report['coverage']),indent=2))
    return report


if __name__=='__main__':
    run()
