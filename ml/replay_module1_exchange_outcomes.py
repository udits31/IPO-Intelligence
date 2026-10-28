"""Separate label-source sensitivity replay; never promotes a live model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.audit_module1_listing_opens import OUTPUT as OPENINGS, verified_outcomes
from ml.model_pipeline import (prepare_dataset, EXCLUDED_NON_CORPORATE_IDS,
                               fit_model, score_model, summarize)
from ml.return_pipeline import prepare_returns, evaluate_track, render, summary, ranking_comparison

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'output/module1_exchange_outcome_replay_20261021.json'
OLD_REPORT = ROOT/'output/module1_return_benchmark.json'
OLD_LEDGER = ROOT/'output/module1_return_prediction_ledger.csv'


def frozen_forecasts_rescored(audit):
    original = pd.read_csv(OLD_LEDGER,float_precision='round_trip')
    merged = original.merge(audit[['ipo_id','exchange_open','exchange']],on='ipo_id',validate='many_to_one')
    if len(merged)!=len(original):
        raise ValueError('Every original return forecast must have a verified outcome')
    merged['target_return'] = (merged.exchange_open/merged.issue_price-1)*100
    rows = {}
    for name,group in merged.groupby('track'):
        paired = group[group.gmp_pct.notna()]
        stats = summary(paired)
        improvement = ((paired.target_return-paired.gmp_pct).abs()-(paired.target_return-paired.prediction).abs()).to_numpy()
        rng = np.random.default_rng(42)
        stats['mae_improvement_over_gmp_pp'] = float(improvement.mean())
        stats['mae_improvement_bootstrap_95_pp'] = np.quantile([
            rng.choice(improvement,len(improvement),replace=True).mean() for _ in range(2000)], [.025,.975]).tolist()
        rows[name] = {'confirmation_2024':summary(group),'paired_2024':stats,
                      'ranking_2024':ranking_comparison(group)}
    merged.to_csv(REPORT.with_name(REPORT.stem+'_frozen_forecasts_ledger.csv'),index=False)
    return rows


def direction_replay(audit):
    original = prepare_dataset()
    original = original[original.year.between(2021,2026) & ~original.id.isin(EXCLUDED_NON_CORPORATE_IDS)].copy()
    corrected = original.merge(audit[['ipo_id','exchange_open','exchange']],left_on='id',right_on='ipo_id',
                               suffixes=('','_audit'),validate='one_to_one')
    corrected['target_return'] = (corrected.exchange_open/corrected.issue_price-1)*100
    corrected['target_binary'] = corrected.target_return.gt(0).astype(int)
    required = set(original.loc[original.year.le(2024),'id'])
    if set(corrected.loc[corrected.year.le(2024),'id']) != required:
        raise ValueError('All 252 original training IPO outcomes must be verified; no selective training subset')
    cases = []
    for year in (2023,2024,2025,2026):
        last_train_year = min(year-1,2024)
        train = corrected[corrected.year.le(last_train_year)]
        test = corrected[corrected.year.eq(year)].copy()
        old_train = original[original.year.le(last_train_year)]
        s,m = fit_model(train)
        old_s,old_m = fit_model(old_train)
        test['probability_refit_on_exchange_labels'] = score_model(s,m,test)
        test['probability_original_model'] = score_model(old_s,old_m,test)
        test['trained_through'] = last_train_year
        cases.append(test)
    ledger = pd.concat(cases)
    result = {}
    for label,years in [('2023_2024',[2023,2024]),('2025_2026',[2025,2026])]:
        part = ledger[ledger.year.isin(years)]
        result[label] = {'original_model_scored_on_verified_outcomes':summarize(part.target_binary,part.probability_original_model,.5),
                         'same_lr_refit_on_verified_labels':summarize(part.target_binary,part.probability_refit_on_exchange_labels,.5)}
    result['coverage'] = {'training_n':len(required),'later_requested_n':int(original.year.ge(2025).sum()),
                          'later_verified_n':int(corrected.year.ge(2025).sum()),
                          'unverified_ids':sorted(set(original.id)-set(corrected.id))}
    ledger[['id','company_name','year','trained_through','exchange','exchange_open','issue_price',
            'target_return','target_binary','probability_original_model','probability_refit_on_exchange_labels']].to_csv(
        REPORT.with_name(REPORT.stem+'_direction_ledger.csv'),index=False)
    return result


def main():
    audit = verified_outcomes()
    direction = direction_replay(audit)
    rescored = frozen_forecasts_rescored(audit)
    frame,cohort_audit = prepare_returns()
    joined = frame.merge(audit[['ipo_id','exchange_open','exchange']],on='ipo_id',validate='one_to_one')
    if set(joined.ipo_id) != set(frame.ipo_id):
        raise ValueError('All pre-close research IPO outcomes must be verified; no selective return cohort')
    joined['listing_open_price'] = joined.exchange_open
    joined['target_return'] = (joined.exchange_open/joined.issue_price-1)*100
    tracks,ledgers = {},[]
    for name,gmp in [('bids_only',False),('bids_plus_gmp',True)]:
        _,result,ledger = evaluate_track(joined,with_gmp=gmp)
        tracks[name] = result
        ledger['track'] = name
        ledgers.append(ledger)
    paired = tracks['bids_plus_gmp']
    edge = bool(paired['ranking_2024']['better_than_gmp_established'])
    report = {'title':'Module 1 — Exchange-verified opening outcomes and GMP comparison',
              'model_version':'research-only-source-sensitivity; live artifacts unchanged',
              'training_years':[2021,2022,2023,2024],
              'direction_model_family':'Unchanged 3-bid StandardScaler + LogisticRegression(C=0.5), cutoff=0.5',
              'direction_is_final_book_not_preclose_validation':True,
              'cohort_audit':cohort_audit,'direction':direction,'tracks':tracks,
              'original_return_forecasts_rescored':rescored,
              'audit_for_model_version':json.loads(OLD_REPORT.read_text(encoding='utf-8'))['model_version'],
              'better_than_gmp_established':edge,'model_promoted':False,
              'conclusion':('A reliable advantage over GMP-only selection has NOT been established.' if not edge else
                            'A retrospective paired selection advantage is indicated; prospective validation remains required.'),
              'input_sha256':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (OPENINGS,OLD_REPORT,OLD_LEDGER,ROOT/'data/canonical_listed_ipos.csv',ROOT/'data/subscription_observations.csv',
                               ROOT/'data/daywise_subscription_research_cohort.csv',ROOT/'data/preclose_gmp_history.csv')},
              'limitations':['NSE opening is used when listed on NSE at the IPO date; exact-code BSE opening otherwise.',
                 'Canonical prices, labels and live model artifacts have not been silently rewritten.',
                 'Offer prices and historical bid publication times still require independent point-in-time offer-document verification.',
                 'Daily GMP quotes must precede the bid date because their update time is unknown.',
                 'The final-book direction replay is not evidence of pre-application-cutoff accuracy.',
                 'The exact original cohort includes the Ruchi Soya follow-on offer; retained for cohort parity, not relabelled as a new IPO.',
                 '2024 and later outcomes have been inspected previously; not a blind test.',
                 'Conditional-on-allotment opening gains are not wallet returns or guaranteed executable profits.']}
    REPORT.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    ledger = pd.concat(ledgers)
    ledger.to_csv(REPORT.with_name(REPORT.stem+'_return_ledger.csv'),index=False)
    lines = [render(report),'\n## Verified-outcome direction sensitivity\n',
             'This is the final-book classifier only; it does not validate application-time prediction.\n',
             '| Years | Verified cases | Original LR vs verified outcomes | Same LR refit on verified outcomes |',
             '|---|---:|---:|---:|']
    for key in ('2023_2024','2025_2026'):
        old = direction[key]['original_model_scored_on_verified_outcomes']
        new = direction[key]['same_lr_refit_on_verified_labels']
        lines.append(f"| {key.replace('_','–')} | {new['sample_size']} | {old['classifier_accuracy_pct']:.1f}% | {new['classifier_accuracy_pct']:.1f}% |")
    lines += ['',f"Coverage: {direction['coverage']}. Unverified IPOs are not counted as successful predictions."]
    fixed = rescored['bids_plus_gmp']['paired_2024']
    lines += ['','## Original frozen return predictions, scored without retraining','',
              f"On {fixed['n']} paired IPOs, original forecasts have {fixed['mae_pp']:.2f} pp return error "
              f"versus {fixed['gmp_mae_pp']:.2f} pp for GMP. Average improvement "
              f"{fixed['mae_improvement_over_gmp_pp']:+.2f} pp; 95% descriptive interval "
              f"{fixed['mae_improvement_bootstrap_95_pp'][0]:+.2f} to {fixed['mae_improvement_bootstrap_95_pp'][1]:+.2f} pp. "
              'These predictions were not refitted after observing corrected 2024 outcomes. '
              'This audit changes the scoring source, not the saved forecasts or deployed model.']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'direction':direction,'paired_return':paired['paired_2024'],
                      'paired_ranking':paired['ranking_2024'],'model_promoted':False},indent=2))


if __name__=='__main__':
    main()
