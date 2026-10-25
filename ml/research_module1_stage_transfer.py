"""Same three-input logistic model: matched penultimate-book vs final-book fit.

Only existing retrospective public tables and verified exchange outcomes.
Earlier-year folds select thresholds; 2024 is confirmation, not a blind test.
No synthetic records, added model family or live artifact replacement.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from ml.audit_module1_listing_opens import OUTPUT as AUDIT, verified_outcomes
from ml.model_pipeline import (prepare_dataset, fit_model, score_model, summarize,
                               choose_apply_threshold, EXCLUDED_NON_CORPORATE_IDS)
from ml.return_pipeline import ROOT, prepare_returns

REPORT = ROOT/'output/module1_stage_transfer_research_20261022.json'


def datasets():
    audit = verified_outcomes()[['ipo_id', 'exchange_open']]
    final = prepare_dataset()
    final = final[final.year.between(2021, 2024) & ~final.id.isin(EXCLUDED_NON_CORPORATE_IDS)]
    final = final.merge(audit, left_on='id', right_on='ipo_id', suffixes=('', '_audit'), validate='one_to_one')
    early, coverage = prepare_returns()
    early = early.merge(audit, on='ipo_id', validate='one_to_one')
    for frame in (final, early):
        frame['target_binary'] = frame.exchange_open.gt(frame.issue_price).astype(int)
        frame['target_return'] = (frame.exchange_open/frame.issue_price-1)*100
        for source, feature in [('qib_subscription', 'log_qib'), ('retail_subscription', 'log_retail'),
                                ('hni_subscription', 'log_hni')]:
            if frame[source].isna().any() or frame[source].lt(0).any():
                raise ValueError('No missing or negative bid may become a model feature')
            frame[feature] = np.log1p(frame[source].clip(lower=.01))
    if len(final) != 252 or early.ipo_id.duplicated().any():
        raise ValueError('Unexpected training coverage or duplicate early record')
    if not pd.to_datetime(early.decision_market_date).lt(early.issue_close_date).all():
        raise ValueError('Early book decision must precede the offer closing date')
    return final, early, coverage


def run():
    final, early, coverage = datasets()
    probabilities = {}
    trace = []
    for name, training_source in [('final_book_training', final),
                                 ('matched_id_final_book_training', final[final.id.isin(early.ipo_id)]),
                                 ('matched_early_book_training', early)]:
        folds = []
        for year in (2022, 2023, 2024):
            test = early[early.year.eq(year)].copy()
            cutoff = pd.to_datetime(test.decision_market_date).min()
            candidate_train = training_source[training_source.year.lt(year)]
            # A January listing can have a December application deadline.
            # Earlier listing YEAR alone is not a sufficient as-of boundary.
            train = candidate_train[candidate_train.listing_date.lt(cutoff)]
            scaler, model = fit_model(train)
            test['probability'] = score_model(scaler, model, test)
            # Same test identities also provide a paired final-input comparison.
            actual_final = final.set_index('id').loc[test.ipo_id].copy()
            test['probability_with_final_inputs_NOT_PRE_CLOSE'] = score_model(scaler, model, actual_final)
            test['training_stage'] = name
            test['trained_through'] = int(train.year.max())
            folds.append(test)
            trace.append(dict(training_stage=name, test_year=year, train_n=len(train), test_n=len(test),
                              training_ipo_ids=sorted(int(v) for v in train.ipo_id),
                              unavailable_at_first_decision_ids=sorted(int(v) for v in
                                  candidate_train.loc[~candidate_train.listing_date.lt(cutoff), 'ipo_id']),
                              first_test_decision=str(cutoff.date()),
                              latest_training_listing=str(train.listing_date.max().date())))
        probabilities[name] = pd.concat(folds)
    results = {}
    for name, rows in probabilities.items():
        selection = rows[rows.year.isin([2022, 2023])]
        threshold, selection_metrics = choose_apply_threshold(selection.target_binary, selection.probability)
        test = rows[rows.year.eq(2024)]
        results[name] = dict(threshold_selected_on_2022_2023=threshold,
            earlier_selection=selection_metrics,
            confirmation_2024=dict(fixed_direction=summarize(test.target_binary, test.probability, .5),
                earlier_selected_policy=summarize(test.target_binary, test.probability, threshold),
                final_input_comparison_NOT_PRE_CLOSE=summarize(test.target_binary,
                    test.probability_with_final_inputs_NOT_PRE_CLOSE, .5)))
    ledger = pd.concat(probabilities.values())
    ledger.to_csv(REPORT.with_name(REPORT.stem+'_ledger.csv'), index=False)
    paths = (ROOT/'data/canonical_listed_ipos.csv', ROOT/'data/subscription_observations.csv',
             ROOT/'data/daywise_subscription_research_cohort.csv', ROOT/'data/nse_verified_offer_dates.csv',
             ROOT/'data/raw_daily_market_time_series.csv', AUDIT, ROOT/'ml/model_pipeline.py',
             ROOT/'ml/research_module1_stage_transfer.py')
    report = dict(model_family='Unchanged StandardScaler + three log-bid LogisticRegression(C=0.5)',
        live_model_changed=False, synthetic_rows=0, cohort_audit=coverage, trace=trace, results=results,
        limitations=['Daily tables were retrieved retrospectively and may have been revised; not immutable investor-time captures.',
                     'The matched-ID final-book track controls the training-IPO coverage difference when comparing stage-specific fits.',
                     'An article publication date is not proof that a penultimate-session row was published at that time.',
                     'Source-complete early-book coverage is non-random; no claim for uncovered IPOs.',
                     'Final-input comparison is deliberately marked NOT_PRE_CLOSE and cannot be counted as application-time accuracy.',
                     '2024 outcomes were inspected in previous research; this is retrospective confirmation, not blind validation.'],
        input_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    text = ['# Module 1: does final-book accuracy transfer before closing?', '',
            'Same three-input logistic family, only earlier listing years train each model.', '',
            '| Training stage | 2024 cases | Early-input direction accuracy | Balanced accuracy | Always positive |',
            '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        row = result['confirmation_2024']['fixed_direction']
        text.append(f"| {name} | {row['sample_size']} | {row['classifier_accuracy_pct']:.1f}% | {row['balanced_accuracy_pct']:.1f}% | {row['always_positive_baseline_pct']:.1f}% |")
    text += ['', *['- '+item for item in report['limitations']]]
    REPORT.with_suffix('.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    return report


if __name__ == '__main__':
    run()
