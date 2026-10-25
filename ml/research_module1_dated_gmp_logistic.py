"""Paired pre-close comparison within the existing logistic model family.

The combined variant adds one real, preceding-date GMP feature. It is not the
paper's three-input fit and is never silently promoted. No source acquisition.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.model_pipeline import FEATURE_COLS, summarize, choose_apply_threshold
from ml.research_module1_stage_transfer import ROOT, datasets

REPORT = ROOT/'output/module1_dated_gmp_logistic_research_20261022.json'
VARIANTS = {'bids_only': FEATURE_COLS, 'gmp_only': ['gmp_pct'],
            'bids_plus_gmp': [*FEATURE_COLS, 'gmp_pct']}


def fit_predict(train, test, features):
    if train[features].isna().any().any() or test[features].isna().any().any():
        raise ValueError('No missing paired feature may be imputed')
    if not train.year.lt(test.year.min()).all():
        raise ValueError('Training must precede each test listing year')
    if not train.listing_date.lt(pd.to_datetime(test.decision_market_date).min()).all():
        raise ValueError('Training outcome is unavailable at first pre-close decision')
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(C=.5, random_state=42, max_iter=1000).fit(
        scaler.transform(train[features]), train.target_binary)
    return model.predict_proba(scaler.transform(test[features]))[:, 1]


def run():
    _, early, _ = datasets()
    paired = early[early.gmp_pct.notna()].copy()
    if not pd.to_datetime(paired.gmp_date).lt(pd.to_datetime(paired.decision_market_date)).all():
        raise ValueError('Same-day or later GMP cannot support this pre-close test')
    traces, ledgers, results = [], [], {}
    for name, features in VARIANTS.items():
        folds = []
        for year in (2022, 2023, 2024):
            test = paired[paired.year.eq(year)].copy()
            cutoff = pd.to_datetime(test.decision_market_date).min()
            train = paired[paired.year.lt(year) & paired.listing_date.lt(cutoff)]
            test['probability'] = fit_predict(train, test, features)
            test['variant'] = name
            folds.append(test)
            traces.append(dict(variant=name, year=year, train_n=len(train), test_n=len(test),
                training_ids=sorted(int(v) for v in train.ipo_id),
                latest_training_outcome=str(train.listing_date.max().date()),
                first_test_decision=str(cutoff.date())))
        rows = pd.concat(folds)
        dev = rows[rows.year.isin([2022, 2023])]
        threshold, selection = choose_apply_threshold(dev.target_binary, dev.probability)
        confirmation = rows[rows.year.eq(2024)]
        results[name] = dict(earlier_selected_threshold=threshold, earlier_selection=selection,
            confirmation_2024=dict(fixed_direction=summarize(confirmation.target_binary, confirmation.probability, .5),
                earlier_selected_policy=summarize(confirmation.target_binary, confirmation.probability, threshold)))
        ledgers.append(rows)
    report = dict(live_model_changed=False, model_family='StandardScaler + LogisticRegression(C=0.5)',
        combined_features_change_paper_input_count=True, synthetic_rows=0,
        paired_by_year={str(k): int(v) for k, v in paired.groupby('year').size().items()},
        all_early_cases=len(early), paired_cases=len(paired),
        confirmation_gmp_nonpositive_count=int(paired[paired.year.eq(2024)].gmp_pct.le(0).sum()),
        traces=traces, results=results,
        limitations=['Only 39 paired confirmation IPOs, including five flat/discount outcomes; non-random quote availability.',
            'All available paired quotes are positive: simple GMP-sign selection equals always-positive on these cases.',
            'Daily GMP dates precede decisions but publisher clocks and retrospective revisions remain unverified.',
            'Logistic family is unchanged, but a fourth input is a research variant requiring disclosure, not the paper model.',
            'Earlier/later outcomes were inspected before; not a new blind or prospective test.'])
    pd.concat(ledgers).to_csv(REPORT.with_name(REPORT.stem+'_ledger.csv'), index=False)
    paths = (ROOT/'data/preclose_gmp_history.csv', ROOT/'data/daywise_subscription_research_cohort.csv',
             ROOT/'data/audit/module1_exchange_listing_open_audit.csv', ROOT/'data/canonical_listed_ipos.csv',
             ROOT/'data/nse_verified_offer_dates.csv', ROOT/'data/raw_daily_market_time_series.csv',
             ROOT/'ml/model_pipeline.py', ROOT/'ml/research_module1_stage_transfer.py',
             ROOT/'ml/research_module1_dated_gmp_logistic.py')
    report['input_sha256'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Module 1: does dated GMP add to the same logistic family?', '',
        'Identical paired test IPOs and earlier training identities for all three variants.', '',
        '| Inputs | 2024 cases | Direction accuracy | Balanced accuracy | Earlier-selected policy accuracy |',
        '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        score = result['confirmation_2024']['fixed_direction']
        policy = result['confirmation_2024']['earlier_selected_policy']
        lines.append(f"| {name} | {score['sample_size']} | {score['classifier_accuracy_pct']:.1f}% | {score['balanced_accuracy_pct']:.1f}% | {policy['classifier_accuracy_pct']:.1f}% |")
    lines += ['', 'Always-positive / positive-GMP-sign baseline: 34/39 = 87.2%.', '',
              *['- '+s for s in report['limitations']]]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    return report


if __name__ == '__main__':
    run()
