"""Same-cohort exchange-price, delivery, and dated-filing comparison.

The original NSE announcements are archived as real observations. Filing
counts are deterministic transforms of subjects and details before the exact
decision cutoff, not invented sentiment labels. Research only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.research_module2_exchange_delivery import (  # noqa: E402
    DEV_PRICE, LATER_PRICE, DEV_DELIVERY, LATER_DELIVERY,
    PRICE_FEATURES, DELIVERY_FEATURES, assemble,
)
from ml.research_module2_filings import FILING_FEATURES, filing_features  # noqa: E402
from ml.research_module2_price_model import score  # noqa: E402


FILINGS = ROOT / 'data/audit/module2_dated_nse_filings_exchange_decisions.jsonl'
OUTPUT = ROOT / 'output/module2_exchange_filings_research.json'
LEDGER = ROOT / 'output/module2_exchange_filings_research_ledger.csv'
GROUPS = {
    'price_only': PRICE_FEATURES,
    'price_delivery': PRICE_FEATURES + DELIVERY_FEATURES,
    'price_filings': PRICE_FEATURES + FILING_FEATURES,
    'price_delivery_filings': PRICE_FEATURES + DELIVERY_FEATURES + FILING_FEATURES,
}


def paired(reference: np.ndarray, challenger: np.ndarray, labels: np.ndarray) -> dict:
    a = (reference >= .5) == labels
    b = (challenger >= .5) == labels
    gain = int((~a & b).sum())
    loss = int((a & ~b).sum())
    return {'challenger_right_reference_wrong': gain,
            'reference_right_challenger_wrong': loss,
            'exact_two_sided_p': round(float(binomtest(min(gain, loss), gain + loss).pvalue), 4)
            if gain + loss else None}


def run() -> dict:
    cases, coverage = assemble()
    archive = [json.loads(line) for line in FILINGS.read_text(encoding='utf-8').splitlines()]
    if len(archive) != len(cases) or any(row['status'] != 'verified_response' for row in archive):
        raise ValueError('Exact-decision NSE filing archive is incomplete')
    keys = {int(row['ipo_id']): row for row in archive}
    if len(keys) != len(archive):
        raise ValueError('Duplicate filing response')
    for case in cases.itertuples(index=False):
        record = keys.get(int(case.ipo_id))
        if record is None or record['decision_date'] != case.decision_date:
            raise ValueError(f'Filing decision date differs from NSE price for IPO {case.ipo_id}')
    features = pd.DataFrame(filing_features(row) for row in archive)
    data = cases.merge(features, on='ipo_id', validate='one_to_one')
    all_features = PRICE_FEATURES + DELIVERY_FEATURES + FILING_FEATURES
    if data[all_features].isna().any().any():
        raise ValueError('Missing feature; no synthetic or imputed data allowed')
    train = data[data.year.between(2021, 2023)]
    validation = data[data.year == 2024]
    development = data[data.year.between(2021, 2024)]
    later = data[data.year.isin([2025, 2026])]
    if min(len(train), len(validation), len(later)) < 20:
        raise ValueError('Insufficient year-separated cohort')
    comparisons, later_probabilities, candidates = {}, {}, []
    for name, columns in GROUPS.items():
        for regularization in (.01, .1, 1.0):
            fitted = make_pipeline(StandardScaler(), LogisticRegression(
                C=regularization, max_iter=2000, random_state=42)).fit(train[columns], train.up)
            p = fitted.predict_proba(validation[columns])[:, 1]
            candidates.append({'group': name, 'C': regularization,
                               'validation': score(validation, p)})
        selected = min((row for row in candidates if row['group'] == name),
                       key=lambda row: row['validation']['brier'])
        final = make_pipeline(StandardScaler(), LogisticRegression(
            C=selected['C'], max_iter=2000, random_state=42)).fit(
                development[columns], development.up)
        p = final.predict_proba(later[columns])[:, 1]
        later_probabilities[name] = p
        comparisons[name] = {'selected_C_on_2024': selected['C'],
                             'validation': selected['validation'],
                             'later': score(later, p),
                             'later_by_year': {str(year): score(later[later.year == year],
                                                            p[later.year.to_numpy() == year])
                                               for year in (2025, 2026)
                                               if (later.year == year).any()}}
    ledger = later[['ipo_id', 'year', 'decision_date', 'future_return_pct', 'up']].copy()
    for name, p in later_probabilities.items():
        ledger[f'{name}_probability'] = p
    ledger.to_csv(LEDGER, index=False)
    result = {
        'protocol': 'Same 285 NSE price/delivery/filing IPOs; train 2021-23, select regularization on 2024 Brier, refit through 2024, exploratory 2025-26.',
        'coverage': {**coverage, 'exact_date_filing_responses': len(archive),
                     'filing_rows_before_cutoff': int(sum(row['eligible_rows'] for row in archive)),
                     'nonempty_filing_cases': int(sum(row['eligible_rows'] > 0 for row in archive))},
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (DEV_PRICE, LATER_PRICE, DEV_DELIVERY, LATER_DELIVERY, FILINGS)},
        'feature_groups': GROUPS,
        'candidates': candidates, 'comparisons': comparisons,
        'paired_filing_addition_to_price_delivery': paired(
            later_probabilities['price_delivery'], later_probabilities['price_delivery_filings'],
            later.up.to_numpy(dtype=bool)),
        'ledger': str(LEDGER.relative_to(ROOT)),
        'limitations': [
            'NSE responses retrieved retrospectively; exchange dissemination timestamps are not proof of archive completeness.',
            'Keyword counts are not linguistic sentiment; attachment contents are not analyzed.',
            '2025-26 outcomes were inspected in earlier studies and are not a blind holdout.',
            'The deployed Module 2 rules and the paper model are unchanged.',
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    report = run()
    print(json.dumps({key: report[key] for key in
                      ('coverage', 'comparisons', 'paired_filing_addition_to_price_delivery')}, indent=2))
