"""Test whether known, matured IPO outcomes can calibrate Module 2.

This is a retrospective, research-only overlay. Each prior outcome enters the
rolling rate only after its 30-session outcome date, strictly before the new
decision date. No deployed rule, paper model, or live API is modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.research_module2_exchange_delivery import (  # noqa: E402
    DEV_PRICE, LATER_PRICE, DEV_DELIVERY, LATER_DELIVERY,
    PRICE_FEATURES, DELIVERY_FEATURES, assemble,
)
from ml.research_module2_price_model import score  # noqa: E402


OUTPUT = ROOT / 'output/module2_online_prior_research.json'
LEDGER = ROOT / 'output/module2_online_prior_research_ledger.csv'
FEATURES = PRICE_FEATURES + DELIVERY_FEATURES


def outcome_dates() -> dict[int, pd.Timestamp]:
    dates = {}
    for path in (DEV_PRICE, LATER_PRICE):
        frame = pd.read_csv(path, parse_dates=['date'])
        for ipo_id, group in frame.groupby('ipo_id'):
            ordered = group.sort_values('date')
            if len(ordered) > 60:
                dates[int(ipo_id)] = ordered.iloc[60].date
    return dates


def rolling_prior(row: pd.Series, all_cases: pd.DataFrame,
                  training_prior: float, window: int, shrink: int) -> tuple[float, int]:
    known = all_cases[(all_cases.outcome_date < row.decision_date)
                      & (all_cases.ipo_id != row.ipo_id)]
    known = known.sort_values(['outcome_date', 'ipo_id']).tail(window)
    n = len(known)
    return (float((known.up.sum() + shrink * training_prior) / (n + shrink)), n)


def adjust(base: np.ndarray, scored: pd.DataFrame, all_cases: pd.DataFrame,
           training_prior: float, window: int, shrink: int) -> tuple[np.ndarray, list[int]]:
    probabilities, counts = [], []
    stable_prior = np.clip(training_prior, 1e-5, 1 - 1e-5)
    prior_logit = np.log(stable_prior / (1 - stable_prior))
    for probability, (_, row) in zip(base, scored.iterrows()):
        local_prior, count = rolling_prior(row, all_cases, training_prior, window, shrink)
        p = np.clip(probability, 1e-5, 1 - 1e-5)
        local = np.clip(local_prior, 1e-5, 1 - 1e-5)
        logit = np.log(p / (1 - p)) + np.log(local / (1 - local)) - prior_logit
        probabilities.append(float(1 / (1 + np.exp(-logit))))
        counts.append(count)
    return np.asarray(probabilities), counts


def fit(train: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.01, max_iter=2000, random_state=42))
    return model.fit(train[FEATURES], train.up)


def run() -> dict:
    data, coverage = assemble()
    data = data.dropna(subset=FEATURES).copy()
    dates = outcome_dates()
    data['decision_date'] = pd.to_datetime(data.decision_date)
    data['outcome_date'] = data.ipo_id.map(dates)
    if data.outcome_date.isna().any() or (data.outcome_date <= data.decision_date).any():
        raise ValueError('Missing or leaked outcome date')
    train = data[data.year.between(2021, 2023)]
    validation = data[data.year == 2024].copy()
    development = data[data.year.between(2021, 2024)]
    later = data[data.year.isin([2025, 2026])].copy()
    first_model = fit(train)
    validation_base = first_model.predict_proba(validation[FEATURES])[:, 1]
    first_prior = float(train.up.mean())
    candidates = []
    for window in (20, 50):
        for shrink in (10, 30):
            p, counts = adjust(validation_base, validation, data,
                               first_prior, window, shrink)
            candidates.append({'window': window, 'shrink': shrink,
                               'validation': score(validation, p),
                               'validation_prior_count_median': float(np.median(counts))})
    selected = min(candidates, key=lambda item: item['validation']['brier'])
    final_model = fit(development)
    later_base = final_model.predict_proba(later[FEATURES])[:, 1]
    later_adjusted, prior_counts = adjust(
        later_base, later, data, float(development.up.mean()),
        selected['window'], selected['shrink'])
    ledger = later[['ipo_id', 'year', 'decision_date', 'outcome_date',
                    'future_return_pct', 'up']].copy()
    ledger['model_probability'] = later_base
    ledger['online_prior_probability'] = later_adjusted
    ledger['known_prior_outcome_count'] = prior_counts
    ledger.to_csv(LEDGER, index=False)
    result = {
        'protocol': 'NSE price/delivery; fixed logistic C=.01; rolling prior window and shrink selected on 2024 Brier; exploratory 2025-26 chronological as-of replay',
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (DEV_PRICE, LATER_PRICE, DEV_DELIVERY, LATER_DELIVERY)},
        'coverage': coverage,
        'training_n': len(train), 'validation_n': len(validation), 'later_n': len(later),
        'candidates': candidates, 'selected': {'window': selected['window'],
                                              'shrink': selected['shrink']},
        'later': {'base': score(later, later_base),
                  'online_prior': score(later, later_adjusted),
                  'by_year': {str(year): {'base': score(later[later.year == year],
                                                        later_base[later.year.to_numpy() == year]),
                                          'online_prior': score(later[later.year == year],
                                                                later_adjusted[later.year.to_numpy() == year])}
                              for year in (2025, 2026) if (later.year == year).any()}},
        'ledger': str(LEDGER.relative_to(ROOT)),
        'limitations': [
            'Earlier mature outcomes were reconstructed from exchange history; original as-published archive timestamps are not stored.',
            '2025-26 labels had been inspected in previous project research; this is not a blind holdout.',
            'Selected on only 46 validation IPOs; do not deploy without prospective verification.',
            'Underlying deployed Module 2 rules and paper claims are unchanged.',
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    report = run()
    print(json.dumps({key: report[key] for key in ('selected', 'later')}, indent=2))
