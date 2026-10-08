"""Test NSE trade-flow features on the same price-and-delivery IPO cohort.

Only exchange observations strictly before the 31st-session decision are
features. Later-year outcomes are already known to the project, so results
are exploratory and cannot be called a blind holdout or deployed accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.research_module2_exchange_delivery import (
    DEV_DELIVERY, LATER_DELIVERY, DEV_PRICE, LATER_PRICE,
    PRICE_FEATURES, DELIVERY_FEATURES, assemble,
)
from ml.research_module2_price_model import score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/module2_exchange_flow_research.json'
LEDGER = ROOT / 'output/module2_exchange_flow_research_ledger.csv'
FLOW = ['turnover_5_to_20', 'trades_5_to_20',
        'trade_value_per_trade_5_to_20', 'deliverable_qty_5_to_20']
GROUPS = {
    'price_delivery': PRICE_FEATURES + DELIVERY_FEATURES,
    'price_delivery_flow': PRICE_FEATURES + DELIVERY_FEATURES + FLOW,
}


def log_ratio(recent: pd.Series, reference: pd.Series) -> float:
    left, right = recent.mean(), reference.mean()
    return float(np.log(left / right)) if pd.notna(left) and pd.notna(right) and left > 0 and right > 0 else np.nan


def run() -> dict:
    cases, coverage = assemble()
    joined = []
    for path in (DEV_DELIVERY, LATER_DELIVERY):
        frame = pd.read_csv(path, parse_dates=['session_date'])
        for case in cases[cases.ipo_id.isin(frame.ipo_id)].itertuples(index=False):
            # The previous delivery session was checked against the price
            # series by assemble(); never use today's unpublished delivery.
            prior = frame[(frame.ipo_id == case.ipo_id)
                          & (frame.session_date <= pd.Timestamp(case.latest_delivery_date))]
            prior = prior.sort_values('session_date').tail(20)
            if len(prior) != 20 or prior.iloc[-1].session_date.date().isoformat() != case.latest_delivery_date:
                raise ValueError(f'Incomplete prior-session flow for IPO {case.ipo_id}')
            if not (prior.session_date < pd.Timestamp(case.decision_date)).all():
                raise ValueError('Decision-day or future trade flow entered the feature window')
            recent = prior.tail(5)
            trade_unit = prior.traded_value / prior.total_trades.replace(0, np.nan)
            joined.append({
                'ipo_id': int(case.ipo_id),
                'turnover_5_to_20': log_ratio(recent.traded_value, prior.traded_value),
                'trades_5_to_20': log_ratio(recent.total_trades, prior.total_trades),
                'trade_value_per_trade_5_to_20': log_ratio(trade_unit.tail(5), trade_unit),
                'deliverable_qty_5_to_20': log_ratio(recent.deliverable_qty, prior.deliverable_qty),
            })
    flow = pd.DataFrame(joined)
    if flow.ipo_id.duplicated().any() or len(flow) != len(cases):
        raise ValueError('Trade-flow join is not one-to-one with the NSE price cohort')
    data = cases.merge(flow, on='ipo_id', validate='one_to_one')
    incomplete = data[list(set(sum(GROUPS.values(), [])))].isna().any(axis=1)
    excluded = int(incomplete.sum())
    data = data[~incomplete].copy()
    train = data[data.year.between(2021, 2023)]
    validation = data[data.year == 2024]
    development = data[data.year.between(2021, 2024)]
    later = data[data.year.isin([2025, 2026])]
    if min(len(train), len(validation), len(later)) < 20:
        raise ValueError('Insufficient complete real-observation cohort')
    comparison, candidates, probabilities = {}, [], {}
    for group, columns in GROUPS.items():
        for regularization in (.01, .1, 1.0):
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=regularization, max_iter=2000, random_state=42)).fit(train[columns], train.up)
            candidates.append({'group': group, 'C': regularization,
                               'validation': score(validation, model.predict_proba(validation[columns])[:, 1])})
        selected = min((row for row in candidates if row['group'] == group),
                       key=lambda row: row['validation']['brier'])
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=selected['C'], max_iter=2000, random_state=42)).fit(development[columns], development.up)
        probability = model.predict_proba(later[columns])[:, 1]
        probabilities[group] = probability
        comparison[group] = {'selected_C_on_2024': selected['C'],
                             'validation': selected['validation'],
                             'later': score(later, probability),
                             'later_by_year': {str(year): score(later[later.year == year],
                                                                probability[later.year.to_numpy() == year])
                                               for year in (2025, 2026) if (later.year == year).any()}}
    truth = later.up.to_numpy(dtype=bool)
    original = (probabilities['price_delivery'] >= .5) == truth
    candidate = (probabilities['price_delivery_flow'] >= .5) == truth
    gain, loss = int((~original & candidate).sum()), int((original & ~candidate).sum())
    ledger = later[['ipo_id', 'year', 'decision_date', 'future_return_pct', 'up']].copy()
    for name, probability in probabilities.items():
        ledger[f'{name}_probability'] = probability
    ledger.to_csv(LEDGER, index=False)
    result = {
        'protocol': 'Same NSE IPO cohort, prior-session exchange trade flow only; fit 2021-23, select C on 2024 Brier, refit through 2024, exploratory 2025-26.',
        'coverage': {**coverage, 'excluded_missing_flow': excluded,
                     'complete_by_year': {str(year): int(count) for year, count in data.groupby('year').size().items()}},
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (DEV_PRICE, LATER_PRICE, DEV_DELIVERY, LATER_DELIVERY)},
        'feature_groups': GROUPS, 'candidates': candidates, 'comparisons': comparison,
        'paired_flow_addition': {'flow_right_baseline_wrong': gain, 'baseline_right_flow_wrong': loss,
                                 'exact_two_sided_p': round(float(binomtest(min(gain, loss), gain + loss).pvalue), 4)
                                 if gain + loss else None},
        'ledger': str(LEDGER.relative_to(ROOT)),
        'limitations': [
            'Historical NSE trade flow was retrieved retrospectively; original publication times are not archived.',
            'Flow enters only through the session preceding the decision; the decision-day close itself is assumed observed.',
            'Later-year outcomes were inspected by earlier studies; this is exploratory, not a blind validation.',
            'The deployed Module 2 monitor and the paper model remain unchanged.',
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps({key: result[key] for key in ('coverage', 'comparisons', 'paired_flow_addition')}, indent=2))
