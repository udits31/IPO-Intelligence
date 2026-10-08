"""Research-only, same-cohort test of real NSE delivery history for Module 2.

No deployment and no synthetic inputs. Delivery statistics end on the prior
completed session to avoid assuming same-day publication by decision close.
Candidate selection uses 2024 only; later-year results remain exploratory
because this repository's 2025-26 outcomes were examined in earlier studies.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.research_module2_price_model import FEATURES as PRICE_FEATURES
from ml.research_module2_price_model import OLD, IPOS, cases, score


LATER_AUDITED = ROOT / 'data/audit/module2_holdout_prices_2025_2026_nse_audited.csv'
DELIVERY_PATHS = {
    'development': ROOT / 'data/audit/module2_nse_delivery_development.csv',
    'holdout': ROOT / 'data/audit/module2_nse_delivery_holdout.csv',
}
OUTPUT = ROOT / 'output/module2_nse_delivery_research.json'
LEDGER = ROOT / 'output/module2_nse_delivery_research_ledger.csv'
DELIVERY_FEATURES = [
    'delivery_5_mean_pct', 'delivery_20_mean_pct',
    'delivery_5_minus_20_pct', 'traded_qty_5_to_20',
]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble_cases() -> tuple[pd.DataFrame, dict]:
    prices = cases(price_paths=(OLD, LATER_AUDITED))
    raw = pd.concat([pd.read_csv(path, parse_dates=['session_date'])
                     for path in DELIVERY_PATHS.values()], ignore_index=True)
    eligible = set()
    for path in DELIVERY_PATHS.values():
        manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        eligible.update(manifest['alignment_audit']['eligible_ipo_ids'])
    raw = raw[raw.ipo_id.isin(eligible)].sort_values(['ipo_id', 'session_date'])
    price_dates = pd.concat([pd.read_csv(path, parse_dates=['date'])
                             for path in (OLD, LATER_AUDITED)], ignore_index=True)
    prior_session = {
        int(ipo_id): group.sort_values('date').iloc[29]['date']
        for ipo_id, group in price_dates.groupby('ipo_id') if len(group) >= 31
    }
    rows = []
    rejections = {'not_aligned_to_nse': 0, 'missing_prior_session': 0,
                  'fewer_than_15_prior_delivery_sessions': 0}
    for case in prices.itertuples(index=False):
        ipo_id = int(case.ipo_id)
        if ipo_id not in eligible:
            rejections['not_aligned_to_nse'] += 1
            continue
        expected_prior = prior_session.get(ipo_id)
        if expected_prior is None:
            rejections['missing_prior_session'] += 1
            continue
        history = raw[(raw.ipo_id == ipo_id) &
                      (raw.session_date <= expected_prior)].tail(20)
        if history.empty or history.iloc[-1].session_date != expected_prior:
            rejections['missing_prior_session'] += 1
            continue
        if len(history) < 15:
            rejections['fewer_than_15_prior_delivery_sessions'] += 1
            continue
        latest = history.tail(5)
        volume_recent = float(latest.traded_qty.mean())
        volume_history = float(history.traded_qty.mean())
        rows.append({
            **case._asdict(),
            'delivery_5_mean_pct': float(latest.deliverable_pct.mean()),
            'delivery_20_mean_pct': float(history.deliverable_pct.mean()),
            'delivery_5_minus_20_pct': float(latest.deliverable_pct.mean()
                                                - history.deliverable_pct.mean()),
            'traded_qty_5_to_20': float(np.log(volume_recent / volume_history))
            if volume_recent > 0 and volume_history > 0 else np.nan,
            'latest_delivery_session': latest.iloc[-1].session_date.date().isoformat(),
        })
    return pd.DataFrame(rows), {'price_cases_before_delivery_gate': len(prices),
                                'rejections': rejections}


def make_model(columns: list[str], c: float) -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
        ('logistic', LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])


def run() -> dict:
    data, coverage = assemble_cases()
    if data.empty:
        raise RuntimeError('No aligned NSE delivery cases are available')
    train = data[data.year.between(2021, 2023)]
    validate = data[data.year == 2024]
    later = data[data.year.isin([2025, 2026])]
    if min(len(train), len(validate), len(later)) < 15:
        raise RuntimeError('Insufficient same-cohort train/validation/later cases')
    candidates = []
    for group_name, columns in [('price_only', PRICE_FEATURES),
                                ('price_plus_nse_delivery', PRICE_FEATURES + DELIVERY_FEATURES)]:
        for c in (0.01, 0.1, 1.0, 10.0):
            fitted = make_model(columns, c).fit(train[columns], train.up)
            validation_prob = fitted.predict_proba(validate[columns])[:, 1]
            candidates.append({'group': group_name, 'C': c,
                               'validation': score(validate, validation_prob)})
    comparisons = {}
    later_probabilities = {}
    development = data[data.year.between(2021, 2024)]
    for group_name, columns in [('price_only', PRICE_FEATURES),
                                ('price_plus_nse_delivery', PRICE_FEATURES + DELIVERY_FEATURES)]:
        selected = min((row for row in candidates if row['group'] == group_name),
                       key=lambda row: row['validation']['brier'])
        fitted = make_model(columns, selected['C']).fit(development[columns], development.up)
        probability = fitted.predict_proba(later[columns])[:, 1]
        later_probabilities[group_name] = probability
        comparisons[group_name] = {
            'selected_C_on_2024': selected['C'],
            'later': score(later, probability),
            'by_year': {str(year): score(later[later.year == year],
                                        probability[later.year.to_numpy() == year])
                        for year in (2025, 2026) if (later.year == year).any()},
        }
    price_correct = (later_probabilities['price_only'] >= 0.5) == later.up.to_numpy(dtype=bool)
    delivery_correct = ((later_probabilities['price_plus_nse_delivery'] >= 0.5)
                        == later.up.to_numpy(dtype=bool))
    delivery_only_correct = int((~price_correct & delivery_correct).sum())
    price_only_correct = int((price_correct & ~delivery_correct).sum())
    discordant = delivery_only_correct + price_only_correct
    paired = {
        'delivery_correct_price_wrong': delivery_only_correct,
        'price_correct_delivery_wrong': price_only_correct,
        'exact_two_sided_p': round(float(binomtest(
            min(delivery_only_correct, price_only_correct), discordant,
            p=0.5).pvalue), 4) if discordant else None,
    }
    ledger = later[['ipo_id', 'year', 'decision_date', 'latest_delivery_session',
                    'future_return_pct', 'up']].copy()
    ledger['price_only_probability'] = later_probabilities['price_only']
    ledger['price_plus_delivery_probability'] = later_probabilities['price_plus_nse_delivery']
    ledger['price_only_correct'] = price_correct
    ledger['price_plus_delivery_correct'] = delivery_correct
    ledger.to_csv(LEDGER, index=False)
    result = {
        'protocol': '2021-23 training; C selected on 2024 Brier per feature group; refit 2021-24; exploratory 2025-26 comparison on identical aligned IPOs',
        'data_sources': {'NSE_delivery': 'https://www.nseindia.com/report-detail/eq_security',
                         'price_history': 'Yahoo raw closes with NSE alignment gate'},
        'input_sha256': {str(path.relative_to(ROOT)): file_hash(path)
                         for path in [OLD, LATER_AUDITED, IPOS, *DELIVERY_PATHS.values()]},
        'publication_guard': 'Delivery features use through session 30 at the session-31 close decision.',
        'coverage': {**coverage, 'training_n': len(train), 'validation_n': len(validate),
                     'later_n': len(later)},
        'candidates': candidates, 'comparisons': comparisons,
        'paired_later_comparison': paired,
        'later_prediction_ledger': str(LEDGER.relative_to(ROOT)),
        'limitations': [
            'NSE delivery was downloaded retrospectively; original publication timestamps were not archived.',
            'Yahoo future closes remain secondary outcomes; corporate-action handling needs further exchange reconciliation.',
            'Later-year results have been inspected in prior repository research and are not a blind test.',
            'No new feature is promoted to the deployed monitor by this experiment.',
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps({'coverage': result['coverage'],
                      'comparisons': result['comparisons'],
                      'paired_later_comparison': result['paired_later_comparison']}, indent=2))
