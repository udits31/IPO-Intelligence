"""Research-only same-cohort Module 2 test with original NSE observations.

Prices and delivery are exchange-sourced; stock features end at session 31
(index 30), and delivery ends at session 30 (index 29) because its release time
on the decision day is not archived. This study does not change the deployed
monitor or claim a blind 2025-26 test: those outcomes were inspected before.
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
from ml.research_module2_price_model import cases, score  # noqa: E402


DEV_PRICE = ROOT / 'data/audit/module2_development_prices_2021_2024_nse_first62.csv'
LATER_PRICE = ROOT / 'data/audit/module2_holdout_prices_2025_2026_nse_full.csv'
DEV_DELIVERY = ROOT / 'data/audit/module2_nse_delivery_development_full_window.csv'
LATER_DELIVERY = ROOT / 'data/audit/module2_nse_delivery_holdout.csv'
DEV_LEDGER = ROOT / 'output/module2_development_decision_ledger_nse_first62_20261020.csv'
LATER_LEDGER = ROOT / 'output/module2_holdout_decision_ledger_nse_full.csv'
OUTPUT = ROOT / 'output/module2_exchange_delivery_research.json'
PREDICTIONS = ROOT / 'output/module2_exchange_delivery_research_ledger.csv'
PRICE_FEATURES = ['stock_5', 'stock_10', 'stock_30', 'nifty_10', 'nifty_30',
                  'drawdown_30', 'volatility_20']
DELIVERY_FEATURES = ['delivery_5_mean_pct', 'delivery_20_mean_pct',
                     'delivery_5_minus_20_pct', 'traded_qty_5_to_20']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble() -> tuple[pd.DataFrame, dict]:
    prices = cases((DEV_PRICE, LATER_PRICE))
    joined = []
    audit = {'price_cases': len(prices), 'no_delivery_ipo': 0,
             'incomplete_31_session_alignment': 0, 'no_prior_delivery': 0}
    for price_path, delivery_path in ((DEV_PRICE, DEV_DELIVERY), (LATER_PRICE, LATER_DELIVERY)):
        price_frame = pd.read_csv(price_path, parse_dates=['date'])
        delivery = pd.read_csv(delivery_path, parse_dates=['session_date'])
        if delivery.duplicated(['ipo_id', 'session_date']).any():
            raise ValueError(f'Duplicate NSE delivery row in {delivery_path}')
        local_cases = prices[prices.ipo_id.isin(price_frame.ipo_id)]
        for case in local_cases.itertuples(index=False):
            observed = price_frame[price_frame.ipo_id == case.ipo_id].sort_values('date').head(31)
            actual = delivery[delivery.ipo_id == case.ipo_id]
            if actual.empty:
                audit['no_delivery_ipo'] += 1
                continue
            aligned = observed.merge(actual, left_on='date', right_on='session_date', how='left',
                                     validate='one_to_one')
            if (len(aligned) != 31 or aligned.nse_close.isna().any()
                    or ((aligned.nse_close / aligned.close_price_adjusted - 1).abs() > .02).any()):
                audit['incomplete_31_session_alignment'] += 1
                continue
            previous = aligned.iloc[:30].tail(20)
            if (len(previous) != 20 or previous.deliverable_pct.isna().any()
                    or previous.traded_qty.isna().any()
                    or previous.iloc[-1].date != observed.iloc[29].date):
                audit['no_prior_delivery'] += 1
                continue
            latest = previous.tail(5)
            recent_qty, base_qty = latest.traded_qty.mean(), previous.traded_qty.mean()
            if recent_qty <= 0 or base_qty <= 0:
                audit['no_prior_delivery'] += 1
                continue
            joined.append({**case._asdict(),
                           'delivery_5_mean_pct': float(latest.deliverable_pct.mean()),
                           'delivery_20_mean_pct': float(previous.deliverable_pct.mean()),
                           'delivery_5_minus_20_pct': float(latest.deliverable_pct.mean()
                                                             - previous.deliverable_pct.mean()),
                           'traded_qty_5_to_20': float(np.log(recent_qty / base_qty)),
                           'latest_delivery_date': previous.iloc[-1].date.date().isoformat()})
    result = pd.DataFrame(joined)
    if not result.empty and not (
        pd.to_datetime(result.latest_delivery_date) < pd.to_datetime(result.decision_date)
    ).all():
        raise ValueError('Delivery input is not strictly before the decision session')
    audit['aligned_cases'] = len(result)
    audit['all_delivery_sessions_before_decision'] = True
    audit['aligned_by_year'] = {str(year): int(count) for year, count in result.groupby('year').size().items()}
    return result, audit


def paired_test(reference: np.ndarray, candidate: np.ndarray, outcome: np.ndarray) -> dict:
    ref_correct = (reference >= .5) == outcome
    new_correct = (candidate >= .5) == outcome
    improved = int((~ref_correct & new_correct).sum())
    worsened = int((ref_correct & ~new_correct).sum())
    discordant = improved + worsened
    return {'candidate_right_reference_wrong': improved,
            'reference_right_candidate_wrong': worsened,
            'exact_two_sided_p': round(float(binomtest(min(improved, worsened), discordant).pvalue), 4)
            if discordant else None}


def run() -> dict:
    data, coverage = assemble()
    columns = PRICE_FEATURES + DELIVERY_FEATURES
    # Missing values are excluded, not filled with guessed feature values.
    complete = data.dropna(subset=columns).copy()
    coverage['complete_feature_cases'] = len(complete)
    train = complete[complete.year.between(2021, 2023)]
    validation = complete[complete.year == 2024]
    later = complete[complete.year.isin([2025, 2026])]
    if min(len(train), len(validation), len(later)) < 20:
        raise RuntimeError('Insufficient complete exchange-observation cohort')
    candidates = []
    comparisons = {}
    probabilities = {}
    for name, features in (('price_only', PRICE_FEATURES),
                           ('price_plus_delivery', columns)):
        for regularization in (.01, .1, 1.0, 10.0):
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=regularization, max_iter=2000, random_state=42))
            model.fit(train[features], train.up)
            p = model.predict_proba(validation[features])[:, 1]
            candidates.append({'name': name, 'C': regularization,
                               'validation': score(validation, p)})
        selected = min((row for row in candidates if row['name'] == name),
                       key=lambda row: row['validation']['brier'])
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=selected['C'], max_iter=2000, random_state=42))
        model.fit(complete[complete.year.between(2021, 2024)][features],
                  complete[complete.year.between(2021, 2024)].up)
        p = model.predict_proba(later[features])[:, 1]
        probabilities[name] = p
        comparisons[name] = {'selected_C_on_2024': selected['C'],
                             'validation': selected['validation'],
                             'later': score(later, p),
                             'later_by_year': {str(year): score(later[later.year == year],
                                                            p[later.year.to_numpy() == year])
                                               for year in (2025, 2026)
                                               if (later.year == year).any()}}
    original = pd.concat([pd.read_csv(DEV_LEDGER), pd.read_csv(LATER_LEDGER)])
    original = original[['ipo_id', 'sell']].drop_duplicates('ipo_id')
    same_cohort = later.merge(original, on='ipo_id', how='left', validate='one_to_one')
    if same_cohort.sell.isna().any():
        raise ValueError('Deployed-monitor decisions missing from same-cohort comparison')
    monitor_correct = np.where(same_cohort.sell.astype(bool), ~later.up.astype(bool),
                               later.up.astype(bool))
    monitor_predict_up = ~same_cohort.sell.to_numpy(dtype=bool)
    ledger = later[['ipo_id', 'year', 'decision_date', 'latest_delivery_date',
                    'future_return_pct', 'up']].copy()
    ledger['deployed_monitor_sell'] = same_cohort.sell.to_numpy()
    ledger['price_only_probability'] = probabilities['price_only']
    ledger['price_plus_delivery_probability'] = probabilities['price_plus_delivery']
    ledger.to_csv(PREDICTIONS, index=False)
    result = {
        'protocol': 'NSE stock closes and prior-session delivery; 2021-23 fit, C selected on 2024 Brier, 2021-24 refit, exploratory 2025-26 same-cohort evaluation',
        'input_sha256': {str(path.relative_to(ROOT)): digest(path)
                         for path in (DEV_PRICE, LATER_PRICE, DEV_DELIVERY, LATER_DELIVERY,
                                      DEV_LEDGER, LATER_LEDGER)},
        'coverage': {**coverage, 'complete_by_year': {str(year): int(count)
                                                      for year, count in complete.groupby('year').size().items()}},
        'features': {'price_only': PRICE_FEATURES, 'delivery': DELIVERY_FEATURES},
        'candidates': candidates,
        'comparisons': comparisons,
        'paired_later': paired_test(probabilities['price_only'],
                                    probabilities['price_plus_delivery'],
                                    later.up.to_numpy(dtype=bool)),
        'deployed_monitor_same_cohort_accuracy_pct': round(float(monitor_correct.mean() * 100), 2),
        'paired_delivery_vs_deployed_monitor': paired_test(
            monitor_predict_up.astype(float), probabilities['price_plus_delivery'],
            later.up.to_numpy(dtype=bool)),
        'prediction_ledger': str(PREDICTIONS.relative_to(ROOT)),
        'limitations': [
            'NSE daily delivery observations were downloaded retrospectively; publication times are not archived.',
            'Sector and Nifty benchmarks in the original monitor still come from secondary history.',
            'Later-year outcomes were previously inspected, so this is exploratory, not a blind test.',
            'The deployed monitor and paper model are unchanged.',
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps({key: result[key] for key in
                      ('coverage', 'comparisons', 'paired_later',
                       'deployed_monitor_same_cohort_accuracy_pct')}, indent=2))
