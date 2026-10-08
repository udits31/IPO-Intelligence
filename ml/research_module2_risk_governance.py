"""Audit threshold reachability, then test a bounded macro risk-off branch.

The monitor remains deterministic: no new learned classifier, synthetic prices,
changed target, or silent live promotion. Hindsight bounds are diagnostics only
and never supply features or select a per-IPO trading action.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, GRID as PRICE_GRID, assemble, decisions, metrics, paired_interval
from ml.replay_module2_financial_rules import TIGHTER, PROFIT_TRAIL
from ml.replay_module2_news_overlay import SOURCES

REPORT = ROOT / 'output/module2_risk_governance_research_20261022.json'
GRID = [{'market_window': None, 'market_max_pct': None, 'stock_max_pct': None}] + [
    dict(zip(('market_window', 'market_max_pct', 'stock_max_pct'), values))
    for values in itertools.product((10, 30), (-3., 0.), (-3., 0.))]


def momentum_features(prices, checkpoint):
    history = prices[prices.date.le(checkpoint)].sort_values('date')
    if len(history) != 31 or history.iloc[-1].date != checkpoint:
        raise ValueError('Momentum inputs require exactly the 31 observed pre-decision closes')
    result = {}
    for label, column in [('stock', 'close_price_adjusted'), ('nifty', 'nifty50_close')]:
        for lag in (10, 30):
            first, last = history.iloc[-lag-1][column], history.iloc[-1][column]
            result[label + '_' + str(lag) + '_pct'] = (
                float((last / first - 1) * 100) if pd.notna(first) and pd.notna(last) and first > 0 else np.nan)
    return result


def assemble_risk(name):
    prices_path, ledger_path = SOURCES[name]
    frame = assemble(prices_path, ledger_path)
    prices = pd.read_csv(prices_path, parse_dates=['date'])
    groups = {key: group for key, group in prices.groupby('ipo_id')}
    features = pd.DataFrame([momentum_features(groups[row.ipo_id], row.decision_date)
                             for row in frame.itertuples()], index=frame.index)
    # The rule search cannot quietly replace the existing observed 30-session
    # market return with a present-day or different-window observation.
    for source, saved in [('stock_30_pct', 'stock_return_30d_pct'), ('nifty_30_pct', 'nifty_return_30d_pct')]:
        observed = features[source].notna() & frame[saved].notna()
        if not np.allclose(features.loc[observed, source], frame.loc[observed, saved], atol=.0051):
            raise ValueError('Observed momentum disagrees with the frozen replay')
    return frame.join(features)


def risk_calls(frame, parameters):
    result = decisions(frame, DEFAULT).copy()
    if parameters['market_window'] is None:
        return result
    market = frame['nifty_' + str(parameters['market_window']) + '_pct']
    # A new experimental risk-governance branch, not a different classifier:
    # both the market and the stock must show observed negative momentum.
    # Unknown market history cannot be filled with a zero or an invented trend.
    risk_off = (market.notna() & frame.stock_10_pct.notna() &
                market.lt(parameters['market_max_pct']) & frame.stock_10_pct.lt(parameters['stock_max_pct']))
    result[risk_off.to_numpy()] = True
    return result


def select(frame, freeze):
    if (frame.empty or not frame.listing_year.between(2021, 2024).all() or
            not frame.outcome_date.le(pd.Timestamp(freeze)).all()):
        raise ValueError('Risk-governance selection requires matured 2021-2024 outcomes only')
    baseline = decisions(frame, DEFAULT)
    candidates = [{'parameters': parameters, 'metrics': metrics(frame, risk_calls(frame, parameters)),
                   'changed_calls': int((risk_calls(frame, parameters) != baseline).sum())}
                  for parameters in GRID]
    choice = max(candidates, key=lambda row: (row['metrics']['balanced_accuracy_pct'],
                                             row['metrics']['accuracy_pct'], -row['changed_calls']))
    return choice, candidates


def hindsight_bound(frame, presets):
    actual_up = frame.day30_return_pct.to_numpy() > 0
    choices = np.vstack([decisions(frame, parameters) for parameters in presets])
    reachable = np.any(choices != actual_up, axis=0)
    return {'settings': len(presets), 'n': len(frame), 'hindsight_ceiling_correct': int(reachable.sum()),
            'hindsight_ceiling_pct': float(reachable.mean() * 100),
            'unreachable_case_ids': frame.loc[~reachable, 'ipo_id'].astype(int).tolist(),
            'uses_future_outcomes_for_diagnostic_only': True,
            'not_prediction_accuracy': True}, reachable


def run():
    development, later = [assemble_risk(name) for name in ('development', 'holdout')]
    confirmation = development[development.listing_year.eq(2024)]
    first = confirmation.decision_date.min()
    early = development[development.listing_year.isin([2022, 2023]) & development.outcome_date.lt(first)]
    early_choice, early_candidates = select(early, first - pd.Timedelta(days=1))
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    calls = risk_calls(later, choice['parameters'])
    score = metrics(later, calls)
    three, reachable_three = hindsight_bound(later, [DEFAULT, TIGHTER, PROFIT_TRAIL])
    all_settings, reachable_all = hindsight_bound(later, PRICE_GRID)
    ledger = later.copy()
    ledger['experimental_risk_off_sell'] = calls
    ledger['experimental_risk_off_correct'] = calls != later.day30_return_pct.gt(0).to_numpy()
    ledger['hindsight_reachable_three_presets_NOT_A_FORECAST'] = reachable_three
    ledger['hindsight_reachable_75_settings_NOT_A_FORECAST'] = reachable_all
    ledger.to_csv(REPORT.with_name(REPORT.stem + '_ledger.csv'), index=False)
    report = {'model_family': 'Deterministic monitor, experimental macro/momentum risk-governance branch',
              'live_core_changed': False, 'new_learned_classifier': False, 'synthetic_training_rows': 0,
              'protocol': 'Same 31st-close checkpoint, next-close execution and 30-later-session direction; all 132 later cases retained.',
              'source': 'Archived official NSE stock closes and dated secondary-provider Nifty index closes; no present-day backfill.',
              'candidate_count': len(GRID), 'coverage': {'development': len(development), 'later': len(later),
                  'matured_before_2025_selection': len(known),
                  'later_missing_10_session_nifty_retained_on_default': int(later.nifty_10_pct.isna().sum())},
              'hindsight_diagnostics_not_model_scores': {'three_earnings_presets': three, 'full_75_price_settings': all_settings},
              'early_choice': early_choice, 'early_candidates': early_candidates,
              'selected_on_earlier_data': choice, 'final_candidates': candidates,
              'confirmation_2024': {'default': metrics(confirmation, decisions(confirmation, DEFAULT)),
                                    'experimental': metrics(confirmation, risk_calls(confirmation, early_choice['parameters']))},
              'later': {'default': metrics(later, decisions(later, DEFAULT)), 'experimental': score,
                        'paired_change': paired_interval(later, calls)},
              'requested_accuracy_established': bool(score['accuracy_pct'] >= 75 and score['balanced_accuracy_pct'] >= 70 and
                  score['accuracy_pct'] > max(score['always_hold_accuracy_pct'], score['always_sell_accuracy_pct'])),
              'input_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (*SOURCES['development'], *SOURCES['holdout'],
                                            ROOT / 'core/monitor_engine.py', ROOT / 'core/attribution_engine.py',
                                            ROOT / 'ml/calibrate_module2_rules.py',
                                            ROOT / 'ml/research_module2_risk_governance.py')},
              'limitations': ['Hindsight ceilings use later outcomes only to falsify insufficient rule ranges; they are not achievable scores and never choose actions.',
                              'A limit below 75% applies only to the specified tested thresholds, not to every possible deterministic monitor.',
                              'Later outcomes have been inspected repeatedly; this is exploratory, not blind or prospective validation.',
                              'New risk-off logic is research-only and is not silently written into the deployed monitor.',
                              'Nifty history is a retrospective secondary-provider series, not independently immutable as-published exchange records.',
                              'Direction accuracy is not a guaranteed executable return or investment recommendation.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Module 2: what limits the current accuracy', '',
             'All 132 later IPOs retain the same checkpoint and 30-session target. No live policy is changed.', '',
             '## Measured experiment', '',
             '| Policy | Actual later accuracy |', '|---|---:|',
             f"| Current monitor | {report['later']['default']['accuracy_pct']:.2f}% |",
             f"| Earlier-data-selected macro/momentum branch | {score['accuracy_pct']:.2f}% |",
             f"| Always sell | {score['always_sell_accuracy_pct']:.2f}% |", '',
             'The new branch did not improve the score and is not promoted.', '',
             '## Why threshold tuning alone is insufficient', '',
             f"The three presets used by the initial earnings experiment can be right on at most {three['hindsight_ceiling_correct']} "
             f"of {len(later)} cases ({three['hindsight_ceiling_pct']:.2f}%), even if a hindsight oracle chooses the best preset for every IPO.", '',
             f"The broader, fixed set of {all_settings['settings']} price-threshold combinations can be right on at most "
             f"{all_settings['hindsight_ceiling_correct']} cases ({all_settings['hindsight_ceiling_pct']:.2f}%) under the same unrealistic assignment.", '',
             'These are impossibility diagnostics for those specified rule ranges, NOT achieved model accuracy, forecasts or an upper bound on all possible rule-based systems. '
             'No hindsight assignment enters prediction features or earlier-data parameter selection.', '',
             'The 75-80% objective therefore needs genuinely useful evidence-linked decisions beyond those tested threshold ranges. '
             'More source rows alone cannot remove this structural limitation. Real earnings/news collection remains useful for testing such decisions, '
             'but coverage must not be advertised as predictive improvement.', '', '## Limitations', '']
    lines.extend('- ' + limitation for limitation in report['limitations'])
    REPORT.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('coverage', 'hindsight_diagnostics_not_model_scores',
                                                 'selected_on_earlier_data', 'confirmation_2024', 'later',
                                                 'requested_accuracy_established')}, indent=2))
    return report


if __name__ == '__main__':
    run()
