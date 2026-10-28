"""Calibrate the existing monitor's thresholds, not a replacement model.

Only real NSE price records and the current-rule replay are used. All IPOs
remain in the later evaluation. Historical news is not fabricated or backfilled.
Development labels must have matured before the parameter-freeze date.
"""
from __future__ import annotations

import hashlib
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV_PRICES = ROOT/'data/audit/module2_development_prices_2021_2024_nse_first62.csv'
LATER_PRICES = ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full.csv'
DEV_LEDGER = ROOT/'output/module2_development_decision_ledger_goal_current_20261021.csv'
LATER_LEDGER = ROOT/'output/module2_holdout_decision_ledger_goal_current_20261021.csv'
OUTPUT = ROOT/'output/module2_same_rule_calibration_20261021.json'
DEFAULT = {'loss_pct': 12., 'lag_loss_pct': 5., 'trail_pct': 8.}
GRID = [dict(zip(DEFAULT, values)) for values in itertools.product(
    (5., 8., 12., 18., 25.), (0., 5., 10.), (4., 8., 12., 18., 25.))]


def assemble(prices_path, ledger_path):
    prices = pd.read_csv(prices_path, parse_dates=['date'])
    if not prices.price_basis.str.startswith('Official NSE').all():
        raise ValueError('Calibration requires original exchange stock closes')
    if prices.duplicated(['ipo_id', 'date']).any():
        raise ValueError('Duplicate price observations')
    ledger = pd.read_csv(ledger_path, parse_dates=['decision_date', 'outcome_date'])
    if ledger.ipo_id.duplicated().any():
        raise ValueError('One IPO must contribute only one checkpoint')
    ipos = pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    series = {int(key): group.sort_values('date').reset_index(drop=True)
              for key, group in prices.groupby('ipo_id')}
    peaks, issues, actual_end_dates = [], [], []
    for row in ledger.itertuples():
        data = series[row.ipo_id]
        if len(data) < 62 or data.iloc[30].date != row.decision_date:
            raise ValueError('The replay checkpoint is not the 31st observed close')
        if not np.isclose(data.iloc[30].close_price_adjusted, row.decision_close, atol=1e-8):
            raise ValueError('The decision price disagrees with the exchange input')
        actual = (data.iloc[60].close_price_adjusted / data.iloc[30].close_price_adjusted - 1)*100
        if not np.isclose(actual, row.day30_return_pct, atol=1e-8):
            raise ValueError('The outcome target differs from the frozen 30-session horizon')
        if data.iloc[61].date != row.outcome_date:
            raise ValueError('Execution outcome date differs from the frozen replay')
        peaks.append(float(data.iloc[:31].close_price_adjusted.max()))
        issues.append(float(ipos.loc[row.ipo_id].issue_price))
        actual_end_dates.append(data.iloc[60].date)
    ledger['peak_close'] = peaks
    ledger['issue_price'] = issues
    ledger['direction_outcome_date'] = actual_end_dates
    if not np.isfinite(ledger[['peak_close', 'issue_price', 'decision_close',
                               'cumulative_gain_pct', 'day30_return_pct']].to_numpy()).all():
        raise ValueError('Missing observations cannot be imputed for this experiment')
    # Exact default-rule parity must pass before interpreting any calibration.
    predicted = decisions(ledger, DEFAULT)
    recorded = ledger.sell.to_numpy(dtype=bool)
    if not np.array_equal(predicted, recorded):
        bad = ledger.loc[predicted != recorded, 'ipo_id'].tolist()
        raise ValueError(f'Vector replay differs from the unchanged core rule for IPOs {bad}')
    return ledger


def decisions(frame, parameters):
    loss, lag_loss, trail = (parameters[k] for k in DEFAULT)
    cumulative = frame.cumulative_gain_pct.to_numpy()
    current, issue, peak = (frame[k].to_numpy() for k in ('decision_close', 'issue_price', 'peak_close'))
    stock_lag = frame.attribution_category.eq('STOCK_SPECIFIC_LAG').to_numpy()
    # Missing sector spread remains unavailable, never filled with zero.
    lag = stock_lag & frame.spread_vs_sector_30d_pct.le(-6).to_numpy()
    return ((cumulative <= -loss) | (current < issue*(1-loss/100))
            | ((cumulative <= -lag_loss) & lag)
            | ((current >= issue) & (current <= peak*(1-trail/100))))


def metrics(frame, sell):
    y = frame.day30_return_pct.to_numpy() > 0
    correct = np.asarray(sell) != y
    positive = y.sum(); negative = (~y).sum()
    hold_recall = float((~sell & y).sum()/positive) if positive else None
    sell_recall = float((sell & ~y).sum()/negative) if negative else None
    return {'n': len(frame), 'correct': int(correct.sum()),
            'accuracy_pct': float(correct.mean()*100),
            'balanced_accuracy_pct': (hold_recall+sell_recall)*50 if positive and negative else None,
            'hold_recall_pct': hold_recall*100 if positive else None,
            'sell_recall_pct': sell_recall*100 if negative else None,
            'always_hold_accuracy_pct': float(y.mean()*100),
            'always_sell_accuracy_pct': float((~y).mean()*100),
            'sell_calls': int(np.asarray(sell).sum()),
            'next_day_accuracy_pct': float((np.asarray(sell) != (frame.day1_return_pct.to_numpy()>0)).mean()*100),
            'conditional_strategy_net_mean_pct': float(np.where(sell, -.3,
                frame.executed_hold_return_pct.to_numpy()).mean())}


def select_parameters(known, freeze_date):
    if known.empty or not known.outcome_date.le(pd.Timestamp(freeze_date)).all():
        raise ValueError('A parameter choice cannot see an outcome after its freeze date')
    if not known.listing_year.between(2021, 2024).all():
        raise ValueError('Post-2024 IPOs cannot choose monitor settings')
    candidates = []
    for p in GRID:
        candidates.append({'parameters': p, 'metrics': metrics(known, decisions(known, p))})
    # Fixed class-balanced objective avoids improving by issuing one class only.
    winner = max(candidates, key=lambda r: (
        r['metrics']['balanced_accuracy_pct'], r['metrics']['accuracy_pct'],
        -sum(abs(r['parameters'][k]-DEFAULT[k]) for k in DEFAULT)))
    return winner, candidates


def paired_interval(frame, challenger):
    truth = frame.day30_return_pct.to_numpy() > 0
    old = decisions(frame, DEFAULT) != truth
    new = challenger != truth
    delta = new.astype(float)-old.astype(float)
    rng = np.random.default_rng(42)
    boot = [rng.choice(delta, len(delta), replace=True).mean()*100 for _ in range(5000)]
    return {'accuracy_change_pp': float(delta.mean()*100),
            'bootstrap_95_pp': np.quantile(boot, [.025, .975]).tolist(),
            'corrected_errors': int((new & ~old).sum()),
            'introduced_errors': int((old & ~new).sum())}


def run(dev_prices=DEV_PRICES, dev_ledger=DEV_LEDGER, output=OUTPUT,
        later_prices=LATER_PRICES, later_ledger=LATER_LEDGER):
    dev_prices, dev_ledger, output = (Path(p).resolve() for p in (dev_prices, dev_ledger, output))
    later_prices, later_ledger = (Path(p).resolve() for p in (later_prices, later_ledger))
    dev = assemble(dev_prices, dev_ledger)
    later = assemble(later_prices, later_ledger)
    confirmation = dev[dev.listing_year.eq(2024)].copy()
    confirmation_start = confirmation.decision_date.min()
    early = dev[dev.listing_year.isin([2022, 2023]) & dev.outcome_date.lt(confirmation_start)]
    chosen_early, early_candidates = select_parameters(early, confirmation_start-pd.Timedelta(days=1))
    # Freeze the final thresholds at the end of 2024, not after seeing a late
    # 2024 IPO's future outcomes in 2025.
    final_known = dev[dev.outcome_date.le('2024-12-31')]
    chosen_final, final_candidates = select_parameters(final_known, '2024-12-31')
    if not later.decision_date.gt('2024-12-31').all():
        raise ValueError('Later evaluation overlaps the final freeze date')
    confirmation_calls = decisions(confirmation, chosen_early['parameters'])
    later_calls = decisions(later, chosen_final['parameters'])
    summary = metrics(later, later_calls)
    # Both the requested accuracy and the class/base-rate checks matter. No
    # subgroup, horizon change, or abstention is counted as reaching the goal.
    target_met = bool(75 <= summary['accuracy_pct'] <= 100
                      and summary['balanced_accuracy_pct'] >= 70
                      and summary['accuracy_pct'] > max(summary['always_hold_accuracy_pct'],
                                                       summary['always_sell_accuracy_pct']))
    result = {'model_family': 'Existing deterministic Module 2 loss/relative-lag/trailing-stop rules',
              'core_rule_changed': False, 'synthetic_training_rows': 0,
              'primary_target': 'sign of the return over 30 later trading sessions; all eligible IPOs',
              'source': 'Real official NSE stock closes; preexisting dated market/sector observations',
              'coverage': {'development_n': len(dev), 'selection_2022_2023_matured_n': len(early),
                           'confirmation_2024_n': len(confirmation),
                           'known_by_2024_end_n': len(final_known), 'later_n': len(later),
                           'development_outcomes_not_mature_at_final_freeze': int((dev.outcome_date>'2024-12-31').sum())},
              'confirmation_first_decision': confirmation_start.strftime('%Y-%m-%d'),
              'early_selected': chosen_early, 'final_selected': chosen_final,
              'confirmation_2024': {'default': metrics(confirmation, decisions(confirmation, DEFAULT)),
                                    'calibrated': metrics(confirmation, confirmation_calls)},
              'later_2025_2026': {'default': metrics(later, decisions(later, DEFAULT)),
                                   'calibrated': summary, 'paired_change': paired_interval(later, later_calls)},
              'later_by_year': {str(year): metrics(group, decisions(group, chosen_final['parameters']))
                                 for year, group in later.groupby('listing_year')},
              'requested_accuracy_target_met': target_met,
              'candidate_count': len(GRID), 'early_candidates': early_candidates, 'final_candidates': final_candidates,
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (dev_prices, later_prices, dev_ledger, later_ledger,
                                         ROOT/'core/monitor_engine.py', ROOT/'core/attribution_engine.py')},
              'limitations': ['2025–2026 outcomes were inspected previously; not a blind test.',
                              'Price-only replay does not measure the full live news/disclosure monitor.',
                              ('Dated NSE bonus/split ratios reconciled; cash dividends and executable trading returns are not measured.'
                               if 'shares_per_listing_share' in pd.read_csv(later_prices,nrows=0).columns else
                               'Corporate actions are unreconciled; do not assert economic returns from raw price jumps.'),
                              'Changing only thresholds may not supply missing predictive information.',
                              'Bootstrap intervals are descriptive; market co-movement can reduce effective sample size.']}
    ledger = later.copy()
    ledger['calibrated_sell'] = later_calls
    ledger['calibrated_correct_30d'] = later_calls != (later.day30_return_pct.to_numpy()>0)
    ledger.to_csv(output.with_name(output.stem+'_ledger.csv'), index=False)
    output.write_text(json.dumps(result, indent=2, default=str)+'\n', encoding='utf-8')
    lines = ['# Module 2 — Same-model threshold calibration', '',
             'The paper’s deterministic monitor family is unchanged. This experiment calibrates only '
             'its existing loss, relative-lag loss and trailing-stop thresholds; it does not introduce a new classifier.', '',
             f"Data: {len(dev)} real exchange-price development cases; {len(later)} later cases. "
             'One 31st-close checkpoint and the same 30-session target per IPO. No synthetic IPOs, price interpolation, '
             'future news, post-2024 parameter fitting, excluded losing trades or changed horizons.', '',
             f"Final threshold freeze: 31 December 2024. {len(final_known)} IPO outcomes were known; "
             f"{result['coverage']['development_outcomes_not_mature_at_final_freeze']} development IPO outcomes "
             'had not matured and could not choose parameters.', '',
             '| Evaluation | Default accuracy | Calibrated accuracy | Always sell | Always hold |',
             '|---|---:|---:|---:|---:|']
    for name, key in [('2024 confirmation', 'confirmation_2024'), ('2025–2026 later replay', 'later_2025_2026')]:
        a,b=result[key]['default'],result[key]['calibrated']
        lines.append(f"| {name} (n={b['n']}) | {a['accuracy_pct']:.2f}% | {b['accuracy_pct']:.2f}% | {b['always_sell_accuracy_pct']:.2f}% | {b['always_hold_accuracy_pct']:.2f}% |")
    lines += ['', f"Selected final thresholds: {chosen_final['parameters']}. "
              f"Later balanced accuracy: {summary['balanced_accuracy_pct']:.2f}%. "
              f"Requested 75–80% accuracy established: {'yes on this retrospective replay only' if target_met else 'NO'}.", '',
              'The goal remains incomplete unless both modules satisfy their own full, timing-correct evidence requirements. '
              'An improvement below the requested target is progress, not success. The live monitor and paper are not changed.', '',
              '## Limitations', ''] + ['- '+x for x in result['limitations']]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-prices',type=Path,default=DEV_PRICES)
    parser.add_argument('--development-ledger',type=Path,default=DEV_LEDGER)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--later-prices',type=Path,default=LATER_PRICES)
    parser.add_argument('--later-ledger',type=Path,default=LATER_LEDGER)
    args=parser.parse_args()
    report = run(args.development_prices,args.development_ledger,args.output,args.later_prices,args.later_ledger)
    print(json.dumps({k: report[k] for k in ('coverage','early_selected','final_selected',
        'confirmation_2024','later_2025_2026','later_by_year','requested_accuracy_target_met')}, indent=2))
