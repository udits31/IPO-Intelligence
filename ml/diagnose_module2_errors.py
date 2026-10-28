"""Describe existing replay failures without fitting more settings or collecting PDFs.

Outcome-based fields below are diagnostics only, never prediction inputs.
All original cases and the established 30-session target remain unchanged.
"""
import hashlib
import json

import numpy as np

from ml.calibrate_module2_rules import ROOT, DEFAULT, assemble, decisions, metrics
from ml.replay_module2_news_overlay import SOURCES

REPORT = ROOT / 'output/module2_error_diagnosis_20261022.json'


def classify(frame):
    """Mutually exclusive explanations of current-rule calls using known inputs."""
    structural = (frame.cumulative_gain_pct.le(-12) |
                  frame.decision_close.lt(frame.issue_price * .88))
    lag = (frame.cumulative_gain_pct.le(-5) &
           frame.attribution_category.eq('STOCK_SPECIFIC_LAG') &
           frame.spread_vs_sector_30d_pct.le(-6))
    trail = frame.decision_close.ge(frame.issue_price) & frame.decision_close.le(frame.peak_close * .92)
    reasons = np.select([structural, lag, trail],
                        ['loss_stop', 'relative_lag_exit', 'trailing_exit'], default='hold')
    if not np.array_equal(reasons != 'hold', decisions(frame, DEFAULT)):
        raise ValueError('Diagnostic explanation differs from actual default decisions')
    return reasons


def describe(frame):
    reasons = classify(frame)
    prediction = decisions(frame, DEFAULT)
    up = frame.day30_return_pct.gt(0).to_numpy()
    rows = []
    for reason in ('loss_stop', 'relative_lag_exit', 'trailing_exit', 'hold'):
        subset = frame[reasons == reason]
        if subset.empty:
            continue
        score = metrics(subset, decisions(subset, DEFAULT))
        rows.append(dict(reason=reason, **score,
                         median_future_return_pct=float(subset.day30_return_pct.median())))
    failed = prediction == up
    returns = frame.day30_return_pct.to_numpy()
    return dict(metrics=metrics(frame, prediction), reason_groups=rows,
                false_sell_recoveries=int((prediction & up).sum()),
                false_hold_declines=int((~prediction & ~up).sum()),
                errors_with_abs_future_return_at_most_5pct=int((failed & (np.abs(returns) <= 5)).sum()),
                errors_with_abs_future_return_above_10pct=int((failed & (np.abs(returns) > 10)).sum()),
                total_errors=int(failed.sum()),
                years={str(year): metrics(group, decisions(group, DEFAULT))
                       for year, group in frame.groupby('listing_year')})


def run():
    development, later = [assemble(*SOURCES[name]) for name in ('development', 'holdout')]
    known = development[development.outcome_date.le('2024-12-31')]
    if len(known) != 214 or len(later) != 132:
        raise ValueError('The established cohort changed; inspect rather than relabel results')
    report = dict(new_parameters_selected=False, live_core_changed=False, synthetic_rows=0,
                  earlier_matured=describe(known), later=describe(later),
                  limitations=['Descriptive after-outcome audit, not a new forecast or blind validation.',
                    'Reason-group rates are not probabilities for new IPOs; small groups are unstable.',
                    'No alternate horizon, flat-price exclusion, abstention or cohort filtering improves the quoted score.'],
                  input_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (*SOURCES['development'], *SOURCES['holdout'], ROOT/'core/monitor_engine.py',
                              ROOT/'ml/calibrate_module2_rules.py', ROOT/'ml/diagnose_module2_errors.py')})
    ledger = later.copy()
    ledger['diagnostic_rule_reason'] = classify(later)
    ledger['diagnostic_wrong'] = decisions(later, DEFAULT) == later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem + '_ledger.csv'), index=False)
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Why Module 2 is not improving', '',
             'No new model, thresholds, PDF extraction or data acquisition. Same primary evaluation.', '',
             '| Rule branch | Earlier n | Earlier accuracy | Later n | Later accuracy |',
             '|---|---:|---:|---:|---:|']
    early = {r['reason']: r for r in report['earlier_matured']['reason_groups']}
    for row in report['later']['reason_groups']:
        previous = early.get(row['reason'])
        lines.append(f"| {row['reason']} | {previous['n'] if previous else 0} | " +
            (f"{previous['accuracy_pct']:.1f}%" if previous else 'Unavailable') +
            f" | {row['n']} | {row['accuracy_pct']:.1f}% |")
    later_stats = report['later']
    lines += ['', f"Later errors: {later_stats['false_sell_recoveries']} sell calls followed by a rise, "
              f"and {later_stats['false_hold_declines']} hold calls followed by a flat/down outcome.",
              f"Only {later_stats['errors_with_abs_future_return_at_most_5pct']} of "
              f"{later_stats['total_errors']} errors have an absolute subsequent move at most 5%; "
              f"{later_stats['errors_with_abs_future_return_above_10pct']} have moves above 10%.", '',
              'These are explanatory outcome diagnostics, not selectable trading features or new success criteria.', '',
              *['- '+item for item in report['limitations']]]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('earlier_matured', 'later')}, indent=2))
    return report


if __name__ == '__main__':
    run()
