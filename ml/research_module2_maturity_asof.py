"""Allow late-2024 outcomes only after maturity, never train on later IPOs.

This is a separately labelled expanding-availability deployment experiment,
not a rewrite of the frozen-2024 result. The candidate rule families and
selection objectives are unchanged; all 132 later IPOs remain test cases.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, decisions, metrics, paired_interval, select_parameters
from ml.replay_module2_financial_rules import EVIDENCE, load_evidence
from ml.replay_module2_news_overlay import SOURCES
from ml.research_module2_filing_response import assemble_response, response_calls, select as select_response

REPORT = ROOT / 'output/module2_maturity_asof_research_20261022.json'
FAMILIES = {'price_thresholds': (select_parameters, decisions),
            'filing_clock_response': (select_response, response_calls)}


def available_development(development, decision_date):
    if not development.listing_year.between(2021, 2024).all() or development.ipo_id.duplicated().any():
        raise ValueError('Only unique 2021-2024 IPOs can enter the training pool')
    # A prior calendar-day close is conservative about when the outcome was
    # available. No same-close batch fit or future execution quote is assumed.
    return development[development.outcome_date.lt(pd.Timestamp(decision_date).normalize())]


def asof_calls(development, later, family):
    if family not in FAMILIES or not later.listing_year.ge(2025).all():
        raise ValueError('Unknown family or non-later evaluation case')
    if set(development.ipo_id) & set(later.ipo_id):
        raise ValueError('Training and evaluation IPO identities overlap')
    selector, caller = FAMILIES[family]
    result = np.zeros(len(later), dtype=bool)
    trace = []
    cache = {}
    for position, case in enumerate(later.itertuples()):
        known = available_development(development, case.decision_date)
        if known.empty:
            raise ValueError('No matured earlier outcomes are available')
        ids = tuple(sorted(int(value) for value in known.ipo_id))
        if ids not in cache:
            choice, _ = selector(known, case.decision_date - pd.Timedelta(nanoseconds=1))
            cache[ids] = choice
        choice = cache[ids]
        result[position] = caller(later.iloc[[position]], choice['parameters'])[0]
        trace.append({'ipo_id': int(case.ipo_id), 'decision_date': str(case.decision_date.date()),
                      'training_n': len(known), 'training_years': sorted(int(value) for value in known.listing_year.unique()),
                      'latest_training_execution_outcome_date': str(known.outcome_date.max().date()),
                      'training_ipo_ids': list(ids), 'parameters': choice['parameters']})
    return result, trace


def run():
    evidence = load_evidence()
    development, later = [assemble_response(name, evidence) for name in ('development', 'holdout')]
    frozen = development[development.outcome_date.le('2024-12-31')]
    results = {}
    ledger = later.copy()
    for name, (selector, caller) in FAMILIES.items():
        fixed, _ = selector(frozen, '2024-12-31')
        fixed_calls = caller(later, fixed['parameters'])
        calls, trace = asof_calls(development, later, name)
        score = metrics(later, calls)
        ledger[name + '_fixed_2024_sell'] = fixed_calls
        ledger[name + '_asof_2024_ipo_outcomes_sell'] = calls
        results[name] = {'frozen_2024_choice': fixed, 'frozen_2024_later_metrics': metrics(later, fixed_calls),
                         'asof_later_metrics': score, 'paired_change_vs_live_default': paired_interval(later, calls),
                         'changed_vs_frozen_2024_calls': int((calls != fixed_calls).sum()),
                         'training_size_min': min(row['training_n'] for row in trace),
                         'training_size_max': max(row['training_n'] for row in trace), 'per_case_training_trace': trace}
    ledger.to_csv(REPORT.with_name(REPORT.stem + '_ledger.csv'), index=False)
    report = {'model_family': 'Same deterministic rule families; expanding availability of 2021-2024 IPO outcomes only',
              'new_learned_classifier': False, 'live_core_changed': False, 'synthetic_training_rows': 0,
              'protocol': 'Same 31st-close checkpoint and 30-session target, all 132 later cases. Only earlier-IPO outcomes completed before each test decision can select settings.',
              'training_ipo_years': [2021, 2022, 2023, 2024], 'post2024_ipo_training_rows': 0,
              'coverage': {'development': len(development), 'known_by_2024_end': len(frozen), 'later': len(later)},
              'later_default_metrics': metrics(later, decisions(later, DEFAULT)), 'families': results,
              'requested_accuracy_established': any(item['asof_later_metrics']['accuracy_pct'] >= 75 and
                  item['asof_later_metrics']['balanced_accuracy_pct'] >= 70 and
                  item['asof_later_metrics']['accuracy_pct'] > max(item['asof_later_metrics']['always_hold_accuracy_pct'],
                                                               item['asof_later_metrics']['always_sell_accuracy_pct'])
                  for item in results.values()),
              'input_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (EVIDENCE, *SOURCES['development'], *SOURCES['holdout'],
                                            ROOT / 'core/monitor_engine.py', ROOT / 'core/attribution_engine.py',
                                            ROOT / 'ml/calibrate_module2_rules.py', ROOT / 'ml/research_module2_filing_response.py',
                                            ROOT / 'ml/research_module2_maturity_asof.py')},
              'limitations': ['This differs from a once-frozen end-2024 model and is reported separately, not substituted into its old score.',
                              'The extra labels belong only to late-2024 IPOs; later 2025-2026 test outcomes never select a setting.',
                              'Retrospectively retrieved financial clocks and prices are not immutable original deployment captures.',
                              'Later test outcomes have been inspected in prior research; this remains exploratory, not blind or prospective.',
                              'Neither branch is automatically promoted to the application or paper.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    text = ['# Module 2: training outcomes admitted only after they mature', '', report['protocol'], '',
            '| Same 132 later IPOs | Direction accuracy |', '|---|---:|',
            f"| Live-rule baseline | {report['later_default_metrics']['accuracy_pct']:.2f}% |"]
    for name, item in results.items():
        text.extend([f"| {name}: frozen end-2024 | {item['frozen_2024_later_metrics']['accuracy_pct']:.2f}% |",
                     f"| {name}: as-of completed 2024 IPO outcomes | {item['asof_later_metrics']['accuracy_pct']:.2f}% |"])
    text.extend(['', 'No 2025-2026 IPO trains either policy. Their outcomes remain evaluation-only. '
                 'This is a separate expanding-availability protocol, not a changed test cohort or horizon.', '', '## Limitations', ''])
    text.extend('- ' + limitation for limitation in report['limitations'])
    REPORT.with_suffix('.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({'coverage': report['coverage'], 'families': {
        name: {key: item[key] for key in ('frozen_2024_later_metrics', 'asof_later_metrics',
                                        'changed_vs_frozen_2024_calls', 'training_size_min', 'training_size_max')}
        for name, item in results.items()}, 'requested_accuracy_established': report['requested_accuracy_established']}, indent=2))
    return report


if __name__ == '__main__':
    run()
