"""Research-only audit of the issue-price recovery veto; no live rule edits.

Same real filing clocks, daily prices, cohort and 30-session target as the
source-response study. Parameters are selected on matured 2021-2024 IPOs.
Later results are exploratory because this cohort has already been inspected.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, decisions, metrics, paired_interval
from ml.replay_module2_financial_rules import EVIDENCE, load_evidence
from ml.replay_module2_news_overlay import SOURCES
from ml.research_module2_filing_response import GRID, assemble_response, response_calls

REPORT = ROOT / 'output/module2_recovery_guard_research_20261022.json'
CANDIDATES = [dict(p, permit_below_issue_recovery=False) for p in GRID] + [
    dict(p, permit_below_issue_recovery=True) for p in GRID
    if p['mode'] in ('positive_only', 'both')]


def calls(frame, parameters):
    result = response_calls(frame, parameters)
    if parameters['permit_below_issue_recovery']:
        recovery = (frame.filing_response_status.eq('verified_daily_source_clock_response') &
                    frame.source_publication_age_days.between(0, parameters['max_publication_age_days']) &
                    frame.stock_response_since_source_pct.gt(parameters['response_confirm_abs_pct']) &
                    frame.excess_response_since_source_pct.gt(parameters['response_confirm_abs_pct']))
        result[recovery.to_numpy()] = False
    return result


def select(frame, freeze):
    if (frame.empty or not frame.listing_year.between(2021, 2024).all() or
            not frame.outcome_date.le(pd.Timestamp(freeze)).all()):
        raise ValueError('Selection requires matured 2021-2024 IPO outcomes')
    baseline = decisions(frame, DEFAULT)
    candidates = [dict(parameters=p, metrics=metrics(frame, calls(frame, p)),
                       changed_calls=int((calls(frame, p) != baseline).sum())) for p in CANDIDATES]
    winner = max(candidates, key=lambda r: (r['metrics']['balanced_accuracy_pct'],
                 r['metrics']['accuracy_pct'], -r['changed_calls'],
                 not r['parameters']['permit_below_issue_recovery']))
    return winner, candidates


def run():
    evidence = load_evidence()
    development, later = [assemble_response(name, evidence) for name in ('development', 'holdout')]
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    prediction = calls(later, choice['parameters'])
    baseline = decisions(later, DEFAULT)
    ledger = later.copy()
    ledger['research_sell'] = prediction
    ledger['research_correct'] = prediction != later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem + '_ledger.csv'), index=False)
    # Counterfactual is a fixed matching-rule ablation, not later-year selection.
    counterpart = dict(choice['parameters'], permit_below_issue_recovery=not
                       choice['parameters']['permit_below_issue_recovery'])
    report = dict(live_core_changed=False, new_learned_classifier=False, synthetic_training_rows=0,
        protocol='Same 132 IPOs, 31st-close decision and 30-later-session direction; no abstentions.',
        selection_n=len(known), candidates=candidates, selected_on_earlier_data=choice,
        later=dict(default=metrics(later, baseline), research=metrics(later, prediction),
                   paired_change=paired_interval(later, prediction),
                   matching_rule_with_opposite_issue_guard=metrics(later, calls(later, counterpart))),
        limitations=['Exploratory: later cohort was already inspected, not an untouched test.',
            'Positive observed response is not an earnings surprise or proof of causal predictive information.',
            'Relaxing a capital-protection veto can increase losses; accuracy alone cannot authorize deployment.',
            'No live engine, manuscript result, cohort or decision horizon is changed.'],
        input_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (EVIDENCE, *SOURCES['development'], *SOURCES['holdout'],
                      ROOT/'ml/research_module2_filing_response.py',
                      ROOT/'ml/research_module2_recovery_guard.py', ROOT/'core/monitor_engine.py')})
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    score = report['later']['research']
    REPORT.with_suffix('.md').write_text('\n'.join([
        '# Module 2: does the issue-price veto prevent useful recovery calls?', '', report['protocol'], '',
        f"Earlier-only selected rule: `{choice['parameters']}`.", '',
        f"Direction accuracy: {score['accuracy_pct']:.2f}%; balanced accuracy: {score['balanced_accuracy_pct']:.2f}%.",
        f"Always-sell baseline: {score['always_sell_accuracy_pct']:.2f}%.", '',
        *['- '+s for s in report['limitations']]])+'\n', encoding='utf-8')
    print(json.dumps(dict(selected=choice, later=report['later']), indent=2))
    return report


if __name__ == '__main__':
    run()
