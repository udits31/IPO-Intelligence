"""Require observed strength for HOLD; stop at earlier-year validation if weak.

Bounded deterministic governance, not another model family or price source.
No PDF extraction, added acquisition, synthetic rows or live engine edits.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, decisions, metrics, paired_interval
from ml.replay_module2_news_overlay import SOURCES
from ml.research_module2_risk_governance import assemble_risk

REPORT = ROOT / 'output/module2_hold_evidence_research_20261022.json'
MODES = ('unchanged', 'positive_10_session_stock', 'positive_30_session_stock',
         'positive_10_session_stock_and_market_relative')


def calls(frame, mode):
    result = decisions(frame, DEFAULT).copy()
    if mode not in MODES:
        raise ValueError('Unknown predeclared HOLD-evidence policy')
    if mode == 'unchanged':
        return result
    column = 'stock_30_pct' if mode == 'positive_30_session_stock' else 'stock_10_pct'
    observed = frame[column].notna()
    positive = frame[column].gt(0)
    if mode == 'positive_10_session_stock_and_market_relative':
        observed &= frame.nifty_10_pct.notna()
        positive &= frame.stock_10_pct.gt(frame.nifty_10_pct)
    # Unknown history retains the default action, never zero-filled strength.
    result[(observed & ~positive).to_numpy()] = True
    return result


def select(frame, freeze):
    if (frame.empty or not frame.listing_year.between(2021, 2023).all() or
            not frame.outcome_date.lt(pd.Timestamp(freeze)).all()):
        raise ValueError('Select only matured 2021-2023 outcomes before confirmation')
    original = decisions(frame, DEFAULT)
    candidates = [dict(mode=mode, metrics=metrics(frame, calls(frame, mode)),
                       changed_calls=int((calls(frame, mode) != original).sum())) for mode in MODES]
    winner = max(candidates, key=lambda r: (r['metrics']['balanced_accuracy_pct'],
                                          r['metrics']['accuracy_pct'], -r['changed_calls']))
    return winner, candidates


def run():
    development = assemble_risk('development')
    confirmation = development[development.listing_year.eq(2024) &
                               development.outcome_date.le('2024-12-31')]
    first = confirmation.decision_date.min()
    early = development[development.listing_year.le(2023) & development.outcome_date.lt(first)]
    choice, candidates = select(early, first)
    prediction = calls(confirmation, choice['mode'])
    baseline, validation = metrics(confirmation, decisions(confirmation, DEFAULT)), metrics(confirmation, prediction)
    passed = bool(choice['mode'] != 'unchanged' and
                  validation['accuracy_pct'] > max(baseline['accuracy_pct'],
                      validation['always_hold_accuracy_pct'], validation['always_sell_accuracy_pct']) and
                  validation['balanced_accuracy_pct'] > baseline['balanced_accuracy_pct'])
    report = dict(model_family='Unchanged deterministic monitor family with research-only HOLD evidence gate',
        live_core_changed=False, synthetic_rows=0, candidate_count=len(MODES), selected=choice, candidates=candidates,
        selection_n=len(early), latest_selection_outcome=str(early.outcome_date.max().date()),
        first_confirmation_decision=str(first.date()), confirmation_n=len(confirmation),
        confirmation=dict(default=baseline, challenger=validation, paired_change=paired_interval(confirmation, prediction)),
        earlier_validation_gate_passed=passed, later_evaluated=False,
        limitations=['Hypothesis informed by inspected later errors; not a new blind holdout.',
                     'The gate requires earlier confirmation accuracy above both constant baselines and the default, plus better balanced accuracy.',
                     '2024 confirmations are restricted to outcomes completed by the end-2024 freeze.',
                     'No changed horizon, missing-value imputation, abstention or live promotion.'])
    if passed:
        later = assemble_risk('holdout')
        later_calls = calls(later, choice['mode'])
        report.update(later_evaluated=True, later=dict(metrics=metrics(later, later_calls),
                      paired_change=paired_interval(later, later_calls)))
    ledger = confirmation.copy()
    ledger['research_sell'] = prediction
    ledger.to_csv(REPORT.with_name(REPORT.stem+'_confirmation_ledger.csv'), index=False)
    paths = (*SOURCES['development'], ROOT/'ml/research_module2_hold_evidence.py',
             ROOT/'ml/research_module2_risk_governance.py', ROOT/'core/monitor_engine.py')
    if passed:
        paths += SOURCES['holdout']
    report['input_sha256'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    REPORT.with_suffix('.md').write_text('\n'.join([
        '# Module 2: require evidence before HOLD', '',
        f"Earlier selected policy: {choice['mode']}; selection cases {len(early)}.",
        f"2024 matured confirmation cases: {len(confirmation)}.",
        f"Default accuracy: {baseline['accuracy_pct']:.2f}%; challenger: {validation['accuracy_pct']:.2f}%.",
        f"Always-hold: {validation['always_hold_accuracy_pct']:.2f}%; always-sell: {validation['always_sell_accuracy_pct']:.2f}%.",
        f"Earlier validation gate passed: {passed}. Later evaluation performed: {passed}.", '',
        *['- '+item for item in report['limitations']]])+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
