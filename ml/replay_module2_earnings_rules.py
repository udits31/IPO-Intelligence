"""Test actual same-quarter earnings changes within the existing rule family.

Seven fixed threshold policies, earlier-only selection, unchanged checkpoints
and all later IPOs. Source review precedes any use of OCR values. No earnings
surprise, synthetic training data or new classifier is introduced.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, assemble, decisions, metrics, paired_interval
from ml.replay_module2_news_overlay import SOURCES
from ml.replay_module2_financial_rules import load_evidence as quarterly_evidence, TIGHTER, PROFIT_TRAIL
from ml.recheck_module2_financial_comparatives import (
    MANIFEST, REVIEWS, OUTPUT as EVIDENCE, REPORT as SOURCE_REPORT, recheck)

REPORT = ROOT / 'output/module2_comparative_earnings_rule_replay_20261022.json'
GRID = [{'mode': 'unchanged', 'max_publication_age_days': None}] + [
    dict(zip(('mode', 'max_publication_age_days'), values))
    for values in itertools.product(('deterioration_only', 'growth_only', 'both'), (30, 90))]


def verify_comparative_snapshot(row,quarter,record,reviews):
    # Reproduce the exact accepted source, not whichever alternative OCR/native
    # cache happens to arrive first while collection continues in the background.
    verified=recheck(record,quarter,reviews,only_extraction_path=ROOT/row['raw_ocr_path'])
    # Failed-page attempts are collection diagnostics, not prediction inputs.
    # Every accepted figure, source hash, render, review and qualifier is pinned.
    essential=lambda value:{key:item for key,item in value.items() if key!='page_attempts'}
    if not verified['feature_accepted'] or essential(verified)!=essential(row):
        raise ValueError('Accepted earnings table fails its frozen raw source/visual-review replay')
    return verified


def load_evidence():
    source_report = json.loads(SOURCE_REPORT.read_text(encoding='utf-8'))
    for relative, expected in source_report['input_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Comparative source check is outdated; rerun the offline recheck')
    if hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() != source_report['output_sha256']:
        raise ValueError('Reviewed comparative evidence was modified after its check')
    current = quarterly_evidence()
    manifest = {r['ipo_id']: r for r in (json.loads(s) for s in MANIFEST.read_text(encoding='utf-8').splitlines())}
    reviews = json.loads(REVIEWS.read_text(encoding='utf-8'))['reviews']
    rows = [json.loads(s) for s in EVIDENCE.read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(current) or {r['ipo_id'] for r in rows} != set(current):
        raise ValueError('Comparative evidence must retain every original checkpoint')
    for row in rows:
        original = current[row['ipo_id']]
        if any(row[key] != original[key] for key in ('ipo_id', 'symbol', 'decision_date')):
            raise ValueError('Comparative issuer or checkpoint differs from the quarterly source')
        if row['feature_accepted']:
            verify_comparative_snapshot(row,original,manifest[row['ipo_id']],reviews)
    return {row['ipo_id']: row for row in rows}


def assemble_earnings(name, evidence):
    frame = assemble(*SOURCES[name])
    columns = {key: [] for key in ('revenue_current_inr', 'revenue_prior_year_quarter_inr',
                                   'profit_current_inr', 'profit_prior_year_quarter_inr',
                                   'financial_publication_age_days')}
    for case in frame.itertuples():
        row = evidence[case.ipo_id]
        if pd.Timestamp(row['decision_date']) != case.decision_date:
            raise ValueError('Earnings date differs from the frozen monitor checkpoint')
        if not row['feature_accepted']:
            for values in columns.values():
                values.append(np.nan)
            continue
        values = row['comparative_financials']
        cutoff = case.decision_date + pd.Timedelta(hours=15, minutes=30)
        age = (cutoff - pd.Timestamp(row['financial_published_at_ist'])).total_seconds() / 86400
        period_age = (case.decision_date - pd.Timestamp(values['current_quarter_end'])).days
        if age < 0 or period_age < 0:
            raise ValueError('A later filing or financial period reached the earnings replay')
        for key in columns:
            columns[key].append((age if key == 'financial_publication_age_days' else values[key])
                                if period_age <= 180 else np.nan)
    for key, values in columns.items():
        frame[key] = values
    return frame


def earnings_masks(frame, max_age):
    fields = ['revenue_current_inr', 'revenue_prior_year_quarter_inr',
              'profit_current_inr', 'profit_prior_year_quarter_inr']
    known = frame[fields].notna().all(axis=1) & frame.financial_publication_age_days.le(max_age)
    revenue_down = frame.revenue_current_inr.lt(frame.revenue_prior_year_quarter_inr)
    profit_down = frame.profit_current_inr.lt(frame.profit_prior_year_quarter_inr)
    turns_loss = frame.profit_current_inr.lt(0) & frame.profit_prior_year_quarter_inr.ge(0)
    deteriorating = known & ((revenue_down & profit_down) | turns_loss)
    # A turnaround from a prior loss is not a conventional earnings-growth
    # observation. Neither missing values nor the absolute size of sales is
    # taken as proof that the business is improving.
    growing = known & frame.revenue_current_inr.gt(frame.revenue_prior_year_quarter_inr) & (
        frame.profit_current_inr.gt(frame.profit_prior_year_quarter_inr) & frame.profit_prior_year_quarter_inr.gt(0))
    return deteriorating.to_numpy(copy=True), growing.to_numpy(copy=True)


def earnings_calls(frame, parameters):
    calls = decisions(frame, DEFAULT).copy()
    if parameters['mode'] == 'unchanged':
        return calls
    negative, positive = earnings_masks(frame, parameters['max_publication_age_days'])
    if parameters['mode'] in ('deterioration_only', 'both'):
        calls[negative] = decisions(frame.loc[negative], TIGHTER)
    if parameters['mode'] in ('growth_only', 'both'):
        positive &= frame.decision_close.ge(frame.issue_price).to_numpy()
        calls[positive] = decisions(frame.loc[positive], PROFIT_TRAIL)
    return calls


def select(frame, freeze):
    if (frame.empty or not frame.listing_year.between(2021, 2024).all() or
            not frame.outcome_date.le(pd.Timestamp(freeze)).all()):
        raise ValueError('Only matured 2021-2024 outcomes may choose earnings-rule settings')
    baseline = decisions(frame, DEFAULT)
    candidates = [{'parameters': parameters, 'metrics': metrics(frame, earnings_calls(frame, parameters)),
                   'changed_calls': int((earnings_calls(frame, parameters) != baseline).sum())}
                  for parameters in GRID]
    choice = max(candidates, key=lambda row: (row['metrics']['balanced_accuracy_pct'],
                                             row['metrics']['accuracy_pct'], -row['changed_calls']))
    return choice, candidates


def run():
    evidence = load_evidence()
    development, later = [assemble_earnings(name, evidence) for name in ('development', 'holdout')]
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    calls = earnings_calls(later, choice['parameters'])
    score = metrics(later, calls)
    ledger = later.copy()
    ledger['earnings_rule_sell'] = calls
    ledger['earnings_rule_correct'] = calls != later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem + '_ledger.csv'), index=False)
    coverage = {name: {'n': len(frame), 'accepted_comparative_cases': int(frame.revenue_current_inr.notna().sum())}
                for name, frame in (('development', development), ('later', later))}
    report = {'rule_family': 'Existing loss/relative-lag/trailing thresholds; dated earnings conditions only',
              'research_stage': 'Partial-source pipeline verification; not a completed earnings-signal accuracy study',
              'candidate_count': len(GRID), 'coverage': coverage, 'selection_cases_matured_before_2025': len(known),
              'selected': choice, 'earlier_candidates': candidates,
              'later': {'default': metrics(later, decisions(later, DEFAULT)),
                        'earnings_rules': score, 'paired_change': paired_interval(later, calls)},
              'comparative_features_changed_any_later_call': bool(np.any(calls != decisions(later, DEFAULT))),
              'requested_accuracy_established': bool(score['accuracy_pct'] >= 75 and score['balanced_accuracy_pct'] >= 70 and
                  score['accuracy_pct'] > max(score['always_hold_accuracy_pct'], score['always_sell_accuracy_pct'])),
              'live_policy_changed': False, 'synthetic_training_rows': 0,
              'input_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (EVIDENCE, SOURCE_REPORT, REVIEWS, *SOURCES['development'],
                                            *SOURCES['holdout'], ROOT / 'core/monitor_engine.py',
                                            ROOT / 'core/attribution_engine.py', ROOT / 'ml/replay_module2_earnings_rules.py')},
              'limitations': ['Extraction coverage is incomplete; missing sources retain the default rule, not zero earnings.',
                              'Quarterly earnings changes are not consensus surprises or causal return forecasts.',
                              'Only existing thresholds are conditioned; no replacement learned classifier or live promotion.',
                              'Later outcomes were inspected in earlier experiments; this is not a blind or prospective test.',
                              'The 31st-close checkpoint and subsequent 30-session sign remain unchanged for all 132 later cases.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('coverage', 'selected', 'later', 'requested_accuracy_established')}, indent=2))
    return report


if __name__ == '__main__':
    run()
