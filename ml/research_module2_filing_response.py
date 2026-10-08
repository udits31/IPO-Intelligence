"""Test observed responses aligned to dated financial-source publication.

The reference clock is verified NSE financial metadata, not necessarily the
first public announcement. Daily closes cannot isolate an intraday jump.
No issue-price anchor, absent market return or future price is fabricated.
The underlying deterministic monitor, primary horizon and full later cohort
remain unchanged; only a bounded research governance branch is tested.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, assemble, decisions, metrics, paired_interval
from ml.replay_module2_financial_rules import EVIDENCE, load_evidence
from ml.replay_module2_news_overlay import SOURCES

REPORT = ROOT / 'output/module2_filing_response_research_20261022.json'
GRID = [{'mode': 'unchanged', 'max_publication_age_days': None, 'response_confirm_abs_pct': None}] + [
    dict(zip(('mode', 'max_publication_age_days', 'response_confirm_abs_pct'), values))
    for values in itertools.product(('negative_only', 'positive_only', 'both'), (10, 30), (0., 3.))]


def ist_naive(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_convert('Asia/Kolkata').tz_localize(None) if stamp.tzinfo is not None else stamp


def response_features(prices, checkpoint, published):
    checkpoint, published = ist_naive(checkpoint).normalize(), ist_naive(published)
    cutoff = checkpoint + pd.Timedelta(hours=15, minutes=30)
    if published > cutoff:
        raise ValueError('Financial source was published after the decision cutoff')
    result = {'filing_response_status': 'no_observed_prepublication_close',
              'financial_source_clock_ist': published.isoformat(),
              'source_publication_age_days': (cutoff - published).total_seconds() / 86400,
              'source_reference_close_date': None, 'observed_post_source_closes': 0,
              'stock_response_since_source_pct': np.nan, 'nifty_response_since_source_pct': np.nan,
              'excess_response_since_source_pct': np.nan}
    history = prices[prices.date.le(checkpoint)].sort_values('date').copy()
    if history.empty or history.iloc[-1].date != checkpoint:
        result['filing_response_status'] = 'missing_decision_quote'
        return result
    if history.date.duplicated().any():
        raise ValueError('Duplicate daily close in the financial response window')
    # Explicit close-session convention, not a claim to have a quote captured
    # immediately before a midday filing or an intraday announcement jump.
    closes = history.date + pd.Timedelta(hours=15, minutes=30)
    preceding = history[closes.lt(published)]
    following = history[closes.ge(published)]
    result['observed_post_source_closes'] = len(following)
    if preceding.empty:
        return result
    start, end = preceding.iloc[-1], history.iloc[-1]
    result['source_reference_close_date'] = start.date.strftime('%Y-%m-%d')
    if len(following) < 2:
        result['filing_response_status'] = 'fewer_than_two_observed_post_source_closes'
        return result
    for label, column in [('stock', 'close_price_adjusted'), ('nifty', 'nifty50_close')]:
        first, last = start[column], end[column]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0:
            result['filing_response_status'] = 'missing_' + label + '_response_prices'
            return result
        result[label + '_response_since_source_pct'] = float((last / first - 1) * 100)
    result['excess_response_since_source_pct'] = (result['stock_response_since_source_pct'] -
                                                 result['nifty_response_since_source_pct'])
    result['filing_response_status'] = 'verified_daily_source_clock_response'
    return result


def assemble_response(name, evidence):
    prices_path, ledger_path = SOURCES[name]
    frame = assemble(prices_path, ledger_path)
    prices = pd.read_csv(prices_path, parse_dates=['date'])
    groups = {key: group for key, group in prices.groupby('ipo_id')}
    features = []
    for case in frame.itertuples():
        source = evidence[case.ipo_id]
        if pd.Timestamp(source['decision_date']) != case.decision_date:
            raise ValueError('Financial clock belongs to another monitor checkpoint')
        result = {'filing_response_status': source['status'], 'financial_source_clock_ist': None,
                  'source_publication_age_days': np.nan, 'source_reference_close_date': None,
                  'observed_post_source_closes': 0, 'stock_response_since_source_pct': np.nan,
                  'nifty_response_since_source_pct': np.nan, 'excess_response_since_source_pct': np.nan}
        if source['status'] == 'verified_quarterly_metrics':
            item = source['financials']
            period_age = (case.decision_date - pd.Timestamp(item['period_end'])).days
            if period_age < 0:
                raise ValueError('Future financial period reached the event response study')
            if period_age <= 180:
                result = response_features(groups[case.ipo_id], case.decision_date, item['published_at_ist'])
            else:
                result['filing_response_status'] = 'financial_period_older_than_180_days'
        features.append(result)
    return frame.join(pd.DataFrame(features, index=frame.index))


def response_calls(frame, parameters):
    calls = decisions(frame, DEFAULT).copy()
    if parameters['mode'] == 'unchanged':
        return calls
    available = (frame.filing_response_status.eq('verified_daily_source_clock_response') &
                 frame.source_publication_age_days.between(0, parameters['max_publication_age_days']))
    threshold = parameters['response_confirm_abs_pct']
    if parameters['mode'] in ('negative_only', 'both'):
        negative = (available & frame.stock_response_since_source_pct.lt(-threshold) &
                    frame.excess_response_since_source_pct.lt(-threshold))
        calls[negative.to_numpy()] = True
    if parameters['mode'] in ('positive_only', 'both'):
        positive = (available & frame.stock_response_since_source_pct.gt(threshold) &
                    frame.excess_response_since_source_pct.gt(threshold) & frame.decision_close.ge(frame.issue_price))
        calls[positive.to_numpy()] = False
    return calls


def select(frame, freeze):
    if (frame.empty or not frame.listing_year.between(2021, 2024).all() or
            not frame.outcome_date.le(pd.Timestamp(freeze)).all()):
        raise ValueError('Filing-response selection requires matured 2021-2024 outcomes only')
    baseline = decisions(frame, DEFAULT)
    candidates = [{'parameters': parameters, 'metrics': metrics(frame, response_calls(frame, parameters)),
                   'changed_calls': int((response_calls(frame, parameters) != baseline).sum())}
                  for parameters in GRID]
    choice = max(candidates, key=lambda row: (row['metrics']['balanced_accuracy_pct'],
                                             row['metrics']['accuracy_pct'], -row['changed_calls']))
    return choice, candidates


def coverage(frame):
    return {'n': len(frame), 'source_status': frame.filing_response_status.value_counts().to_dict(),
            'verified_response_windows': int(frame.filing_response_status.eq('verified_daily_source_clock_response').sum())}


def run():
    evidence = load_evidence()
    development, later = [assemble_response(name, evidence) for name in ('development', 'holdout')]
    confirmation = development[development.listing_year.eq(2024)]
    first = confirmation.decision_date.min()
    early = development[development.listing_year.le(2023) & development.outcome_date.lt(first)]
    early_choice, early_candidates = select(early, first - pd.Timedelta(days=1))
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    calls = response_calls(later, choice['parameters'])
    score = metrics(later, calls)
    ledger = later.copy()
    ledger['experimental_filing_response_sell'] = calls
    ledger['experimental_filing_response_correct'] = calls != later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem + '_ledger.csv'), index=False)
    report = {'model_family': 'Same deterministic monitor family; bounded filing-clock price-response governance research',
              'new_learned_classifier': False, 'live_core_changed': False, 'synthetic_training_rows': 0,
              'protocol': 'Same 31st-close checkpoint, next-close execution and 30-later-session direction; all 132 later cases retained.',
              'source_clock_definition': 'Verified selected NSE financial-source publication, not necessarily the first public earnings announcement.',
              'response_definition': 'Daily close before the source clock to decision close; at least two observed post-source closes; same-date Nifty response subtracted.',
              'candidate_count': len(GRID), 'coverage': {'development': coverage(development), 'later': coverage(later),
                  'early_matured_2021_2023_selection': len(early), 'matured_before_2025_selection': len(known)},
              'early_choice': early_choice, 'early_candidates': early_candidates,
              'selected_on_earlier_data': choice, 'final_candidates': candidates,
              'confirmation_2024': {'default': metrics(confirmation, decisions(confirmation, DEFAULT)),
                                    'response_rules': metrics(confirmation, response_calls(confirmation, early_choice['parameters']))},
              'later': {'default': metrics(later, decisions(later, DEFAULT)), 'response_rules': score,
                        'paired_change': paired_interval(later, calls)},
              'requested_accuracy_established': bool(score['accuracy_pct'] >= 75 and score['balanced_accuracy_pct'] >= 70 and
                  score['accuracy_pct'] > max(score['always_hold_accuracy_pct'], score['always_sell_accuracy_pct'])),
              'input_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (EVIDENCE, *SOURCES['development'], *SOURCES['holdout'],
                                            ROOT / 'core/monitor_engine.py', ROOT / 'core/attribution_engine.py',
                                            ROOT / 'ml/research_module2_filing_response.py')},
              'limitations': ['The selected financial source can be later than an earlier PDF or press release; this is not an isolated first-announcement return.',
                              'Daily session-close timing does not establish an intraday quote immediately before a midday publication.',
                              'Nifty prices are retrospective secondary-provider observations, not independently immutable exchange archives.',
                              'Missing anchors, source clocks or market prices retain the default rule; IPOs are not removed and issue price is not a substitute anchor.',
                              'Repeated later-year inspection makes this exploratory, not blind or prospective validation.',
                              'No live policy or paper result is promoted. Literature on announcement responses is not a score for this application.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    text = ['# Module 2: price response aligned to a financial-source clock', '', report['protocol'], '',
            '| Policy | Later direction accuracy |', '|---|---:|',
            f"| Current monitor | {report['later']['default']['accuracy_pct']:.2f}% |",
            f"| Earlier-data-selected source-response rules | {score['accuracy_pct']:.2f}% |",
            f"| Always sell | {score['always_sell_accuracy_pct']:.2f}% |", '',
            report['source_clock_definition'], '', report['response_definition'], '', '## Limitations', '']
    text.extend('- ' + item for item in report['limitations'])
    REPORT.with_suffix('.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('coverage', 'selected_on_earlier_data', 'confirmation_2024', 'later',
                                                 'requested_accuracy_established')}, indent=2))
    return report


if __name__ == '__main__':
    run()
