"""Earlier-only test of real earnings events in the deterministic monitor.

Unlike choosing different price thresholds, an observed deterioration can add
an event-risk exit. Improving results may defer a profit-taking trail only
with positive price confirmation, never an underwater structural stop. This
is an explicit research overlay, not a replacement learned classifier or a
silent edit of the deployed monitor.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, decisions, metrics, paired_interval
from ml.replay_module2_financial_rules import load_evidence as load_quarters, EVIDENCE as QUARTERS
from ml.replay_module2_earnings_rules import load_evidence as load_comparatives
from ml.recheck_module2_financial_comparatives import OUTPUT as COMPARATIVES, REPORT as SOURCE_REPORT
from ml.research_module2_risk_governance import assemble_risk
from ml.replay_module2_news_overlay import SOURCES

REPORT = ROOT / 'output/module2_earnings_event_research_20261022.json'
GRID = [{'mode': 'unchanged', 'max_age_days': None, 'momentum_confirm_abs_pct': None}] + [
    dict(zip(('mode', 'max_age_days', 'momentum_confirm_abs_pct'), values))
    for values in itertools.product(('negative_only', 'positive_only', 'both'), (30, 90), (0., 3.))]
FIELDS = ('current_revenue_inr', 'current_profit_inr', 'prior_revenue_inr', 'prior_profit_inr',
          'result_publication_age_days')


def assemble_events(name, quarters, comparatives):
    frame = assemble_risk(name)
    values = {field: [] for field in FIELDS}
    profit_eligible=[]
    for case in frame.itertuples():
        quarter, comparative = quarters[case.ipo_id], comparatives[case.ipo_id]
        if any(pd.Timestamp(row['decision_date']) != case.decision_date for row in (quarter, comparative)):
            raise ValueError('Earnings event belongs to a different monitor checkpoint')
        item = dict.fromkeys(FIELDS, np.nan)
        eligible=False
        if quarter['status'] == 'verified_quarterly_metrics':
            source = quarter['financials']
            cutoff = case.decision_date + pd.Timedelta(hours=15, minutes=30)
            age = (cutoff - pd.Timestamp(source['published_at_ist'])).total_seconds() / 86400
            period_age = (case.decision_date - pd.Timestamp(source['period_end'])).days
            if age < 0 or period_age < 0:
                raise ValueError('Earnings event was not published before the decision')
            if period_age <= 180:
                item.update(current_revenue_inr=source['revenue_inr'], current_profit_inr=source['profit_inr'],
                            result_publication_age_days=age)
                if comparative['feature_accepted']:
                    previous = comparative['comparative_financials']
                    if previous['current_quarter_end'] != source['period_end']:
                        raise ValueError('Comparative and current filing periods differ')
                    item.update(prior_revenue_inr=previous['revenue_prior_year_quarter_inr'],
                                prior_profit_inr=previous['profit_prior_year_quarter_inr'])
                    eligible=comparative['profit_comparison_qualification']['eligible_for_directional_event']
        for field in FIELDS:
            values[field].append(item[field])
        profit_eligible.append(eligible)
    for field, items in values.items():
        frame[field] = items
    frame['comparative_profit_event_eligible']=profit_eligible
    return frame


def event_masks(frame, parameters):
    fresh = (frame.current_revenue_inr.gt(0) & frame.current_profit_inr.notna() &
             frame.result_publication_age_days.between(0, parameters['max_age_days']))
    comparable = fresh & frame.prior_revenue_inr.gt(0) & frame.prior_profit_inr.notna() & frame.comparative_profit_event_eligible
    sales_change = frame.current_revenue_inr - frame.prior_revenue_inr
    profit_change = frame.current_profit_inr - frame.prior_profit_inr
    margin_change_pp = (frame.current_profit_inr / frame.current_revenue_inr -
                        frame.prior_profit_inr / frame.prior_revenue_inr) * 100
    # A verified loss remains a loss even when a comparative cannot be found.
    # Absent financials, however, are not zero profit or a negative event.
    deteriorating = fresh & (frame.current_profit_inr.lt(0) |
        (comparable & ((sales_change.lt(0) & profit_change.lt(0)) | margin_change_pp.le(-2.))))
    # Includes a genuine turnaround from a prior loss, but does not label its
    # percentage change conventional growth or infer an earnings surprise.
    improving = comparable & frame.current_profit_inr.gt(0) & sales_change.gt(0) & profit_change.gt(0) & margin_change_pp.ge(0)
    confirmation = parameters['momentum_confirm_abs_pct']
    negative = deteriorating & frame.stock_10_pct.lt(-confirmation)
    positive = improving & frame.stock_10_pct.gt(confirmation) & frame.decision_close.ge(frame.issue_price)
    return negative.to_numpy(), positive.to_numpy()


def event_calls(frame, parameters):
    result = decisions(frame, DEFAULT).copy()
    if parameters['mode'] == 'unchanged':
        return result
    negative, positive = event_masks(frame, parameters)
    if parameters['mode'] in ('negative_only', 'both'):
        result[negative] = True
    if parameters['mode'] in ('positive_only', 'both'):
        result[positive] = False
    return result


def select(frame, freeze):
    if (frame.empty or not frame.listing_year.between(2021, 2024).all() or
            not frame.outcome_date.le(pd.Timestamp(freeze)).all()):
        raise ValueError('Earnings-event selection requires matured 2021-2024 outcomes only')
    baseline = decisions(frame, DEFAULT)
    candidates = [{'parameters': parameters, 'metrics': metrics(frame, event_calls(frame, parameters)),
                   'changed_calls': int((event_calls(frame, parameters) != baseline).sum())}
                  for parameters in GRID]
    winner = max(candidates, key=lambda row: (row['metrics']['balanced_accuracy_pct'],
                                             row['metrics']['accuracy_pct'], -row['changed_calls']))
    return winner, candidates


def coverage(frame):
    return {'n': len(frame), 'verified_current_quarter_cases': int(frame.current_revenue_inr.notna().sum()),
            'verified_year_earlier_comparatives': int(frame.prior_revenue_inr.notna().sum()),
            'source_qualified_profit_comparisons':int((frame.prior_revenue_inr.notna() & ~frame.comparative_profit_event_eligible).sum()),
            'observed_loss_making_quarters': int(frame.current_profit_inr.lt(0).sum())}


def run():
    quarters, comparatives = load_quarters(), load_comparatives()
    development, later = [assemble_events(name, quarters, comparatives) for name in ('development', 'holdout')]
    confirmation = development[development.listing_year.eq(2024)]
    first = confirmation.decision_date.min()
    early = development[development.listing_year.isin([2022, 2023]) & development.outcome_date.lt(first)]
    early_choice, early_candidates = select(early, first - pd.Timedelta(days=1))
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    calls = event_calls(later, choice['parameters'])
    score = metrics(later, calls)
    ledger = later.copy()
    ledger['experimental_earnings_event_sell'] = calls
    ledger['experimental_earnings_event_correct'] = calls != later.day30_return_pct.gt(0).to_numpy()
    ledger.to_csv(REPORT.with_name(REPORT.stem + '_ledger.csv'), index=False)
    report = {'model_family': 'Same deterministic monitor family, explicit earnings-event/price-confirmation research overlay',
              'new_learned_classifier': False, 'live_core_changed': False, 'synthetic_training_rows': 0,
              'research_stage': 'Incomplete comparative extraction; exploratory, not a completed earnings-signal validation',
              'protocol': 'Same 31st-close checkpoint, next-close execution and 30-later-session direction; all 132 later IPOs retained.',
              'candidate_count': len(GRID), 'coverage': {'development': coverage(development), 'later': coverage(later),
                                                       'matured_before_2025_selection': len(known)},
              'early_choice': early_choice, 'early_candidates': early_candidates,
              'selected_on_earlier_data': choice, 'final_candidates': candidates,
              'confirmation_2024': {'default': metrics(confirmation, decisions(confirmation, DEFAULT)),
                                    'event_rules': metrics(confirmation, event_calls(confirmation, early_choice['parameters']))},
              'later': {'default': metrics(later, decisions(later, DEFAULT)), 'event_rules': score,
                        'paired_change': paired_interval(later, calls)},
              'requested_accuracy_established': bool(score['accuracy_pct'] >= 75 and score['balanced_accuracy_pct'] >= 70 and
                  score['accuracy_pct'] > max(score['always_hold_accuracy_pct'], score['always_sell_accuracy_pct'])),
              'input_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (QUARTERS, COMPARATIVES, SOURCE_REPORT, *SOURCES['development'], *SOURCES['holdout'],
                                            ROOT / 'core/monitor_engine.py', ROOT / 'core/attribution_engine.py',
                                            ROOT / 'ml/research_module2_risk_governance.py',
                                            ROOT / 'ml/replay_module2_earnings_rules.py',
                                            ROOT / 'ml/research_module2_earnings_events.py')},
              'limitations': ['Current loss/profit and changes are observed accounting results, not consensus surprises or guaranteed return drivers.',
                              'Explicit source-reviewed conflicts between before/after-tax directions qualify comparative events without inventing normalized profits; other accounting distortions may remain unidentified.',
                              'Missing comparatives do not remove IPOs or imply flat growth; original rules remain where required evidence is absent.',
                              'Positive events cannot override an underwater structural stop; negative events require observed adverse price confirmation.',
                              'Publication clocks are historical source labels; retrospectively retrieved revisions and archival completeness remain limitations.',
                              'Later outcomes were examined in previous experiments; all comparisons are exploratory, not blind or prospective.',
                              'The overlay is not promoted, and no paper claim is changed by this experiment.']}
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    text = ['# Module 2: earnings-event decisions with price confirmation', '', report['protocol'], '',
            '| Policy | Later direction accuracy |', '|---|---:|',
            f"| Current monitor | {report['later']['default']['accuracy_pct']:.2f}% |",
            f"| Earlier-data-selected earnings events | {score['accuracy_pct']:.2f}% |",
            f"| Always sell | {score['always_sell_accuracy_pct']:.2f}% |", '',
            'This is an explicit event-risk decision experiment, not another assignment of the same three price thresholds. '
            'It preserves the deterministic model family, all cases and the primary horizon. No live policy is promoted.', '',
            'Comparative-source extraction remains incomplete. Current loss-making quarters can be observed without inventing a prior quarter; '
            'growth or margin changes require verified prior-year figures.', '', '## Limitations', '']
    text.extend('- ' + item for item in report['limitations'])
    REPORT.with_suffix('.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('coverage', 'selected_on_earlier_data', 'confirmation_2024', 'later',
                                                 'requested_accuracy_established')}, indent=2))
    return report


if __name__ == '__main__':
    run()
