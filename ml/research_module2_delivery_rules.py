"""Earlier-year gate for deterministic delivery-confirmed price actions.

No model-family replacement. Existing public NSE delivery tables are aligned
to the corrected exchange prices. Delivery ends one session before decision;
missing histories and windows crossing a share-basis change keep default rules.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import ROOT, DEFAULT, decisions, metrics, paired_interval
from ml.replay_module2_news_overlay import SOURCES
from ml.research_module2_risk_governance import assemble_risk

REPORT = ROOT/'output/module2_delivery_rules_research_20261022.json'
DELIVERY = {'development': ROOT/'data/audit/module2_nse_delivery_development_full_window.csv',
            'holdout': ROOT/'data/audit/module2_nse_delivery_holdout.csv'}
MODES = ('unchanged', 'distribution_exit', 'accumulation_hold', 'both')


def assemble(name):
    frame = assemble_risk(name)
    price = pd.read_csv(SOURCES[name][0], parse_dates=['date'])
    delivery = pd.read_csv(DELIVERY[name], parse_dates=['session_date'])
    if delivery.duplicated(['ipo_id', 'session_date']).any():
        raise ValueError('Duplicate public delivery observation')
    observations = []
    for case in frame.itertuples():
        history = price[price.ipo_id.eq(case.ipo_id) & price.date.lt(case.decision_date)].sort_values('date').tail(20)
        raw = delivery[delivery.ipo_id.eq(case.ipo_id)]
        window = history.merge(raw, left_on='date', right_on='session_date', how='left', validate='one_to_one')
        status = 'verified_prior_session_delivery'
        record = dict(delivery_status=status, latest_delivery_session=None,
                      delivery_5_weighted_pct=np.nan, delivery_20_weighted_pct=np.nan,
                      traded_quantity_5_to_20=np.nan)
        if len(window) != 20 or window.nse_close.isna().any():
            record['delivery_status'] = 'missing_aligned_prior_delivery'
        elif not np.allclose(window.nse_close, window.unadjusted_nse_close, atol=.011, rtol=0):
            record['delivery_status'] = 'raw_exchange_close_conflict'
        elif window.shares_per_listing_share.nunique() != 1:
            record['delivery_status'] = 'share_basis_change_inside_volume_window'
        elif (window[['traded_qty', 'deliverable_qty']].isna().any().any() or
              window.traded_qty.le(0).any() or window.deliverable_qty.lt(0).any() or
              window.deliverable_qty.gt(window.traded_qty).any()):
            record['delivery_status'] = 'invalid_observed_delivery'
        else:
            recent = window.tail(5)
            record.update(latest_delivery_session=str(window.date.max().date()),
                delivery_5_weighted_pct=float(recent.deliverable_qty.sum()/recent.traded_qty.sum()*100),
                delivery_20_weighted_pct=float(window.deliverable_qty.sum()/window.traded_qty.sum()*100),
                traded_quantity_5_to_20=float(recent.traded_qty.mean()/window.traded_qty.mean()))
        observations.append(record)
    return frame.join(pd.DataFrame(observations, index=frame.index))


def calls(frame, mode):
    if mode not in MODES:
        raise ValueError('Unknown fixed delivery rule')
    result = decisions(frame, DEFAULT).copy()
    observed = frame.delivery_status.eq('verified_prior_session_delivery')
    confirmed = (observed & frame.delivery_5_weighted_pct.gt(frame.delivery_20_weighted_pct) &
                 frame.traded_quantity_5_to_20.gt(1))
    if mode in ('distribution_exit', 'both'):
        result[(confirmed & frame.stock_10_pct.lt(0)).to_numpy()] = True
    if mode in ('accumulation_hold', 'both'):
        # No structural underwater stop is waived by an observed volume increase.
        result[(confirmed & frame.stock_10_pct.gt(0) & frame.decision_close.ge(frame.issue_price)).to_numpy()] = False
    return result


def run():
    development = assemble('development')
    confirmation = development[development.listing_year.eq(2024) & development.outcome_date.le('2024-12-31')]
    first = confirmation.decision_date.min()
    early = development[development.listing_year.between(2021, 2023) & development.outcome_date.lt(first)]
    original = decisions(early, DEFAULT)
    candidates = [dict(mode=mode, metrics=metrics(early, calls(early, mode)),
                       changed_calls=int((calls(early, mode) != original).sum())) for mode in MODES]
    choice = max(candidates, key=lambda r: (r['metrics']['balanced_accuracy_pct'],
                                          r['metrics']['accuracy_pct'], -r['changed_calls']))
    predicted = calls(confirmation, choice['mode'])
    baseline, score = metrics(confirmation, decisions(confirmation, DEFAULT)), metrics(confirmation, predicted)
    passed = bool(choice['mode'] != 'unchanged' and score['accuracy_pct'] >
        max(baseline['accuracy_pct'], score['always_hold_accuracy_pct'], score['always_sell_accuracy_pct']) and
        score['balanced_accuracy_pct'] > baseline['balanced_accuracy_pct'])
    report = dict(live_core_changed=False, new_model_family=False, synthetic_rows=0,
        selection_n=len(early), confirmation_n=len(confirmation), candidates=candidates, selected=choice,
        development_coverage=development.delivery_status.value_counts().to_dict(),
        confirmation=dict(default=baseline, challenger=score, paired_change=paired_interval(confirmation, predicted)),
        earlier_gate_passed=passed, later_evaluated=False,
        limitations=['Delivery percentage does not identify institutions or prove accumulation/distribution.',
          'Daily delivery publication clocks are unverified: only prior-session data is used.',
          'Retrospective research, not blind or prospective validation; earlier studies inspected later data.',
          'Missing, misaligned or corporate-action-crossing windows keep the original action; no IPO is dropped.'])
    paths = (SOURCES['development'][0], SOURCES['development'][1], DELIVERY['development'],
             ROOT/'ml/research_module2_delivery_rules.py', ROOT/'core/monitor_engine.py')
    if passed:
        later = assemble('holdout')
        prediction = calls(later, choice['mode'])
        report.update(later_evaluated=True, later=dict(metrics=metrics(later, prediction),
            paired_change=paired_interval(later, prediction), coverage=later.delivery_status.value_counts().to_dict()))
        paths += (*SOURCES['holdout'], DELIVERY['holdout'])
    confirmation.assign(research_sell=predicted).to_csv(REPORT.with_name(REPORT.stem+'_confirmation_ledger.csv'), index=False)
    report['input_sha256'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    REPORT.with_suffix('.md').write_text('\n'.join([
        '# Module 2: delivery-confirmed price rules', '',
        f"Selected on {len(early)} earlier cases: {choice['mode']}.",
        f"Matured 2024 confirmation: {len(confirmation)} cases; default {baseline['accuracy_pct']:.2f}%, challenger {score['accuracy_pct']:.2f}%.",
        f"Earlier gate passed: {passed}; later evaluated: {passed}.", '',
        *['- '+s for s in report['limitations']]])+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
