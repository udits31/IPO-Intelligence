"""Observed event facts and chronology/invariant guards, not generated IPOs."""
import json

import numpy as np
import pytest

from ml.calibrate_module2_rules import DEFAULT, decisions
from ml.replay_module2_financial_rules import EVIDENCE as QUARTERS
from ml.recheck_module2_financial_comparatives import OUTPUT as COMPARATIVES
from ml.research_module2_earnings_events import GRID, assemble_events, event_calls, event_masks, select


def observed(name):
    sources = [{r['ipo_id']: r for r in (json.loads(s) for s in path.read_text(encoding='utf-8').splitlines())}
               for path in (QUARTERS, COMPARATIVES)]
    return assemble_events(name, *sources)


def test_all_later_cases_and_110_real_current_quarters_remain_in_the_test():
    frame = observed('holdout')
    assert len(frame) == 132
    assert frame.current_revenue_inr.notna().sum() == 110
    assert frame.current_profit_inr.lt(0).sum() == 9
    missing = frame.current_revenue_inr.isna().to_numpy()
    for parameters in GRID:
        calls = event_calls(frame, parameters)
        assert len(calls) == 132
        np.testing.assert_array_equal(calls[missing], decisions(frame, DEFAULT)[missing])


def test_positive_events_do_not_waive_any_underwater_structural_stop():
    frame = observed('development')
    underwater = frame.decision_close.lt(frame.issue_price).to_numpy()
    baseline = decisions(frame, DEFAULT)
    for parameters in GRID:
        calls = event_calls(frame, parameters)
        assert not np.any(baseline[underwater] & ~calls[underwater])


def test_event_masks_require_actual_price_confirmation_not_financial_words_alone():
    frame = observed('development')
    parameters = {'mode': 'both', 'max_age_days': 90, 'momentum_confirm_abs_pct': 3.}
    negative, positive = event_masks(frame, parameters)
    assert frame.loc[negative, 'stock_10_pct'].lt(-3).all()
    assert frame.loc[positive, 'stock_10_pct'].gt(3).all()
    assert frame.loc[positive, 'current_profit_inr'].gt(0).all()
    assert frame.loc[positive, 'prior_revenue_inr'].notna().all()
    assert not np.any(negative & positive)


def test_later_or_unmatured_outcomes_cannot_select_event_logic():
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(observed('holdout'), '2024-12-31')
    development = observed('development')
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(development, '2024-12-31')
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    assert len(known) == 214 and len(candidates) == 13 and choice in candidates


def test_real_tax_driven_profit_comparisons_are_qualified_not_rewritten():
    frame=observed('holdout').set_index('ipo_id')
    assert frame.loc[530,'current_profit_inr']==64060000.
    assert frame.loc[530,'prior_profit_inr']==2037440000.
    assert not frame.loc[530,'comparative_profit_event_eligible']
    assert frame.loc[577,'current_profit_inr']==-367550000.
    assert frame.loc[577,'prior_profit_inr']==-420400000.
    assert not frame.loc[577,'comparative_profit_event_eligible']
    # The source qualification does not hide the company or its current loss.
    assert len(frame)==132 and frame.current_profit_inr.lt(0).sum()==9
