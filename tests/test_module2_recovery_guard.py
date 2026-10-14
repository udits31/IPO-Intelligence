"""Guard research uses existing public-source observations, not invented IPOs."""
import numpy as np
import pytest

from ml.calibrate_module2_rules import DEFAULT, decisions
from ml.replay_module2_financial_rules import load_evidence
from ml.research_module2_filing_response import assemble_response, response_calls
from ml.research_module2_recovery_guard import CANDIDATES, calls, select


def test_disabled_recovery_is_exact_original_response_rule_and_missing_stays_default():
    later = assemble_response('holdout', load_evidence())
    missing = ~later.filing_response_status.eq('verified_daily_source_clock_response').to_numpy()
    baseline = decisions(later, DEFAULT)
    assert len(later) == 132
    for parameters in CANDIDATES:
        result = calls(later, parameters)
        np.testing.assert_array_equal(result[missing], baseline[missing])
        if not parameters['permit_below_issue_recovery']:
            np.testing.assert_array_equal(result, response_calls(later, parameters))


def test_recovery_never_creates_sell_or_changes_nonpositive_response():
    later = assemble_response('holdout', load_evidence())
    for parameters in CANDIDATES:
        if not parameters['permit_below_issue_recovery']:
            continue
        original = response_calls(later, parameters)
        result = calls(later, parameters)
        assert not np.any(result & ~original)
        changed = original != result
        threshold = parameters['response_confirm_abs_pct']
        assert later.loc[changed, 'stock_response_since_source_pct'].gt(threshold).all()
        assert later.loc[changed, 'excess_response_since_source_pct'].gt(threshold).all()
        assert later.loc[changed, 'decision_close'].lt(later.loc[changed, 'issue_price']).all()


def test_later_and_unmatured_cases_cannot_select_recovery_parameters():
    evidence = load_evidence()
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(assemble_response('holdout', evidence), '2024-12-31')
    development = assemble_response('development', evidence)
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(development, '2024-12-31')
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    assert len(known) == 214 and len(candidates) == 21 and choice in candidates
