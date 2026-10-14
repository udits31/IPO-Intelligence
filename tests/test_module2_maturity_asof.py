"""Maturity chronology over the actual earlier/later IPO cohorts."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.replay_module2_financial_rules import EVIDENCE
from ml.research_module2_filing_response import assemble_response
from ml.research_module2_maturity_asof import available_development, asof_calls


def observed():
    evidence = {r['ipo_id']: r for r in (json.loads(s) for s in EVIDENCE.read_text(encoding='utf-8').splitlines())}
    return assemble_response('development', evidence), assemble_response('holdout', evidence)


def test_training_pool_contains_only_earlier_ipos_with_completed_outcomes():
    development, later = observed()
    for date in sorted(later.decision_date.unique()):
        known = available_development(development, date)
        assert known.listing_year.between(2021, 2024).all()
        assert known.outcome_date.lt(pd.Timestamp(date)).all()
        assert not set(known.ipo_id) & set(later.ipo_id)
    with pytest.raises(ValueError, match='2021-2024'):
        available_development(later, later.decision_date.max())


def test_real_late_2024_outcomes_enter_after_not_before_their_maturity():
    development, _ = observed()
    late = development[development.outcome_date.gt('2024-12-31')].sort_values('outcome_date').iloc[0]
    assert late.ipo_id not in set(available_development(development, late.outcome_date).ipo_id)
    assert late.ipo_id in set(available_development(development, late.outcome_date + pd.Timedelta(days=1)).ipo_id)


def test_predictions_do_not_read_later_labels_and_all_cases_remain():
    development, later = observed()
    calls, trace = asof_calls(development, later, 'filing_clock_response')
    # Alter evaluation labels only to probe isolation, never to train or score
    # an actual experiment on invented data.
    damaged = later.copy()
    damaged['day30_return_pct'] *= -1
    damaged['executed_hold_return_pct'] *= -1
    repeated, repeated_trace = asof_calls(development, damaged, 'filing_clock_response')
    np.testing.assert_array_equal(calls, repeated)
    assert trace == repeated_trace and len(calls) == len(trace) == 132
    assert max(row['training_n'] for row in trace) == 244
