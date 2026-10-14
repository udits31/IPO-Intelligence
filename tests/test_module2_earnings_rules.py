"""Observed financial cases and invariant guards; never synthetic training data."""
import json

import numpy as np
import pytest

from ml.calibrate_module2_rules import DEFAULT, decisions
from ml.recheck_module2_financial_comparatives import OUTPUT
from ml.replay_module2_earnings_rules import GRID, assemble_earnings, earnings_calls, earnings_masks, select


def observed(name):
    if not OUTPUT.exists():
        pytest.skip('Actual financial-source extraction is not installed')
    return assemble_earnings(name, {r['ipo_id']: r for r in
        (json.loads(s) for s in OUTPUT.read_text(encoding='utf-8').splitlines())})


def test_actual_decline_growth_and_prior_loss_are_different_evidence():
    frame = observed('development')
    down, up = earnings_masks(frame, 90)
    positions = {ipo_id: i for i, ipo_id in enumerate(frame.ipo_id)}
    assert down[positions[313]] and not up[positions[313]]  # ELIN: both declined.
    assert up[positions[248]] and not down[positions[248]]  # SUPRIYA: both grew.
    assert not up[positions[276]]  # NUVOCO: prior loss, not ordinary profit growth.


def test_every_later_case_retained_and_missing_earnings_keep_original_rules():
    frame = observed('holdout')
    assert len(frame) == 132
    missing = frame.revenue_current_inr.isna().to_numpy()
    baseline = decisions(frame, DEFAULT)
    for parameters in GRID:
        calls = earnings_calls(frame, parameters)
        assert len(calls) == len(frame)
        np.testing.assert_array_equal(calls[missing], baseline[missing])


def test_later_or_not_yet_mature_outcomes_cannot_choose_settings():
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(observed('holdout'), '2024-12-31')
    development = observed('development')
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(development, '2024-12-31')
    known = development[development.outcome_date.le('2024-12-31')]
    chosen, candidates = select(known, '2024-12-31')
    assert chosen in candidates and len(candidates) == 7 and len(known) == 214
