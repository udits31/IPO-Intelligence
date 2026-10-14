"""Real-price rule reachability and chronological risk-governance checks."""
import numpy as np
import pandas as pd
import pytest

from ml.calibrate_module2_rules import DEFAULT, GRID as PRICE_GRID, decisions
from ml.replay_module2_news_overlay import SOURCES
from ml.replay_module2_financial_rules import TIGHTER, PROFIT_TRAIL
from ml.research_module2_risk_governance import (
    GRID, assemble_risk, hindsight_bound, momentum_features, risk_calls, select)


def test_momentum_uses_only_actual_pre_checkpoint_prices():
    prices = pd.read_csv(SOURCES['holdout'][0], parse_dates=['date'])
    one = prices[prices.ipo_id.eq(501)].sort_values('date')
    checkpoint = one.iloc[30].date
    assert momentum_features(one, checkpoint) == momentum_features(one.iloc[:31], checkpoint)
    with pytest.raises(ValueError, match='31 observed'):
        momentum_features(one.iloc[:30], checkpoint)


def test_price_threshold_ceiling_cannot_be_mistaken_for_achieved_accuracy():
    later = assemble_risk('holdout')
    three, _ = hindsight_bound(later, [DEFAULT, TIGHTER, PROFIT_TRAIL])
    all_settings, _ = hindsight_bound(later, PRICE_GRID)
    assert three['hindsight_ceiling_correct'] == 81
    assert all_settings['hindsight_ceiling_correct'] == 94
    assert all_settings['n'] == 132 and all_settings['not_prediction_accuracy']
    assert all_settings['uses_future_outcomes_for_diagnostic_only']


def test_all_cases_kept_and_unknown_market_does_not_become_zero():
    later = assemble_risk('holdout')
    assert len(later) == 132
    # Deliberately remove observed test inputs to verify the missing-data gate;
    # this damaged frame is never an experiment's training/evaluation dataset.
    damaged = later.copy()
    damaged[['nifty_10_pct', 'nifty_30_pct']] = np.nan
    for parameters in GRID:
        assert len(risk_calls(later, parameters)) == 132
        np.testing.assert_array_equal(risk_calls(damaged, parameters), decisions(later, DEFAULT))


def test_later_and_unmatured_outcomes_cannot_choose_the_new_rule():
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(assemble_risk('holdout'), '2024-12-31')
    development = assemble_risk('development')
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(development, '2024-12-31')
    known = development[development.outcome_date.le('2024-12-31')]
    chosen, candidates = select(known, '2024-12-31')
    assert chosen in candidates and len(candidates) == 9 and len(known) == 214
