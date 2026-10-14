"""Real earlier IPOs establish the stop-before-later validation contract."""
import numpy as np
import pytest

from ml.calibrate_module2_rules import DEFAULT, decisions
from ml.research_module2_risk_governance import assemble_risk
from ml import research_module2_hold_evidence as study


def test_hold_gate_never_weakens_an_existing_exit():
    development = assemble_risk('development')
    baseline = decisions(development, DEFAULT)
    for mode in study.MODES:
        predicted = study.calls(development, mode)
        assert not np.any(baseline & ~predicted)
        if mode == 'unchanged':
            np.testing.assert_array_equal(predicted, baseline)
        else:
            column = 'stock_30_pct' if mode == 'positive_30_session_stock' else 'stock_10_pct'
            missing = development[column].isna().to_numpy(copy=True)
            if mode.endswith('market_relative'):
                missing |= development.nifty_10_pct.isna().to_numpy()
            np.testing.assert_array_equal(predicted[missing], baseline[missing])


def test_confirmation_or_unmatured_outcomes_cannot_select_policy():
    development = assemble_risk('development')
    with pytest.raises(ValueError, match='matured 2021-2023'):
        study.select(development, '2024-12-31')
    early = development[development.listing_year.le(2023)]
    with pytest.raises(ValueError, match='matured 2021-2023'):
        study.select(early, early.outcome_date.min())


def test_failed_real_confirmation_never_loads_later_prices(monkeypatch, tmp_path):
    original = study.assemble_risk
    accessed = []

    def earlier_only(name):
        accessed.append(name)
        assert name == 'development', 'Failed validation must not open later data'
        return original(name)

    monkeypatch.setattr(study, 'assemble_risk', earlier_only)
    monkeypatch.setattr(study, 'REPORT', tmp_path/'result.json')
    report = study.run()
    assert accessed == ['development']
    assert report['confirmation_n'] == 61
    assert report['confirmation']['challenger']['correct'] == 28
    assert report['earlier_validation_gate_passed'] is False
    assert report['later_evaluated'] is False
    assert 'later' not in report
