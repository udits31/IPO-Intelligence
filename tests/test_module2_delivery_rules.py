"""Public exchange delivery windows, corrected prices and early rejection."""
import numpy as np
import pandas as pd

from ml.calibrate_module2_rules import DEFAULT, decisions
from ml import research_module2_delivery_rules as study


def test_delivery_ends_before_decision_and_all_earlier_cases_remain():
    frame = study.assemble('development')
    assert len(frame) == 244
    observed = frame.delivery_status.eq('verified_prior_session_delivery')
    assert int(observed.sum()) == 213
    assert (pd.to_datetime(frame.loc[observed, 'latest_delivery_session']) <
            frame.loc[observed, 'decision_date']).all()
    assert frame.loc[observed, 'delivery_5_weighted_pct'].between(0, 100).all()
    assert frame.loc[observed, 'delivery_20_weighted_pct'].between(0, 100).all()
    assert frame.loc[~observed, 'delivery_5_weighted_pct'].isna().all()
    baseline = decisions(frame, DEFAULT)
    for mode in study.MODES:
        calls = study.calls(frame, mode)
        np.testing.assert_array_equal(calls[~observed.to_numpy()], baseline[~observed.to_numpy()])
        underwater = frame.decision_close.lt(frame.issue_price).to_numpy()
        assert not np.any(baseline[underwater] & ~calls[underwater])


def test_failed_earlier_delivery_gate_does_not_read_holdout(monkeypatch, tmp_path):
    original = study.assemble
    names = []

    def earlier_only(name):
        names.append(name)
        assert name == 'development'
        return original(name)

    monkeypatch.setattr(study, 'assemble', earlier_only)
    monkeypatch.setattr(study, 'REPORT', tmp_path/'delivery.json')
    report = study.run()
    assert names == ['development']
    assert report['selected']['mode'] == 'unchanged'
    assert report['earlier_gate_passed'] is False
    assert report['later_evaluated'] is False
    assert 'later' not in report
