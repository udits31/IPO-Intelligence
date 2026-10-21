from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ml.collect_preclose_gmp import parse_gmp_history
from ml.return_pipeline import (evaluate_track, feature_frame, ranking_comparison,
                                prepare_returns, risk_score)
from services.return_forecast_service import make_return_forecast, rank_forecasts

IST = ZoneInfo('Asia/Kolkata')


def test_gmp_dates_amounts_and_missing_are_not_invented():
    content = '''<table><tr><th>Date</th><th>GMP</th><th>Kostak</th></tr>
      <tr><td>Today</td><td>₹90</td></tr>
      <tr><td>19 December</td><td>₹-</td></tr>
      <tr><td>18 December</td><td>₹0</td></tr>
      <tr><td>17 December</td><td>₹-10</td></tr>
      <tr><td>16 December</td><td>30%</td></tr></table>'''
    assert parse_gmp_history(content, '2021-12-20') == [
        {'gmp_date': '2021-12-17', 'gmp_rs': -10.},
        {'gmp_date': '2021-12-18', 'gmp_rs': 0.}]


def test_conflicting_gmp_rows_excluded():
    content = '<table><tr><td>Date</td><td>GMP</td></tr><tr><td>1 Jan</td><td>5</td></tr><tr><td>1 Jan</td><td>6</td></tr></table>'
    assert parse_gmp_history(content, '2024-01-02') == []


def test_feature_contract_has_no_outcomes_or_final_book():
    frame, _ = prepare_returns()
    x = feature_frame(frame)
    assert list(x) == ['log_qib_subscription', 'log_hni_subscription', 'log_retail_subscription', 'log_issue_price', 'log_issue_size_cr']
    paired = frame[frame.gmp_pct.notna()]
    assert (pd.to_datetime(paired.gmp_date) < pd.to_datetime(paired.decision_market_date)).all()
    assert set(frame.year) == {2021, 2022, 2023, 2024}
    assert not frame.ipo_id.duplicated().any()


def test_missing_bid_is_not_zero_imputation():
    frame, _ = prepare_returns()
    frame.loc[frame.index[0], 'qib_subscription'] = np.nan
    with pytest.raises(ValueError):
        feature_frame(frame)


def test_confirmation_outcome_never_selects_model_or_changes_its_predictions():
    frame, _ = prepare_returns()
    _, first, ledger = evaluate_track(frame)
    altered = frame.copy()
    altered.loc[altered.year == 2024, 'target_return'] = -80.
    _, second, changed = evaluate_track(altered)
    assert first['selected_candidate'] == second['selected_candidate']
    np.testing.assert_allclose(list(first['development_mae_pp'].values()), list(second['development_mae_pp'].values()), rtol=1e-12)
    np.testing.assert_allclose(ledger.prediction, changed.prediction)
    np.testing.assert_allclose(ledger.lower, changed.lower)


def candidate(name, *, score=10., source='2026-10-21T15:50:00+05:30', issued='2026-10-21T15:51:00+05:30'):
    return {'company_name': name, 'opens_at_ist': '2026-10-19T10:00:00+05:30',
            'cutoff_at_ist': '2026-10-21T17:00:00+05:30',
            'return_forecast': {'computed_at_ist': issued, 'source_asof_ist': source,
                                'priority_score_pp': score, 'ranking_eligible': True}}


def test_ranking_uses_return_and_downside_not_green_probability_or_input_order():
    assert float(risk_score(30, -50)) < float(risk_score(10, -2))
    now = datetime(2026, 10, 21, 16, tzinfo=IST)
    result = rank_forecasts([candidate('A', score=1), candidate('B', score=10)], now=now)
    assert result['top_priority'] == 'B'
    assert [r['rank'] for r in result['ranked']] == [1, 2]


def test_closed_stale_and_future_books_cannot_rank():
    rows = [candidate('stale', source='2026-10-21T15:00:00+05:30'),
            candidate('future', source='2026-10-21T16:01:00+05:30')]
    result = rank_forecasts(rows, now=datetime(2026, 10, 21, 16, tzinfo=IST))
    assert not result['ranked'] and len(result['excluded']) == 2
    closed = rank_forecasts([candidate('A')], now=datetime(2026, 10, 21, 17, tzinfo=IST))
    assert closed['skip_all'] and not closed['ranked']


def test_skip_all_possible_and_missing_not_ranked():
    missing = candidate('Missing'); missing['return_forecast'] = None
    result = rank_forecasts([candidate('A', score=-1), missing], now=datetime(2026, 10, 21, 16, tzinfo=IST))
    assert result['skip_all'] and result['top_priority'] is None
    assert result['ranked'][0]['priority_label'] == 'BELOW RISK HURDLE'


def test_ranking_benchmark_never_combines_different_date_opportunities():
    frame = pd.DataFrame([{'ipo_id': 1, 'company_name': 'A', 'decision_market_date': '2024-01-01',
                           'gmp_pct': 10, 'prediction': 12, 'lower': -5, 'target_return': 8},
                          {'ipo_id': 2, 'company_name': 'B', 'decision_market_date': '2024-01-02',
                           'gmp_pct': 15, 'prediction': 10, 'lower': -3, 'target_return': -4}])
    assert ranking_comparison(frame)['panels'] == 0


def test_service_cannot_issue_after_cutoff_or_score_future_input():
    item = {'company_name': 'Test', 'category': 'OPEN', 'issue_open_date': '2026-10-19',
            'issue_close_date': '2026-10-21', 'date_range': '19 - 21 Oct',
            'issue_price': 100, 'issue_size_cr': 400}
    values = dict(qib_subscription=2, retail_subscription=1, hni_subscription=3)
    assert make_return_forecast(item, values, source_asof='2026-10-21T16:55:00+05:30',
                               decision_at=datetime(2026, 10, 21, 17, tzinfo=IST)) is None
    assert make_return_forecast(item, values, source_asof='2026-10-21T16:55:00+05:30',
                               decision_at=datetime(2026, 10, 21, 16, tzinfo=IST)) is None


def test_service_returns_finite_frozen_forecast_with_distinct_version():
    item = {'company_name': 'Test', 'category': 'OPEN', 'issue_open_date': '2026-10-19',
            'issue_close_date': '2026-10-21', 'date_range': '19 - 21 Oct',
            'issue_price': 100, 'issue_size_cr': 400}
    result = make_return_forecast(item, dict(qib_subscription=2, retail_subscription=1, hni_subscription=3),
        source_asof='2026-10-21T15:50:00+05:30', decision_at=datetime(2026, 10, 21, 16, tzinfo=IST))
    if result is None:
        import json,sklearn
        from ml.return_pipeline import META
        metadata=json.loads(META.read_text())
        versions=metadata.get('runtime_versions',metadata.get('reproducibility',metadata.get('runtime',{})))
        if versions.get('sklearn')!=sklearn.__version__:
            pytest.skip('Archived optional return artifact requires its recorded scikit-learn runtime; direction stack is tested separately')
    assert result['model_version'].startswith('module1-preclose-return-')
    assert result['lower_return_pct'] <= result['central_return_pct'] <= result['upper_return_pct']
    assert 0 <= result['positive_listing_probability'] <= 1
    assert result['gmp_baseline_return_pct'] is None
