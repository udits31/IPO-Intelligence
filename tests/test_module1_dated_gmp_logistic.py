"""Actual paired pre-close observations, identical comparisons, no baseline inflation."""
import json
from pathlib import Path

import pandas as pd


def evidence():
    stem = 'output/module1_dated_gmp_logistic_research_20261022'
    return json.loads(Path(stem+'.json').read_text(encoding='utf-8')), pd.read_csv(stem+'_ledger.csv')


def test_dated_quotes_precede_decision_and_all_variants_use_same_cases():
    report, ledger = evidence()
    assert pd.to_datetime(ledger.gmp_date).lt(pd.to_datetime(ledger.decision_market_date)).all()
    reference = None
    for _, rows in ledger.groupby('variant'):
        assert not rows.ipo_id.duplicated().any()
        ids = sorted(rows.ipo_id)
        if reference is None:
            reference = ids
        assert ids == reference
    for year in (2022, 2023, 2024):
        traces = [r for r in report['traces'] if r['year'] == year]
        assert len(traces) == 3
        assert all(r['training_ids'] == traces[0]['training_ids'] for r in traces)
        assert all(pd.Timestamp(r['latest_training_outcome']) < pd.Timestamp(r['first_test_decision']) for r in traces)


def test_report_does_not_turn_majority_prediction_into_an_edge():
    report, ledger = evidence()
    for variant, result in report['results'].items():
        rows = ledger[ledger.variant.eq(variant) & ledger.year.eq(2024)]
        assert len(rows) == 39 and rows.target_binary.sum() == 34
        score = result['confirmation_2024']['fixed_direction']
        assert score['classifier_accuracy_pct'] == score['always_positive_baseline_pct'] == 87.2
        assert score['balanced_accuracy_pct'] == 50
        assert score['negative_recall_pct'] == 0
    assert report['live_model_changed'] is False
    assert report['combined_features_change_paper_input_count'] is True


def test_threshold_selection_contains_only_earlier_folds():
    from ml.model_pipeline import choose_apply_threshold
    report, ledger = evidence()
    for name, rows in ledger.groupby('variant'):
        earlier = rows[rows.year.isin([2022, 2023])]
        threshold, _ = choose_apply_threshold(earlier.target_binary, earlier.probability)
        assert threshold == report['results'][name]['earlier_selected_threshold']
