"""Real pre-close books: time boundary, matched identities and honest baseline."""
import json
from pathlib import Path

import pandas as pd


def report():
    return json.loads(Path('output/module1_stage_transfer_research_20261022.json').read_text(encoding='utf-8'))


def test_every_training_outcome_precedes_first_preclose_decision():
    data = report()
    source = pd.read_csv('data/canonical_listed_ipos.csv', parse_dates=['listing_date']).set_index('id')
    for trace in data['trace']:
        assert pd.Timestamp(trace['latest_training_listing']) < pd.Timestamp(trace['first_test_decision'])
        actual = source.loc[trace['training_ipo_ids']]
        assert actual.listing_date.lt(pd.Timestamp(trace['first_test_decision'])).all()
        assert actual.listing_date.dt.year.lt(trace['test_year']).all()


def test_stage_comparison_has_identical_training_ids_and_test_ipos():
    data = report()
    traces = {(r['training_stage'], r['test_year']): r for r in data['trace']}
    for year in (2022, 2023, 2024):
        assert traces['matched_id_final_book_training', year]['training_ipo_ids'] == traces['matched_early_book_training', year]['training_ipo_ids']
    ledger = pd.read_csv('output/module1_stage_transfer_research_20261022_ledger.csv')
    reference = None
    for _, rows in ledger.groupby('training_stage'):
        ids = sorted(rows.ipo_id)
        assert not rows.ipo_id.duplicated().any()
        if reference is None:
            reference = ids
        assert ids == reference
        assert pd.to_datetime(rows.decision_market_date).lt(pd.to_datetime(rows.issue_close_date)).all()


def test_thresholds_use_earlier_predictions_not_2024_outcomes():
    from ml.model_pipeline import choose_apply_threshold
    data = report()
    ledger = pd.read_csv('output/module1_stage_transfer_research_20261022_ledger.csv')
    for stage, rows in ledger.groupby('training_stage'):
        earlier = rows[rows.year.isin([2022, 2023])]
        threshold, _ = choose_apply_threshold(earlier.target_binary, earlier.probability)
        assert threshold == data['results'][stage]['threshold_selected_on_2022_2023']
        score = data['results'][stage]['confirmation_2024']['fixed_direction']
        assert score['sample_size'] == 75
        assert score['always_positive_baseline_pct'] == 81.3
    assert data['live_model_changed'] is False
