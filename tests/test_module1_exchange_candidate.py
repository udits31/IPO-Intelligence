"""Same logistic model, fully verified training labels and no silent promotion."""
import json
from pathlib import Path

import numpy as np
import pytest

from ml import model_pipeline as pipeline


def test_exchange_candidate_cannot_overwrite_deployed_artifact():
    before = pipeline.BUNDLE_PATH.read_bytes()
    with pytest.raises(ValueError, match='must not silently overwrite'):
        pipeline.run_walk_forward_training(pipeline.BUNDLE_PATH, exchange_verified=True)
    assert pipeline.BUNDLE_PATH.read_bytes() == before


def test_saved_candidate_keeps_training_family_cohort_and_reports_pending_labels():
    candidate = json.loads(Path('output/module1_exchange_verified_candidate_bundle_20261022.json').read_text(encoding='utf-8'))
    deployed = json.loads(pipeline.BUNDLE_PATH.read_text(encoding='utf-8'))
    assert candidate['training_samples'] == deployed['training_samples'] == 252
    assert candidate['training_years'] == deployed['training_years'] == [2021, 2022, 2023, 2024]
    assert candidate['training_ipo_ids_sha256'] == deployed['training_ipo_ids_sha256']
    assert candidate['feature_cols'] == deployed['feature_cols'] == pipeline.FEATURE_COLS
    assert candidate['outcome_provenance']['candidate_only'] is True
    assert candidate['outcome_provenance']['training_class_correction_ids'] == [269, 275, 315, 346, 351, 371]
    assert len(candidate['outcome_provenance']['unverified_later_ids_pending_not_scored']) == 10
    assert candidate['benchmark']['threshold_selection']['years'] == [2022, 2023]
    assert candidate['decision_threshold'] == .53
    assert candidate['benchmark']['later_year_summary']['sample_size'] == 179


def test_candidate_coefficients_replay_exact_corrected_same_family_fit():
    from ml.audit_module1_listing_opens import verified_outcomes
    original = pipeline.prepare_dataset()
    train = original[original.year.between(2021, 2024) &
                     ~original.id.isin(pipeline.EXCLUDED_NON_CORPORATE_IDS)]
    train = train.merge(verified_outcomes()[['ipo_id', 'exchange_open']], left_on='id', right_on='ipo_id',
                        suffixes=('', '_audit'), validate='one_to_one')
    train['target_binary'] = train.exchange_open.gt(train.issue_price).astype(int)
    assert len(train) == 252
    scaler, model = pipeline.fit_model(train)
    candidate = json.loads(Path('output/module1_exchange_verified_candidate_bundle_20261022.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(scaler.mean_, candidate['scaler_mean'], atol=1e-10)
    np.testing.assert_allclose(model.coef_[0], candidate['coefficients'], atol=1e-5)
    np.testing.assert_allclose(model.intercept_[0], candidate['intercept'], atol=1e-5)
