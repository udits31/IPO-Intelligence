"""Protect the promoted model's cohort, serialization, and reported replay."""

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from ml.model_pipeline import DATA_FILES, MODEL_VERSION, build_model_bundle, fit_model, prepare_dataset


class ModelBundleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads(Path('ml/model_bundle.json').read_text(encoding='utf-8'))
        cls.rebuilt = build_model_bundle()
        cls.data = prepare_dataset()

    def test_training_cohort_and_input_fingerprints(self):
        self.assertEqual(self.saved['model_version'], MODEL_VERSION)
        self.assertEqual(self.saved['training_years'], [2021, 2022, 2023, 2024])
        self.assertEqual(self.saved['training_samples'], 252)
        self.assertEqual(self.saved['excluded_non_corporate_ids'], [293, 307])
        self.assertEqual(len(self.saved['bootstrap_refits']), 200)
        # BLAS/scikit-learn can differ by a few final floating-point bits across
        # runs. The model contract is numerical reproducibility, not JSON-bit
        # identity for 200 resampled refits.
        for saved_refit, rebuilt_refit in zip(self.saved['bootstrap_refits'], self.rebuilt['bootstrap_refits']):
            self.assertEqual(saved_refit.keys(), rebuilt_refit.keys())
            for key in ('mean', 'scale', 'coef'):
                np.testing.assert_allclose(saved_refit[key], rebuilt_refit[key], rtol=0, atol=1e-12)
            self.assertAlmostEqual(saved_refit['intercept'], rebuilt_refit['intercept'], places=12)
        for path in DATA_FILES:
            self.assertEqual(
                self.saved['input_sha256'][str(path).replace('\\', '/')],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_saved_model_is_exact_fit_of_training_cohort(self):
        cohort = self.data[self.data['year'].between(2021, 2024)
                           & ~self.data['id'].isin([293, 307])]
        scaler, model = fit_model(cohort)
        np.testing.assert_allclose(self.saved['scaler_mean'], scaler.mean_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.saved['scaler_scale'], scaler.scale_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.saved['coefficients'], model.coef_[0], rtol=0, atol=1e-12)
        self.assertAlmostEqual(self.saved['intercept'], model.intercept_[0], places=12)

    def test_replay_is_generated_not_entered_by_hand(self):
        self.assertEqual(self.saved['benchmark'], self.rebuilt['benchmark'])
        summary = self.saved['benchmark']['summary']
        later = self.saved['benchmark']['later_year_summary']
        self.assertEqual(summary['sample_size'], 149)
        self.assertEqual(later['sample_size'], 189)
        self.assertEqual(summary['tn'] + summary['fp'], summary['nonpositive_listings'])
        self.assertEqual(later['tn'] + later['fp'], later['nonpositive_listings'])
        self.assertGreater(summary['negative_recall_pct'], 0)
        self.assertGreater(later['negative_recall_pct'], 0)
        direction = self.saved['benchmark']['later_year_direction_summary']
        self.assertEqual(direction['sample_size'], later['sample_size'])
        self.assertEqual(direction['threshold'], 0.5)
        self.assertEqual(direction['classifier_accuracy_pct'], 76.2)
        self.assertLess(direction['negative_recall_pct'], later['negative_recall_pct'])
        splits = self.saved['benchmark']['direction_splits']
        self.assertEqual([part['test_year'] for part in splits], [2022, 2023, 2024, 2025, 2026])
        self.assertEqual(sum(part['sample_size'] for part in splits[-2:]), later['sample_size'])


if __name__ == '__main__':
    unittest.main()
