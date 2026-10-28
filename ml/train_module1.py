"""Compatibility entry point for the audited Module 1 training pipeline."""

import json

from ml.model_pipeline import FEATURE_COLS, prepare_dataset, run_walk_forward_training

__all__ = ['FEATURE_COLS', 'prepare_dataset', 'run_walk_forward_training']


if __name__ == '__main__':
    benchmark, _ = run_walk_forward_training()
    print(json.dumps({
        'model': benchmark['model_name'],
        'threshold': benchmark['decision_threshold'],
        '2023_2024': benchmark['summary'],
        '2025_2026_retrospective': benchmark['later_year_summary'],
    }, indent=2))
