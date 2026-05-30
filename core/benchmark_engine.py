"""Serve the exact audit bundled with the active Module 1 model."""

import json
from functools import lru_cache

from ml.model_pipeline import BUNDLE_PATH, MODEL_VERSION


@lru_cache(maxsize=1)
def build_walk_forward_benchmark():
    with BUNDLE_PATH.open(encoding='utf-8') as source:
        bundle = json.load(source)
    if bundle.get('model_version') != MODEL_VERSION:
        raise ValueError('Benchmark and active model versions disagree.')
    return bundle['benchmark']
