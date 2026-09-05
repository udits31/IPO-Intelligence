"""Reproduce the deployed Module 1 paper tables from the frozen data snapshot.

This is a retrospective chronological replay, not a point-in-time trading backtest.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.model_pipeline import (
    EXCLUDED_NON_CORPORATE_IDS,
    fit_model,
    prepare_dataset,
    score_model,
    summarize,
)


def interval(values: list[float]) -> list[float]:
    return [round(float(x), 4) for x in np.percentile(values, [2.5, 97.5])]


def bootstrap(y: np.ndarray, p: np.ndarray, threshold: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    nonpositives = np.flatnonzero(y == 0)
    aucs, recalls, precisions = [], [], []
    for _ in range(2000):
        draw = np.r_[rng.choice(positives, len(positives)),
                     rng.choice(nonpositives, len(nonpositives))]
        outcome, probability = y[draw], p[draw]
        aucs.append(roc_auc_score(outcome, probability))
        calls = probability >= threshold
        recalls.append(np.mean(~calls[outcome == 0]))
        if calls.any():
            precisions.append(np.mean(outcome[calls]))
    return {
        'auc_95_pct': interval(aucs),
        'nonpositive_recall_95_pct': interval(recalls),
        'apply_precision_95_pct': interval(precisions),
    }


def reliability(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for low, high in zip(np.arange(0, 1, .2), np.arange(.2, 1.01, .2)):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            rows.append({
                'band': f'{low:.1f}-{high:.1f}', 'n': int(mask.sum()),
                'mean_probability': round(float(p[mask].mean()), 3),
                'positive_fraction': round(float(y[mask].mean()), 3),
            })
    return rows


def main() -> None:
    frame = prepare_dataset()
    cohort = frame[frame.year.between(2021, 2024)
                   & ~frame.id.isin(EXCLUDED_NON_CORPORATE_IDS)]
    result: dict = {
        'cohort': {
            'n': int(len(cohort)),
            'positive': int(cohort.target_binary.sum()),
            'nonpositive': int((1 - cohort.target_binary).sum()),
            'by_year': [],
            'mean_open_return_pct': round(float(cohort.target_return.mean()), 2),
            'median_open_return_pct': round(float(cohort.target_return.median()), 2),
            'min_open_return_pct': round(float(cohort.target_return.min()), 2),
            'max_open_return_pct': round(float(cohort.target_return.max()), 2),
        },
        'folds': {}, 'aggregates': {}, 'comparator': {}, 'analog_k_sensitivity': [],
    }
    for year, group in cohort.groupby('year'):
        result['cohort']['by_year'].append({
            'year': int(year), 'n': int(len(group)),
            'positive': int(group.target_binary.sum()),
            'nonpositive': int((1 - group.target_binary).sum()),
        })
    all_folds = {}
    xgb_folds = {}
    for year in (2022, 2023, 2024, 2025, 2026):
        train = cohort[cohort.year <= min(year - 1, 2024)]
        test = frame[frame.year == year]
        probability = score_model(*fit_model(train), test)
        all_folds[year] = (test.target_binary.to_numpy(), probability)
        comparator = XGBClassifier(
            n_estimators=80, max_depth=1, learning_rate=.05,
            min_child_weight=6, subsample=.9, colsample_bytree=1,
            reg_lambda=5, random_state=42, eval_metric='logloss', n_jobs=2,
        ).fit(train[['log_qib', 'log_retail', 'log_hni']], train.target_binary)
        xgb_folds[year] = comparator.predict_proba(test[['log_qib', 'log_retail', 'log_hni']])[:, 1]
    threshold = .82
    for year, (y, p) in all_folds.items():
        train = cohort[cohort.year <= min(year - 1, 2024)]
        row = summarize(y, p, threshold)
        row['train_n'] = int(len(train))
        row['log_loss'] = round(float(log_loss(y, p)), 4)
        row['prevalence_brier'] = round(float(brier_score_loss(y, np.full(len(y), train.target_binary.mean()))), 4)
        row.update(bootstrap(y, p, threshold, 8000 + year))
        result['folds'][str(year)] = row
        xgb_p = xgb_folds[year]
        rng = np.random.default_rng(12000 + year)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        differences = []
        for _ in range(2000):
            draw = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            differences.append(roc_auc_score(y[draw], p[draw]) - roc_auc_score(y[draw], xgb_p[draw]))
        result['comparator'][str(year)] = {
            'xgb_auc': round(float(roc_auc_score(y, xgb_p)), 4),
            'xgb_brier': round(float(brier_score_loss(y, xgb_p)), 4),
            'lr_minus_xgb_auc': round(float(roc_auc_score(y, p) - roc_auc_score(y, xgb_p)), 4),
            'lr_minus_xgb_auc_95_pct': interval(differences),
        }
    for label, years in {'primary_2023_2024': (2023, 2024),
                         'later_2025_2026': (2025, 2026)}.items():
        y = np.concatenate([all_folds[year][0] for year in years])
        p = np.concatenate([all_folds[year][1] for year in years])
        row = summarize(y, p, threshold)
        row['log_loss'] = round(float(log_loss(y, p)), 4)
        row.update(bootstrap(y, p, threshold, 9000 + years[0]))
        row['reliability'] = reliability(y, p)
        result['aggregates'][label] = row
    train = cohort[cohort.year == 2021]
    validation = cohort[cohort.year == 2022]
    features = ['log_qib', 'log_retail', 'log_hni']
    scaler = StandardScaler().fit(train[features])
    a, b = scaler.transform(train[features]), scaler.transform(validation[features])
    labels = train.target_binary.to_numpy()
    returns = train.target_return.to_numpy()
    target = validation.target_binary.to_numpy()
    for k in (5, 10, 15, 20, 25):
        indices = NearestNeighbors(n_neighbors=k).fit(a).kneighbors(b, return_distance=False)
        probability = labels[indices].mean(axis=1)
        median_return = np.median(returns[indices], axis=1)
        result['analog_k_sensitivity'].append({
            'k': k, 'auc': round(float(roc_auc_score(target, probability)), 4),
            'brier': round(float(brier_score_loss(target, probability)), 4),
            'median_return_mae_pct_points': round(float(mean_absolute_error(validation.target_return, median_return)), 2),
        })
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
