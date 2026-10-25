"""Research-only: can dated day-wise bids support the paper's final-book model?

All training rows are real IPO observations. The ridge step forecasts the
eventual book from a penultimate-day book; it does not create training rows.
Archived article tables may have been revised, so this is not a prospective
investor-time accuracy claim and never changes the deployed model bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ml.model_pipeline import EXCLUDED_NON_CORPORATE_IDS, FEATURE_COLS, fit_model, prepare_dataset
from ml.research_daywise_model import prepare as prepare_early


ROOT = Path(__file__).resolve().parents[1]
EARLY_PATH = ROOT / 'data/daywise_subscription_research_cohort.csv'
OUTPUT = ROOT / 'output/early_book_projection_research.json'
LEDGER = ROOT / 'output/early_book_projection_research_ledger.csv'
RAW = ['qib_subscription', 'retail_subscription', 'hni_subscription']


def metrics(target: np.ndarray, probability: np.ndarray) -> dict:
    label = probability >= .5
    return {
        'n': len(target),
        'positive_rate_pct': round(float(target.mean() * 100), 2),
        'accuracy_pct': round(float((label == target).mean() * 100), 2),
        'balanced_accuracy_pct': round(float(balanced_accuracy_score(target, label) * 100), 2),
        'negative_recall_pct': round(float(((~label) & (~target)).sum() / (~target).sum() * 100), 2),
        'brier': round(float(brier_score_loss(target, probability)), 4),
        'auc': round(float(roc_auc_score(target, probability)), 4),
    }


def run() -> dict:
    canonical = pd.read_csv(ROOT / 'data/canonical_listed_ipos.csv')
    archive = pd.read_csv(EARLY_PATH)
    early, audit = prepare_early(archive, canonical)
    final = prepare_dataset()
    final = final[final.year.between(2021, 2024) & ~final.id.isin(EXCLUDED_NON_CORPORATE_IDS)]
    if final[RAW].isna().any().any() or final.id.duplicated().any() or early.ipo_id.duplicated().any():
        raise ValueError('Missing final bids or duplicate IPO records')
    joined = early.merge(final[['id', *RAW, 'target_binary']], left_on='ipo_id', right_on='id',
                         suffixes=('_early', '_final'), validate='one_to_one')
    if len(joined) != len(early):
        raise ValueError('An early IPO lacks a paper-model final-book row')
    if (joined.target != joined.target_binary).any():
        raise ValueError('Opening outcome does not match the canonical label')
    fold_rows = []
    ledger = []
    for year in (2022, 2023, 2024):
        train_final = final[final.year < year]
        train_early = joined[joined.year < year]
        test = joined[joined.year == year].copy()
        if min(len(train_final), len(train_early), len(test)) < 20:
            raise ValueError(f'Insufficient year-separated cohort for {year}')
        model_scaler, model = fit_model(train_final)
        early_matrix = np.log1p(test[[f'{name}_early' for name in RAW]].to_numpy(dtype=float))
        train_matrix = np.log1p(train_early[[f'{name}_early' for name in RAW]].to_numpy(dtype=float))
        train_target = np.log1p(train_early[[f'{name}_final' for name in RAW]].to_numpy(dtype=float))
        projection_scaler = StandardScaler().fit(train_matrix)
        projection = Ridge(alpha=10.0).fit(projection_scaler.transform(train_matrix), train_target)
        projected_log = np.maximum(projection.predict(projection_scaler.transform(early_matrix)), 0.0)
        # The classifier and its original three features are unchanged. The
        # candidate only changes which real-time estimate enters those fields.
        direct_frame = pd.DataFrame(early_matrix, columns=FEATURE_COLS)
        projected_frame = pd.DataFrame(projected_log, columns=FEATURE_COLS)
        direct = model.predict_proba(model_scaler.transform(direct_frame))[:, 1]
        projected = model.predict_proba(model_scaler.transform(projected_frame))[:, 1]
        stage_scaler = StandardScaler().fit(train_matrix)
        stage_model = LogisticRegression(C=.5, max_iter=1000, random_state=42).fit(
            stage_scaler.transform(train_matrix), train_early.target.to_numpy())
        stage = stage_model.predict_proba(stage_scaler.transform(early_matrix))[:, 1]
        truth = test.target.to_numpy(dtype=bool)
        a, b = direct >= .5, projected >= .5
        gain, loss = int(((a != truth) & (b == truth)).sum()), int(((a == truth) & (b != truth)).sum())
        fold_rows.append({
            'test_year': year, 'final_book_train_n': len(train_final),
            'trajectory_train_n': len(train_early), 'test_n': len(test),
            'direct_final_book_classifier_on_early_bids': metrics(truth, direct),
            'projected_book_then_same_classifier': metrics(truth, projected),
            'early_stage_logistic_comparator': metrics(truth, stage),
            'always_positive_accuracy_pct': round(float(truth.mean() * 100), 2),
            'paired_projection_vs_direct': {
                'improved': gain, 'worsened': loss,
                'exact_two_sided_p': round(float(binomtest(min(gain, loss), gain + loss).pvalue), 4)
                if gain + loss else None,
            },
        })
        for index, row in enumerate(test.itertuples(index=False)):
            ledger.append({'ipo_id': int(row.ipo_id), 'year': year, 'outcome_positive': bool(truth[index]),
                           'direct_probability': float(direct[index]),
                           'projected_probability': float(projected[index]),
                           'stage_probability': float(stage[index])})
    ledger_frame = pd.DataFrame(ledger)
    ledger_frame.to_csv(LEDGER, index=False)
    blend_rows = []
    for projected_weight in (0.0, .25, .5, .75, 1.0):
        by_year = {}
        for year in (2022, 2023, 2024):
            sample = ledger_frame[ledger_frame.year == year]
            blended = (projected_weight * sample.projected_probability.to_numpy()
                       + (1 - projected_weight) * sample.direct_probability.to_numpy())
            by_year[str(year)] = metrics(sample.outcome_positive.to_numpy(dtype=bool), blended)
        blend_rows.append({'projected_weight': projected_weight, 'by_year': by_year,
                           'development_balanced_accuracy_mean_pct': round(
                               (by_year['2022']['balanced_accuracy_pct']
                                + by_year['2023']['balanced_accuracy_pct']) / 2, 2)})
    # Lock the blend on 2022-23 only; 2024 is read exactly once as the check.
    selected = max(blend_rows, key=lambda row: row['development_balanced_accuracy_mean_pct'])
    selected_2024 = selected['by_year']['2024']
    baseline_2024 = fold_rows[-1]['always_positive_accuracy_pct']
    passes_gate = (selected_2024['accuracy_pct'] > baseline_2024
                   and selected_2024['negative_recall_pct'] >= 30
                   and selected_2024['balanced_accuracy_pct'] >= 60)
    result = {
        'protocol': '2021-24 real penultimate-day archive; for each 2022-24 fold, fit paper-family final-book LR and separate ridge book projection only on earlier years; test held-out year.',
        'source_audit': audit,
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (EARLY_PATH, ROOT / 'data/canonical_listed_ipos.csv',
                                      ROOT / 'data/subscription_observations.csv')},
        'folds': fold_rows,
        'blend_sensitivity': blend_rows,
        'development_selected_projected_weight': selected['projected_weight'],
        'selected_2024_passes_accuracy_and_downside_gate': passes_gate,
        'ledger': str(LEDGER.relative_to(ROOT)),
        'limitations': [
            'Archived day-wise article tables may have been revised after the decision day; this is not an investor-time backtest.',
            'Penultimate-day end-of-day rows cannot validate 10:00-17:00 intraday predictions.',
            'Matched source coverage is incomplete and non-random; 2025-26 early-book archive is not in this cohort.',
            'Research only: neither the deployed model bundle nor the paper classifier has changed.',
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(run()['folds'], indent=2))
