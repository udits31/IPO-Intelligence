"""Reproducible Module 1 model and chronological evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


DATA_FILES = (Path('data/canonical_listed_ipos.csv'), Path('data/subscription_observations.csv'))
BUNDLE_PATH = Path('ml/model_bundle.json')
MODEL_VERSION = 'module1-corporate-subscription-lr-2021-2024-v2'
FEATURE_COLS = ['log_qib', 'log_retail', 'log_hni']
TRAIN_YEARS = (2021, 2022, 2023, 2024)
EXCLUDED_NON_CORPORATE_IDS = {293, 307}
EXPECTED_TRAIN_ROWS = 252
AVOID_THRESHOLD = 0.50
# A pre-specified statistical direction cutoff, separate from the
# development-selected, downside-sensitive application threshold.
DIRECTION_THRESHOLD = 0.50


def prepare_dataset() -> pd.DataFrame:
    """Join final-book observations to verified exchange opening prices."""
    listed = pd.read_csv(DATA_FILES[0])
    subscriptions = pd.read_csv(DATA_FILES[1])
    if listed['id'].duplicated().any() or subscriptions['ipo_id'].duplicated().any():
        raise ValueError('Duplicate IPO identifiers would duplicate training examples.')
    frame = listed.merge(
        subscriptions[['ipo_id', 'qib_subscription', 'retail_subscription', 'hni_subscription']],
        left_on='id', right_on='ipo_id', how='left', validate='one_to_one',
    )
    frame['listing_date'] = pd.to_datetime(frame['listing_date'], errors='coerce')
    frame['year'] = frame['listing_date'].dt.year
    sources = ['qib_subscription', 'retail_subscription', 'hni_subscription']
    frame['missing_subscription_count'] = frame[sources].isna().sum(axis=1)
    for source, feature in zip(sources, FEATURE_COLS):
        # The two 2021 retail gaps belong to non-corporate trusts and are
        # excluded from training. This fill keeps the broader audit frame
        # numeric; no missing bid reaches the promoted training cohort.
        frame[feature] = np.log1p(frame[source].fillna(1.0).clip(lower=0.01))
    price = pd.to_numeric(frame['issue_price'], errors='coerce')
    open_price = pd.to_numeric(frame['listing_open_price'], errors='coerce')
    frame['target_return'] = (open_price / price - 1.0) * 100.0
    frame['target_binary'] = (frame['target_return'] > 0).astype(int)
    excluded = frame[frame['id'].isin(EXCLUDED_NON_CORPORATE_IDS)]
    if set(excluded['id']) != EXCLUDED_NON_CORPORATE_IDS or not excluded['company_name'].str.contains('Trust', case=False).all():
        raise ValueError('The non-corporate trust exclusions no longer match their audited records.')
    cohort = frame[frame['year'].between(TRAIN_YEARS[0], TRAIN_YEARS[-1])
                   & ~frame['id'].isin(EXCLUDED_NON_CORPORATE_IDS)]
    if len(cohort) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f'Expected {EXPECTED_TRAIN_ROWS} IPOs in 2021–24; found {len(cohort)}.')
    if cohort[['issue_price', 'listing_open_price', *FEATURE_COLS]].isna().any().any():
        raise ValueError('The training cohort has missing prices or model features.')
    if (cohort['issue_price'] <= 0).any() or (cohort['listing_open_price'] <= 0).any():
        raise ValueError('The training cohort has a nonpositive issue or opening price.')
    recorded = pd.to_numeric(cohort['listing_return_pct'], errors='coerce')
    if ((recorded - cohort['target_return']).abs() > 0.01).any():
        raise ValueError('Recorded returns disagree with exchange opening prices.')
    return frame


def fit_model(train: pd.DataFrame):
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    model = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
    model.fit(scaler.transform(train[FEATURE_COLS]), train['target_binary'].to_numpy())
    return scaler, model


def score_model(scaler, model, test: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(test[FEATURE_COLS]))[:, 1]


def bootstrap_models(cohort: pd.DataFrame, repeats: int = 200) -> list[dict]:
    """Year-stratified refits for sensitivity, not a calibrated confidence interval."""
    rng = np.random.default_rng(20261007)
    groups = [cohort[cohort['year'] == year] for year in TRAIN_YEARS]
    result = []
    for _ in range(repeats):
        sample = pd.concat([group.iloc[rng.integers(0, len(group), len(group))] for group in groups])
        scaler, model = fit_model(sample)
        result.append({
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
        })
    return result


def summarize(y_true, positive_probability, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(positive_probability, dtype=float)
    predicted_positive = p >= threshold
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, predicted_positive, labels=[0, 1]).ravel()]
    return {
        'sample_size': int(len(y)), 'positive_listings': int((y == 1).sum()),
        'nonpositive_listings': int((y == 0).sum()), 'threshold': round(float(threshold), 3),
        'roc_auc': round(float(roc_auc_score(y, p)), 4),
        'brier_score': round(float(brier_score_loss(y, p)), 4),
        'classifier_accuracy_pct': round(float((predicted_positive == y).mean() * 100), 1),
        'always_positive_baseline_pct': round(float(y.mean() * 100), 1),
        'balanced_accuracy_pct': round(float((tn / (tn + fp) + tp / (tp + fn)) * 50), 1),
        'negative_recall_pct': round(float(tn / (tn + fp) * 100), 1),
        'positive_recall_pct': round(float(tp / (tp + fn) * 100), 1),
        'apply_precision_pct': round(float(tp / (tp + fp) * 100), 1) if tp + fp else None,
        'warning_precision_pct': round(float(tn / (tn + fn) * 100), 1) if tn + fn else None,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def choose_apply_threshold(dev_y, dev_p) -> tuple[float, dict]:
    """Use only the 2022 and 2023 replay to set the risk threshold."""
    options = (summarize(dev_y, dev_p, step / 100) for step in range(35, 86))
    best = max(options, key=lambda row: (row['balanced_accuracy_pct'], row['apply_precision_pct'] or 0))
    return best['threshold'], best


def tier_statistics(frame: pd.DataFrame, probabilities: np.ndarray, lo: float, hi: float) -> dict:
    selected = (probabilities >= lo) & (probabilities < hi)
    values = frame['target_return'].to_numpy(dtype=float)[selected]
    labels = frame['target_binary'].to_numpy(dtype=int)[selected]
    if not len(values):
        return {'sample_size': 0, 'median_return_pct': None, 'range_50_pct': None,
                'range_80_pct': None, 'downside_p10_pct': None, 'win_rate_pct': None}
    return {
        'sample_size': int(len(values)),
        'median_return_pct': round(float(np.median(values)), 1),
        'range_50_pct': [round(float(v), 1) for v in np.percentile(values, [25, 75])],
        'range_80_pct': [round(float(v), 1) for v in np.percentile(values, [10, 90])],
        'downside_p10_pct': round(float(np.percentile(values, 10)), 1),
        'win_rate_pct': round(float(labels.mean() * 100), 1),
    }


def record_rows(test: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> list[dict]:
    records = []
    for (_, row), p in zip(test.iterrows(), probabilities):
        actual_positive = bool(row['target_binary'])
        records.append({
            'ipo_id': int(row['id']), 'company_name': str(row['company_name']),
            'nse_symbol': str(row['nse_symbol']) if pd.notna(row.get('nse_symbol')) else None,
            'listing_date': row['listing_date'].strftime('%Y-%m-%d'),
            'test_year': int(row['year']), 'trained_through': int(row['year']) - 1,
            'issue_price': round(float(row['issue_price']), 2),
            'actual_open_price': round(float(row['listing_open_price']), 2),
            'actual_open_return_pct': round(float(row['target_return']), 2),
            'positive_probability_pct': round(float(p) * 100, 1),
            'risk_call': 'APPLY' if p >= threshold else ('ELEVATED RISK' if p < AVOID_THRESHOLD else 'CAUTION'),
            'actual_result': 'Positive' if actual_positive else 'Flat or discount',
            'direction_correct': bool((p >= threshold) == actual_positive),
        })
    return records


def build_model_bundle(*, exchange_verified: bool = False) -> dict:
    frame = prepare_dataset()
    bundle_version = MODEL_VERSION
    outcome_provenance = None
    if exchange_verified:
        from ml.audit_module1_listing_opens import OUTPUT as outcome_audit, verified_outcomes
        audit = verified_outcomes()
        eligible = frame[frame.year.between(2021, 2026) &
                         ~frame.id.isin(EXCLUDED_NON_CORPORATE_IDS)].copy()
        required = set(eligible.loc[eligible.year.le(2024), 'id'])
        frame = eligible.merge(audit[['ipo_id', 'exchange_open', 'exchange']],
                               left_on='id', right_on='ipo_id', suffixes=('', '_audit'),
                               validate='one_to_one')
        if set(frame.loc[frame.year.le(2024), 'id']) != required or len(required) != EXPECTED_TRAIN_ROWS:
            raise ValueError('Verified candidate requires every original training IPO; no selective subset')
        original_labels = frame.target_binary.copy()
        frame['listing_open_price'] = frame.exchange_open
        frame['target_return'] = (frame.exchange_open / frame.issue_price - 1) * 100
        frame['target_binary'] = frame.target_return.gt(0).astype(int)
        changed = frame[frame.year.le(2024) & original_labels.ne(frame.target_binary)]
        bundle_version = MODEL_VERSION + '-exchange-outcome-candidate'
        outcome_provenance = {
            'audit_path': str(outcome_audit),
            'audit_sha256': hashlib.sha256(outcome_audit.read_bytes()).hexdigest(),
            'authority': 'Original exchange daily archives rechecked offline for hash, date, identity and opening price',
            'selection': 'NSE listing open when eligible; exact-scrip BSE otherwise',
            'training_class_correction_ids': sorted(int(v) for v in changed.id),
            'unverified_later_ids_pending_not_scored': sorted(int(v) for v in set(eligible.id)-set(frame.id)),
            'candidate_only': True,
        }
    cohort = frame[frame['year'].between(TRAIN_YEARS[0], TRAIN_YEARS[-1])
                   & ~frame['id'].isin(EXCLUDED_NON_CORPORATE_IDS)].copy()
    folds = {}
    for year in (2022, 2023, 2024, 2025, 2026):
        train_through = min(year - 1, 2024)
        train = cohort[cohort['year'] <= train_through]
        test = frame[frame['year'] == year].copy()
        scaler, model = fit_model(train)
        folds[year] = (test, score_model(scaler, model, test), int(len(train)))
    dev_y = np.concatenate([folds[y][0]['target_binary'].to_numpy() for y in (2022, 2023)])
    dev_p = np.concatenate([folds[y][1] for y in (2022, 2023)])
    threshold, development_metrics = choose_apply_threshold(dev_y, dev_p)
    splits = []
    for year in (2022, 2023, 2024, 2025, 2026):
        test, p, train_n = folds[year]
        splits.append({
            'test_year': year, 'trained_through': min(year - 1, 2024),
            'train_samples': train_n, 'test_samples': int(len(test)),
            'role': 'threshold_selection' if year in (2022, 2023) else (
                'retrospective_confirmation' if year == 2024 else 'later_year_retrospective'),
            **summarize(test['target_binary'], p, threshold),
        })
    main_y = np.concatenate([folds[y][0]['target_binary'].to_numpy() for y in (2023, 2024)])
    main_p = np.concatenate([folds[y][1] for y in (2023, 2024)])
    later_y = np.concatenate([folds[y][0]['target_binary'].to_numpy() for y in (2025, 2026)])
    later_p = np.concatenate([folds[y][1] for y in (2025, 2026)])
    tier_frame = pd.concat([folds[y][0] for y in (2022, 2023, 2024)])
    tier_p = np.concatenate([folds[y][1] for y in (2022, 2023, 2024)])
    tiers = {
        'skip': tier_statistics(tier_frame, tier_p, 0, AVOID_THRESHOLD),
        'caution': tier_statistics(tier_frame, tier_p, AVOID_THRESHOLD, threshold),
        'apply': tier_statistics(tier_frame, tier_p, threshold, 1.00001),
    }
    scaler, model = fit_model(cohort)
    input_reference = {
        source: {
            'p25': round(float(cohort[source].quantile(.25)), 2),
            'median': round(float(cohort[source].median()), 2),
        }
        for source in ('qib_subscription', 'retail_subscription', 'hni_subscription')
    }
    benchmark = {
        'methodology': 'Chronological replay: each IPO is scored by a model trained on earlier listing years.',
        'target': 'Positive exchange opening price relative to IPO issue price',
        'gmp_used': False,
        'gmp_note': 'GMP is excluded because historical point-in-time observations are not verified.',
        'model_name': 'Subscription-only logistic regression', 'model_version': bundle_version,
        'decision_threshold': threshold, 'avoid_threshold': AVOID_THRESHOLD,
        'direction_threshold': DIRECTION_THRESHOLD,
        'direction_summary': summarize(main_y, main_p, DIRECTION_THRESHOLD),
        'later_year_direction_summary': summarize(later_y, later_p, DIRECTION_THRESHOLD),
        'direction_splits': [
            {'test_year': year, 'trained_through': min(year - 1, 2024),
             **summarize(folds[year][0]['target_binary'], folds[year][1], DIRECTION_THRESHOLD)}
            for year in (2022, 2023, 2024, 2025, 2026)
        ],
        'threshold_selection': {'years': [2022, 2023], 'objective': 'balanced accuracy',
                                'metrics': development_metrics},
        'summary': summarize(main_y, main_p, threshold),
        'later_year_summary': summarize(later_y, later_p, threshold),
        'splits': splits,
        'records': sorted(
            [record for year in (2023, 2024) for record in record_rows(folds[year][0], folds[year][1], threshold)],
            key=lambda record: record['listing_date'], reverse=True,
        ),
        'data_quality': {
            'historical_subscription_fills': int(cohort['missing_subscription_count'].sum()),
            'excluded_non_corporate_ids': sorted(EXCLUDED_NON_CORPORATE_IDS),
            'note': 'Two trust offerings with missing retail books are excluded from the corporate training cohort. No training bid is imputed. Source publication timestamps include placeholders; these tests are retrospective.',
        },
        'diagnosis': 'A warning is a risk flag, not a prediction that a discount is certain. Compare downside recall with false warnings.',
    }
    if outcome_provenance is not None:
        benchmark['outcome_provenance'] = outcome_provenance
    if benchmark['summary']['negative_recall_pct'] <= 0 or benchmark['later_year_summary']['negative_recall_pct'] <= 0:
        raise ValueError('Model promotion stopped: negative listings were not flagged.')
    return {
        'schema_version': 1, 'model_version': bundle_version,
        'outcome_provenance': outcome_provenance,
        'model_name': 'Subscription-only logistic regression',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_years': list(TRAIN_YEARS), 'training_samples': int(len(cohort)),
        'training_ipo_ids_sha256': hashlib.sha256(','.join(str(v) for v in sorted(cohort['id'])).encode()).hexdigest(),
        'excluded_non_corporate_ids': sorted(EXCLUDED_NON_CORPORATE_IDS),
        'input_sha256': {str(path).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in DATA_FILES},
        'feature_cols': FEATURE_COLS,
        'feature_source': 'Final QIB, retail, and HNI subscription multiples, available after book closure.',
        'training_input_reference': input_reference,
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'coefficients': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
        'bootstrap_refits': bootstrap_models(cohort),
        'bootstrap_note': '10th–90th percentile across 200 year-stratified training resamples; sensitivity only, not a confidence or prediction interval.',
        'decision_threshold': threshold, 'avoid_threshold': AVOID_THRESHOLD,
        'empirical_tiers': tiers, 'benchmark': benchmark,
        'limitations': [
            'Candidate model family was explored using historical outcomes, including later years; later-year checks are retrospective.',
            'Historical subscription publication timestamps include placeholders and need independent source verification.',
            'No listing-return regression estimate is promoted; its sign did not beat the positive-return baseline.',
        ],
    }


def run_walk_forward_training(output_path: Path = BUNDLE_PATH, *, exchange_verified: bool = False) -> tuple[dict, dict]:
    if exchange_verified and Path(output_path).resolve() == BUNDLE_PATH.resolve():
        raise ValueError('Exchange-verified candidate must not silently overwrite the deployed bundle')
    bundle = build_model_bundle(exchange_verified=exchange_verified)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, output_path)
    return bundle['benchmark'], {'feature_cols': FEATURE_COLS, 'coefficients': bundle['coefficients']}


if __name__ == '__main__':
    benchmark, _ = run_walk_forward_training()
    print(json.dumps({
        'model': benchmark['model_name'], 'threshold': benchmark['decision_threshold'],
        '2023_2024': benchmark['summary'],
        '2025_2026_retrospective': benchmark['later_year_summary'],
    }, indent=2))
