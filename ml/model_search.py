"""Read-only, chronological candidate comparison for Module 1.

All candidates use fields available by final book closure. Historical source
timestamps are not independently audited, so these are retrospective results.
This script does not change the promoted model bundle.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml.model_pipeline import fit_model, prepare_dataset, score_model, summarize


SUBS = ['log_qib', 'log_retail', 'log_hni']
FIVE = [*SUBS, 'log_issue_size', 'log_issue_price']
YEARS = [2022, 2023, 2024, 2025, 2026]


def candidates():
    return {
        'lr3_c005': (SUBS, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=1000))),
        'lr3_c05': (SUBS, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=1000))),
        'lr3_c5': (SUBS, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=5, max_iter=1000))),
        'lr5_c01': (FIVE, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))),
        'lr5_c1': (FIVE, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))),
        'poly3_c005': (SUBS, lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(C=.05, max_iter=1000))),
        'poly3_c05': (SUBS, lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(C=.5, max_iter=1000))),
        'hgb3_d2': (SUBS, lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=.05, min_samples_leaf=10, random_state=42)),
        'hgb5_d2': (FIVE, lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=.05, min_samples_leaf=10, random_state=42)),
        'rf3': (SUBS, lambda: RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, max_features=2, random_state=42, n_jobs=2)),
        'xgb3_d1': (SUBS, lambda: XGBClassifier(n_estimators=80, max_depth=1, learning_rate=.05, min_child_weight=6, subsample=.9, colsample_bytree=1, reg_lambda=5, random_state=42, eval_metric='logloss', n_jobs=2)),
        'xgb5_d2': (FIVE, lambda: XGBClassifier(n_estimators=80, max_depth=2, learning_rate=.05, min_child_weight=6, subsample=.9, colsample_bytree=.9, reg_lambda=5, random_state=42, eval_metric='logloss', n_jobs=2)),
        'lda3': (SUBS, lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))),
        'qda3': (SUBS, lambda: make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=.4))),
        'svc3_rbf': (SUBS, lambda: make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=.5, kernel='rbf', random_state=42), cv=3, ensemble=False))),
        'svc5_rbf': (FIVE, lambda: make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=.5, kernel='rbf', random_state=42), cv=3, ensemble=False))),
    }


def evaluate():
    frame = prepare_dataset().copy()
    frame['log_issue_size'] = np.log1p(frame['issue_size_cr'].clip(lower=0))
    frame['log_issue_price'] = np.log1p(frame['issue_price'].clip(lower=0))
    predictions = defaultdict(dict)
    labels = {}
    for year in YEARS:
        train = frame[frame.year.between(2021, min(year - 1, 2024))
                      & ~frame.id.isin([293, 307])]
        test = frame[frame.year == year]
        labels[year] = test.target_binary.to_numpy()
        for name, (features, factory) in candidates().items():
            model = factory().fit(train[features], train.target_binary)
            predictions[name][year] = model.predict_proba(test[features])[:, 1]
    for year in YEARS:
        predictions['blend_lr_hgb'][year] = .7 * predictions['lr3_c05'][year] + .3 * predictions['hgb3_d2'][year]
        predictions['blend_lr_xgb'][year] = .7 * predictions['lr3_c05'][year] + .3 * predictions['xgb3_d1'][year]
        predictions['blend_lr3_lr5'][year] = .5 * predictions['lr3_c05'][year] + .5 * predictions['lr5_c1'][year]
        predictions['blend_lr3_poly'][year] = .5 * predictions['lr3_c05'][year] + .5 * predictions['poly3_c005'][year]

    results = []
    for name, yearly in predictions.items():
        dev_y = np.r_[labels[2022], labels[2023]]
        dev_p = np.r_[yearly[2022], yearly[2023]]
        choices = (summarize(dev_y, dev_p, step / 100) for step in range(40, 91))
        threshold = max(choices, key=lambda row: (row['balanced_accuracy_pct'], row['apply_precision_pct'] or 0))['threshold']
        entry = {'candidate': name, 'threshold': threshold}
        for key, years in [('dev', [2022, 2023]), ('confirm2024', [2024]), ('later', [2025, 2026])]:
            y = np.concatenate([labels[year] for year in years])
            p = np.concatenate([yearly[year] for year in years])
            m = summarize(y, p, threshold)
            entry[key] = {
                'n': len(y), 'auc': round(roc_auc_score(y, p), 4),
                'brier': round(brier_score_loss(y, p), 4),
                'log_loss': round(log_loss(y, p), 4),
                'accuracy': m['classifier_accuracy_pct'],
                'balanced_accuracy': m['balanced_accuracy_pct'],
                'negative_recall': m['negative_recall_pct'],
                'warning_precision': m['warning_precision_pct'],
                'apply_precision': m['apply_precision_pct'],
                'warnings': m['tn'] + m['fn'], 'negative_caught': m['tn'],
            }
        results.append(entry)
    return sorted(results, key=lambda item: (-item['dev']['auc'], item['dev']['brier']))


def cohort_audit():
    """Check whether non-corporate trusts and their missing retail books matter."""
    frame = prepare_dataset()
    for label, exclude_trusts in [('all_254', False), ('corporate_252', True)]:
        cohort = frame[frame.year.between(2021, 2024)]
        if exclude_trusts:
            cohort = cohort[~cohort.company_name.str.contains('Trust', case=False, na=False)]
        folds = {}
        for year in YEARS:
            train = cohort[cohort.year <= min(year - 1, 2024)]
            test = frame[frame.year == year]
            folds[year] = (test.target_binary.to_numpy(), score_model(*fit_model(train), test))
        dev_y = np.r_[folds[2022][0], folds[2023][0]]
        dev_p = np.r_[folds[2022][1], folds[2023][1]]
        threshold = max((summarize(dev_y, dev_p, i / 100) for i in range(40, 91)),
                        key=lambda row: (row['balanced_accuracy_pct'], row['apply_precision_pct'] or 0))['threshold']
        output = {'cohort': label, 'training_samples': len(cohort), 'threshold': threshold}
        for name, years in [('development', (2022, 2023)), ('confirmation_2024', (2024,)),
                            ('later_2025_2026', (2025, 2026))]:
            y = np.concatenate([folds[year][0] for year in years])
            p = np.concatenate([folds[year][1] for year in years])
            summary = summarize(y, p, threshold)
            output[name] = {key: summary[key] for key in (
                'roc_auc', 'brier_score', 'classifier_accuracy_pct',
                'balanced_accuracy_pct', 'negative_recall_pct', 'apply_precision_pct',
                'warning_precision_pct', 'tn', 'fp', 'fn', 'tp',
            )}
        print(json.dumps(output))


def calibration_audit():
    """Fit calibrators only to 2022–23 out-of-year model scores."""
    frame = prepare_dataset()
    cohort = frame[frame.year.between(2021, 2024) & ~frame.id.isin([293, 307])]
    folds = {}
    for year in YEARS:
        train = cohort[cohort.year <= min(year - 1, 2024)]
        test = frame[frame.year == year]
        folds[year] = (test.target_binary.to_numpy(), score_model(*fit_model(train), test))
    y_dev = np.r_[folds[2022][0], folds[2023][0]]
    p_dev = np.r_[folds[2022][1], folds[2023][1]]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6)).reshape(-1, 1)
    sigmoid = LogisticRegression(C=1, max_iter=1000).fit(logit(p_dev), y_dev)
    isotonic = IsotonicRegression(out_of_bounds='clip').fit(p_dev, y_dev)
    transforms = {
        'uncalibrated': lambda p: p,
        'sigmoid_dev': lambda p: sigmoid.predict_proba(logit(p))[:, 1],
        'isotonic_dev': lambda p: isotonic.predict(p),
    }
    for name, transform in transforms.items():
        output = {'calibration': name}
        for label, years in [('confirmation_2024', (2024,)), ('later_2025_2026', (2025, 2026))]:
            y = np.concatenate([folds[year][0] for year in years])
            p = transform(np.concatenate([folds[year][1] for year in years]))
            output[label] = {'auc': round(roc_auc_score(y, p), 4),
                             'brier': round(brier_score_loss(y, p), 4),
                             'log_loss': round(log_loss(y, np.clip(p, 1e-6, 1-1e-6)), 4)}
        print(json.dumps(output))


if __name__ == '__main__':
    for result in evaluate():
        print(json.dumps(result))
    cohort_audit()
    calibration_audit()
