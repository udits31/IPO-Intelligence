# Module 1: financial extension and downside errors

Training: 170 IPOs, 2021–2024 only. Test: the same 155 IPOs, 2025–2026. No overlap.
Selected using only earlier-year chronological validation: {'family': 'gradient_boosting', 'transform': 'raw', 'max_depth': 2, 'n_estimators': 160, 'name': 'valuation_simple_candidate_15', 'feature_set': 'valuation_simple'}. Cutoff: 0.75.

| Model | Correct / 155 | Accuracy | Flat/negative caught | Positive caught | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| p_combined | 116/155 | 74.8% | 13/49 | 103/106 | 0.8057 | 0.169 |
| p_gmp_sign | 106/155 | 68.4% | 3/49 | 103/106 | 0.5165 | 0.3161 |
| p_gmp_fitted | 106/155 | 68.4% | 0/49 | 106/106 | 0.8004 | 0.1795 |
| financial_extension | 104/155 | 67.1% | 24/49 | 80/106 | 0.6742 | 0.211 |

Corrected 13 of the original 36 false positives; newly missed 24 previously correct positive openings.

Paired comparisons: {'p_combined': {'change_pp': -7.741935483870968, 'bootstrap_95_pp': [-15.483870967741936, 0.0], 'corrected_errors': 14, 'introduced_errors': 26}, 'p_gmp_sign': {'change_pp': -1.2903225806451613, 'bootstrap_95_pp': [-9.67741935483871, 7.741935483870968], 'corrected_errors': 23, 'introduced_errors': 25}, 'p_gmp_fitted': {'change_pp': -1.2903225806451613, 'bootstrap_95_pp': [-10.32258064516129, 7.741935483870968], 'corrected_errors': 24, 'introduced_errors': 26}}

Yearly results: {'2025': {'sample_size': 86, 'positive_listings': 60, 'nonpositive_listings': 26, 'threshold': 0.75, 'roc_auc': 0.6551, 'brier_score': 0.2098, 'classifier_accuracy_pct': 66.3, 'always_positive_baseline_pct': 69.8, 'balanced_accuracy_pct': 59.5, 'negative_recall_pct': 42.3, 'positive_recall_pct': 76.7, 'apply_precision_pct': 75.4, 'warning_precision_pct': 44.0, 'tn': 11, 'fp': 15, 'fn': 14, 'tp': 46}, '2026': {'sample_size': 69, 'positive_listings': 46, 'nonpositive_listings': 23, 'threshold': 0.75, 'roc_auc': 0.6919, 'brier_score': 0.2125, 'classifier_accuracy_pct': 68.1, 'always_positive_baseline_pct': 66.7, 'balanced_accuracy_pct': 65.2, 'negative_recall_pct': 56.5, 'positive_recall_pct': 73.9, 'apply_precision_pct': 77.3, 'warning_precision_pct': 52.0, 'tn': 13, 'fp': 10, 'fn': 12, 'tp': 34}}

Financial coverage: {'train': {'n': 170, 'annual_tables': 163, 'feature_observed': {'profit_margin_pct': 163, 'return_on_assets_pct': 163, 'debt_to_networth': 62, 'revenue_growth_asinh': 158, 'profit_growth_asinh': 159, 'annual_loss': 163, 'ipo_earnings_yield_pct': 61, 'log_ipo_price_book': 60, 'financial_sector': 170}}, 'test': {'n': 155, 'annual_tables': 154, 'feature_observed': {'profit_margin_pct': 154, 'return_on_assets_pct': 154, 'debt_to_networth': 138, 'revenue_growth_asinh': 154, 'profit_growth_asinh': 154, 'annual_loss': 154, 'ipo_earnings_yield_pct': 138, 'log_ipo_price_book': 132, 'financial_sector': 155}}}

Source-cross-checked sensitivity: {'sample_size': 155, 'positive_listings': 106, 'nonpositive_listings': 49, 'threshold': 0.75, 'roc_auc': 0.6844, 'brier_score': 0.2083, 'classifier_accuracy_pct': 67.7, 'always_positive_baseline_pct': 68.4, 'balanced_accuracy_pct': 62.7, 'negative_recall_pct': 49.0, 'positive_recall_pct': 76.4, 'apply_precision_pct': 76.4, 'warning_precision_pct': 49.0, 'tn': 24, 'fp': 25, 'fn': 25, 'tp': 81}

## Limits

- Later-year errors motivated this extension, so the test is reused exploratory evidence, not independent confirmation.
- All model, preprocessing and cutoff fitting/selection uses 2021–2024 only.
- Financial periods precede decisions, but public historical pages do not certify original fact publication times.
- Primary test bids and all 155 IPOs are unchanged; source substitutions are sensitivity only.
- A trained cutoff changes the call policy, not probability calibration. No live or saved forecast was overwritten.

Every original and revised call, actual opening and correctness flag: `module1_financial_fix_20261024_predictions.csv`.
