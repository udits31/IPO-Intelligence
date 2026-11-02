# Module 1 downside investigation: fixed 2025–2026 test IPOs

Trained on 170 eligible 2021–2024 IPOs. Evaluated on exactly the same 155 later-year IPOs.
Selected on earlier-year chronological validation: {'family': 'logistic', 'transform': 'raw', 'C': 0.05, 'class_weight': None, 'name': 'candidate_00'}; decision cutoff 0.575.

| Approach | Accuracy | Flat/negative caught | Positive caught | AUC | Brier |
|---|---:|---:|---:|---:|---:|
| previous | 74.8% | 13/49 | 103/106 | 0.8057 | 0.169 |
| gmp_sign | 68.4% | 3/49 | 103/106 | 0.5165 | 0.3161 |
| gmp_fitted | 68.4% | 0/49 | 106/106 | 0.8004 | 0.1795 |
| selected_fix | 72.3% | 7/49 | 105/106 | 0.808 | 0.18 |
| selected_fix_at_point5 | 68.4% | 0/49 | 106/106 | 0.808 | 0.18 |

## Why the previous calls failed

The training positive rate was 77.6%, compared with 68.4% in this test. The previous logistic intercept was 1.793; positive-demand inputs often outweighed weak GMP. This is an observed model behaviour, not proof of a causal market mechanism.
The training-selected fix corrects 0 of the original 36 false positives. It newly misses 0 previously correct positive openings. Net accuracy change: {'change_pp': -2.5806451612903225, 'bootstrap_95_pp': [-6.451612903225806, 0.6451612903225806], 'corrected_errors': 2, 'introduced_errors': 6}.
Each of the 36 wrong positive calls has a row containing its inputs, actual opening return, old feature contributions, new probability and new call in the companion CSV.

## Independent source check

{
  "status_counts": {
    "corroborated": 134,
    "no_matching_subscription_article": 12,
    "conflicting_sources": 9
  },
  "conflicting_issuer_ids": [
    510,
    532,
    545,
    584,
    591,
    624,
    625,
    638,
    686
  ],
  "fixed_model": {
    "sample_size": 155,
    "positive_listings": 106,
    "nonpositive_listings": 49,
    "threshold": 0.575,
    "roc_auc": 0.808,
    "brier_score": 0.1801,
    "classifier_accuracy_pct": 72.9,
    "always_positive_baseline_pct": 68.4,
    "balanced_accuracy_pct": 57.7,
    "negative_recall_pct": 16.3,
    "positive_recall_pct": 99.1,
    "apply_precision_pct": 71.9,
    "warning_precision_pct": 88.9,
    "tn": 8,
    "fp": 41,
    "fn": 1,
    "tp": 105
  },
  "previous_model": {
    "sample_size": 155,
    "positive_listings": 106,
    "nonpositive_listings": 49,
    "threshold": 0.5,
    "roc_auc": 0.799,
    "brier_score": 0.1709,
    "classifier_accuracy_pct": 74.8,
    "always_positive_baseline_pct": 68.4,
    "balanced_accuracy_pct": 61.9,
    "negative_recall_pct": 26.5,
    "positive_recall_pct": 97.2,
    "apply_precision_pct": 74.1,
    "warning_precision_pct": 81.2,
    "tn": 13,
    "fp": 36,
    "fn": 3,
    "tp": 103
  },
  "interpretation": "IPOWatch preferred consistently where verified; unresolved provider conflicts remain uncertainty, not proven corrections."
}

## Limits

- This is retrospective development after later-year errors were inspected; independent prospective confirmation is still required.
- No later-year labels enter fit, model selection or cutoff selection; the selected model need not be the best later-year diagnostic.
- Changing cutoff changes decisions, not underlying probability calibration. Report probability metrics separately.
- Same 155 test issuers retained, including all losses and flat opens. No live promotion, paper or Module 2 changes.

All 24 candidate diagnostics and the complete earlier-year selection trace are retained in the JSON. Later-year results do not choose the winner.
