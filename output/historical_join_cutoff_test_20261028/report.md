# Historical financial joins and decision cutoff test

Training: 2021–2024. Cutoff selection: 2023 out-of-time scores. Confirmation: 2024. Test: 2025–2026. No live model promotion.

| Variant | Cutoff | Combined clean Day-2 correct | 2025 | 2026 | Flat/negative caught | Positive caught |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fixed50 | 0.500 | 115/163 (70.6%) | 70.7% | 70.3% | 2/50 | 113/113 |
| baseline_validated_cutoff | 0.625 | 127/163 (77.9%) | 72.7% | 85.9% | 26/50 | 101/113 |
| financial_join_fixed50 | 0.500 | 116/163 (71.2%) | 70.7% | 71.9% | 3/50 | 113/113 |
| financial_join_validated_cutoff | 0.675 | 122/163 (74.8%) | 71.7% | 79.7% | 31/50 | 91/113 |

## Coverage

{
  "baseline": {
    "pub_ipo_earnings_yield_pct": {
      "training_observed": 72,
      "training_n": 186,
      "test_observed": 172,
      "test_n": 173
    },
    "pub_log_ipo_price_book": {
      "training_observed": 66,
      "training_n": 186,
      "test_observed": 152,
      "test_n": 173
    }
  },
  "financial_join": {
    "pub_ipo_earnings_yield_pct": {
      "training_observed": 116,
      "training_n": 186,
      "test_observed": 172,
      "test_n": 173
    },
    "pub_log_ipo_price_book": {
      "training_observed": 96,
      "training_n": 186,
      "test_observed": 152,
      "test_n": 173
    }
  }
}

## Exclusions

{
  "connected_rows": 48,
  "excluded_rows": 92,
  "reasons": {
    "document_not_demonstrably_before_decision": 65,
    "financial_period_not_prior": 4,
    "identity_or_status": 23
  }
}

Previously inspected retrospective test cohorts; chronological fitting does not turn them into fresh holdouts.
Document date is retrospective availability evidence, not independently certified publication time. A later fetched file is not rejected merely because it was downloaded in 2026; a document dated after the decision is excluded.
Financial features are missing where the source does not establish a unique period/basis. No test labels determine joins, models or thresholds.
