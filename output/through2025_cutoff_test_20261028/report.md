# Training through 2025: matched 2026 comparison

290 training IPOs from 2021–2025. Cutoff selected on out-of-time 2024 predictions, confirmed on 2025, then fixed for 2026. No model promotion.

| Variant | Cutoff | All 69 in 2026 | Same 64 stage-verified IPOs | Flat/negative caught (64) | Positive caught (64) |
|---|---:|---:|---:|---:|---:|
| baseline_fixed50 | 0.500 | 56/69 (81.2%) | 54/64 (84.4%) | 16/19 | 38/45 |
| baseline_prior_fixed625 | 0.625 | 49/69 (71.0%) | 47/64 (73.4%) | 19/19 | 28/45 |
| baseline_validated_cutoff | 0.850 | 36/69 (52.2%) | 32/64 (50.0%) | 19/19 | 13/45 |
| financial_join_fixed50 | 0.500 | 55/69 (79.7%) | 53/64 (82.8%) | 15/19 | 38/45 |
| financial_join_prior_fixed625 | 0.625 | 49/69 (71.0%) | 47/64 (73.4%) | 19/19 | 28/45 |
| financial_join_validated_cutoff | 0.825 | 38/69 (55.1%) | 34/64 (53.1%) | 19/19 | 15/45 |

The all-69 sensitivity includes five records without verified Day-2 stage alignment; the same-64 comparison excludes those identical records from every variant.
These are previously examined historical cohorts. The updated through-2025 models are causal research refits and are separate from the deployed artifact.

## Pre-2026 confirmation

baseline_validated_cutoff: 2025 confirmation accuracy 52.9%, balanced accuracy 63.1%, cutoff 0.850.
financial_join_validated_cutoff: 2025 confirmation accuracy 60.6%, balanced accuracy 67.1%, cutoff 0.825.
