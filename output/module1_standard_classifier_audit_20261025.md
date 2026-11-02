# Module 1: standard-classifier audit and the 20 missed gains

**2026 candidate: 56/69 correct (81.2%). Strict 2021–2024 training on all 155: 118/155 (76.1%).**

| Evaluation | Training | Test | Correct | Overall accuracy |
|---|---|---:|---:|---:|
| Previous strict blend | 2021–2024 | Same 155 IPOs | 118/155 | 76.1% |
| Previous blend, 2026 slice | 2021–2024 | Same 69 IPOs | 55/69 | 79.7% |
| Updated GMP-only, earlier-selected cutoff | 2021–2025 | Same 69 IPOs | 55/69 | 79.7% |
| Updated five-view stack, custom 57.5% cutoff | 2021–2025 | Same 69 IPOs | 52/69 | 75.4% |
| Updated five-view stack, standard predict() | 2021–2025 | Same 69 IPOs | 56/69 | 81.2% |

## What was changed

- Recovered 18 previously missing 2025 interim books, including brand-name and malformed-number handling. Numeric conflicts were not guessed.
- Included 16 earlier IPOs whose usable books had been excluded for missing GMP. Missing GMP remains missing and is handled inside fitting.
- Training now has 290 earlier IPOs: 186 from 2021–2024 and 104 from 2025. None of the 69 2026 IPOs enters either expert fitting or meta calibration.
- Combined fixed views of bids/GMP, market conditions, financial facts and demand trajectories. The combiner was trained on earlier out-of-fold predictions.
- The standard classifier uses its normal 50% class rule. The custom 57.5% policy was too conservative in this test. Both results are retained.

## Error counts

- Correct positive openings: 38/46; missed positive openings: 8.
- Correct flat/negative openings: 18/23; wrong positive calls: 5.
- Of the original 20 missed gains, 12 are 2025 training/calibration cases and eight are eligible 2026 tests. The updated model recovers 3 of those eight. All 20 remain in the review CSV.
- AUC: 0.879; Brier: 0.1516; 95% Wilson accuracy interval: [70.38706411837158, 88.64525167564403].

## Interpretation

This crosses 80% for the 2026 candidate, with all 69 IPOs counted. It does not establish 80–85% for the original strict 155-row test or for future live forecasts. The standard-rule comparison was added after inspecting the failed custom-cutoff result; this is transparent exploratory evidence, not a fresh untouched holdout.
The gain over the previous 2026 blend is one correct IPO. Paired comparison: {'change_pp': 1.4492753623188406, 'bootstrap_95_pp': [-7.246376811594203, 10.144927536231885], 'corrected_errors': 5, 'introduced_errors': 4}.

## Files

- All 155 IPOs, strict scores and updated eligibility: `module1_standard_classifier_audit_20261025_all_155_status.csv`.
- Original 20 missed gains: `module1_standard_classifier_audit_20261025_20_missed_positive_review.csv`.
- All 69 updated 2026 predictions and actual opens: `module1_standard_classifier_audit_20261025_2026_predictions.csv`.
- All 13 updated wrong calls: `module1_standard_classifier_audit_20261025_2026_errors.csv`.
- Reproducible candidate artifact: `ml\artifacts\module1_2026_standard_classifier_research_20261025.joblib`.

## Limits

- The standard 50% classification comparison was added after the custom-policy test was inspected. No numeric cutoff search on 2026 was performed, but this is post-hoc exploratory evidence.
- 81.2% applies to the 69 2026 IPOs after training through 2025, not to the full 155 IPOs.
- The 2025 rows are training/calibration, never scored as holdout for this updated model.
- The improvement over the previous 2026 blend is one IPO and its paired interval includes zero.
- Historical inputs are reconstructed; original pre-close forecasts and publication times are not certified.
- The live bundle, paper and Module 2 remain unchanged. A prospectively frozen test is needed before claiming sustained accuracy above 80%.
