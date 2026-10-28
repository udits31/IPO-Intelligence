# OpenBell — matched model comparison

Training: 2021–2024 only. Test labels: 2025–2026. Research replay; live artifacts and saved calls unchanged.

Original training: 252 IPOs. Matched early-book training: 186. Original holdout: 189; richer-input overlap: 173; verified Day-2 subset: 163.

| Cohort | Period | Original model /252 | Same-book logistic /186 | Five-expert /186 | GMP-only | Always positive |
|---|---|---:|---:|---:|---:|---:|
| clean_day2_163 | combined | 91/163 (55.8%) | 115/163 (70.6%) | 115/163 (70.6%) | 113/163 (69.3%) | 113/163 (69.3%) |
| clean_day2_163 | 2025 | 55/99 (55.6%) | 68/99 (68.7%) | 70/99 (70.7%) | 68/99 (68.7%) | 68/99 (68.7%) |
| clean_day2_163 | 2026 | 36/64 (56.2%) | 47/64 (73.4%) | 45/64 (70.3%) | 45/64 (70.3%) | 45/64 (70.3%) |
| all_richer_inputs_173 | combined | 97/173 (56.1%) | 119/173 (68.8%) | 119/173 (68.8%) | 117/173 (67.6%) | 117/173 (67.6%) |
| all_richer_inputs_173 | 2025 | 57/104 (54.8%) | 71/104 (68.3%) | 73/104 (70.2%) | 71/104 (68.3%) | 71/104 (68.3%) |
| all_richer_inputs_173 | 2026 | 40/69 (58.0%) | 48/69 (69.6%) | 46/69 (66.7%) | 46/69 (66.7%) | 46/69 (66.7%) |
| previous_fixed155 | combined | 87/155 (56.1%) | 108/155 (69.7%) | 108/155 (69.7%) | 106/155 (68.4%) | 106/155 (68.4%) |
| previous_fixed155 | 2025 | 47/86 (54.7%) | 60/86 (69.8%) | 62/86 (72.1%) | 60/86 (69.8%) | 60/86 (69.8%) |
| previous_fixed155 | 2026 | 40/69 (58.0%) | 48/69 (69.6%) | 46/69 (66.7%) | 46/69 (66.7%) | 46/69 (66.7%) |

Paired five-expert minus original-model accuracy difference: {"candidate_only_correct": 69, "baseline_only_correct": 45, "accuracy_difference_pp": 14.723926380368098, "difference_bootstrap_95_pp": [2.4539877300613497, 26.380368098159508], "exact_mcnemar_p": 0.030776323854403202}

## Return estimation and weekly ranking

| Return companion | MAE (percentage points) | Spearman | Mean weekly selected listing return |
|---|---:|---:|---:|
| r_legacy_three_input_ridge | 17.31 | 0.657 | 23.83% |
| r_five_view_ridge | 10.01 | 0.734 | 26.99% |
| r_constant_training_mean | 21.02 | Undefined (constant) | 12.71% |

## Actual deployed model: 2026-only comparison

This separate comparison permits training through 2025. It never scores 2025 as unseen.

| Cohort | Deployed five-expert /290 | Same-stage logistic /290 | Original logistic /252 on the same early books |
|---|---:|---:|---:|
| all69 | 56/69 (81.2%) | 52/69 (75.4%) | 40/69 (58.0%) |
| verified_day2_64 | 54/64 (84.4%) | 51/64 (79.7%) | 36/64 (56.2%) |

GMP return comparisons use only the identical rows with a real dated quote; see report.json for denominators, annual results and weekly selections.
The full-cohort GMP-only classifier uses its training median for missing quotes. The report separately provides dated-GMP direction comparisons excluding missing quotes for every model.

## Limits

Architecture and these years have already been inspected during development; this is a leakage-checked retrospective comparison, not a fresh independent holdout.
The original model was trained on final subscriptions. Its early-book replay is a stage-shift diagnostic. The matched logistic control isolates model architecture on the same training issuers/books.
Ten interim-stage rows are outside the primary verified-Day-2 comparison; sixteen original holdout IPOs lack matched early-book features. The exclusions are enumerated in coverage_all189.csv.
Three exchange opening values differ from the original dataset; matched comparisons consistently use the recovered exchange openings, with differences recorded in report.json.
Weekly rankings do not account for allotment, available application capital or execution costs. No profit or independent generalisation claim follows from this rerun.
