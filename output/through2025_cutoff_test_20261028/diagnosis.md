# Why the earlier cutoff did not transfer after retraining

This is a read-only diagnosis of existing research artifacts on exactly the same 64 stage-verified 2026 IPOs. No new model was fitted, cutoff selected, or live artifact changed during this diagnosis.

## What changed

The earlier 70.6% versus 77.9% comparison used one through-2024 fitted five-expert model. Its positive-call threshold changed from 0.50 to 0.625; this was not a new model architecture. Both were measured on the same 163 test IPOs from 2025–2026.

Adding 2025 refits both the five experts and their logistic combiner. A threshold selected for the earlier fit does not necessarily transfer to the refit.

| Training | Decision threshold | Correct on the same 64 in 2026 | Winners caught /45 | Flat/negative caught /19 |
|---|---:|---:|---:|---:|
| 2021–2024 | 0.50 | 45/64 (70.3%) | 45 | 0 |
| 2021–2024 | 0.625 | 55/64 (85.9%) | 43 | 12 |
| 2021–2025 | 0.50 | 54/64 (84.4%) | 38 | 16 |
| 2021–2025 | 0.625 | 47/64 (73.4%) | 28 | 19 |

For the refit, 13 IPOs scored between 0.50 and 0.625: ten actually opened positive and three opened flat/negative. Raising the threshold sacrificed ten correct positives to avoid three incorrect positives: a net seven fewer correct calls.

## Scores improved even though a transferred decision rule failed

On the same 64 IPOs, ROC AUC increased from 0.9006 to 0.9216. Brier score decreased from 0.1503 to 0.1354 (lower is better). These are retrospective diagnostics, not proof of calibrated probabilities or independent generalisation.

The median score for actual flat/negative IPOs fell from 57.6% to 43.5%. The median score for winners stayed approximately 76.5%, but the winning lower quartile fell from 71.2% to 56.5%. Thus, the old 62.5% threshold also excluded more winners after retraining.

## Historical validation was unstable

In expanding-window validation, the meta learner had only 28 prior out-of-fold IPOs to fit before predicting 2023. Its 2024 validation winner/nonwinner score medians were 95.1%/82.0%, compared with 80.5%/65.3% for 2025 validation. This evidence shows an unstable score scale across refits and periods; it does not identify a single causal mechanism.

The threshold selected on 2024 validation was 0.85. It achieved only 52.9% accuracy on the separate 2025 confirmation set and was rejected. Its subsequent 2026 result is a failed research diagnostic, not a viable release candidate.

2025 also differed from earlier training: 71/104 positive outcomes (68.3%), versus 145/186 (78.0%) through 2024. Financial coverage was uneven: baseline earnings yield was observed for 72/186 earlier training IPOs versus 103/104 in 2025. Both sample composition and missingness changed along with model parameters. More observations alone do not establish higher test accuracy.

## Current conclusion

The through-2025 refit at 0.50 achieves 56/69 (81.2%) across the full 2026 cohort, and 54/64 (84.4%) on the stage-verified subset. The old through-2024 fit at 0.625 achieved 55/64 on that same subset: only one extra correct result. The 77.9% figure used a different, larger 2025–2026 test cohort and must not be directly compared with a 2026-only result as if the test population were unchanged.

Keep model architecture, score quality, cutoff policy, and test cohort separate in all reporting. A revised cutoff must pass chronological pre-test confirmation and be frozen before future outcomes; do not search 2026 labels for a higher headline. Live weights and prediction ledgers remain unchanged.

Per-IPO paired cases: `cutoff_transfer_cases.csv`. Machine-readable diagnostics: `cutoff_transfer_diagnosis.json`.
