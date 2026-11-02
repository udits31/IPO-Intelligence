# Module 1 — Pre-close return forecasts and IPO priority test

## What was built

A separate return regressor and early-book probability classifier, trained only on 2021–2024 IPOs. Inputs are penultimate-session QIB/HNI/retail, issue price and size. The paired variant learns a correction to a dated GMP quote. The existing final-book classifier is unchanged.

Candidates are fixed ridge, random forest and Huber boosting. Selection uses earlier-year 2022/2023 predictions; 2024 is retrospective confirmation, not a blind or prospective holdout. The 80% band uses earlier-year out-of-fold residuals; observed coverage is reported, not assumed.

## Results

| Track | 2024 IPOs | Return MAE (percentage points) | GMP-only MAE on same cases | 80% band coverage | Selection panels | Model minus GMP |
|---|---:|---:|---:|---:|---:|---:|
| bids_only (boosted_huber) | 75 | 26.85 | See paired track | 54.7% | 4 | +0.00 pp |
| bids_plus_gmp (boosted_huber) | 39 | 24.26 | 24.96 | 53.8% | 4 | +0.00 pp |

## Does it beat GMP-only selection?

A reliable advantage over GMP-only selection has NOT been established. Use the new forecasts as experimental research, not a proven superior allocation strategy.

Selection compares only IPOs with the same snapshot date and a valid preceding-date GMP quote. Same-day GMP is excluded because its update clock is unknown. Both selectors use the same feature-supported IPOs; exclusions remain in the prediction ledger. Risk score = central return minus 0.5 × downside-to-zero at the lower band minus a 0.5 percentage-point cost scenario. The top positive score is selected, otherwise skip all. GMP picks its highest positive quote after the same cost hurdle. No parameters were tuned on 2024 selection profits.

These are conditional-on-allotment opening returns, NOT portfolio profits. The 0.5% cost is a sensitivity assumption, not an exchange fee schedule. No allotment probabilities, capital reuse or guaranteed execution are invented.

Paired return errors: model 24.26 pp versus GMP 24.96 pp on 39 IPOs; model improvement +0.70 pp, bootstrap 95% interval -4.04 to +5.33 pp.

## Every paired choice

| Date | IPOs | Model selection | GMP selection | Model return after scenario | GMP return after scenario |
|---|---:|---|---|---:|---:|
| 2024-02-08 | 3 | Rashi Peripherals Ltd. | Rashi Peripherals Ltd. | +7.22% | +7.22% |
| 2024-09-10 | 2 | Bajaj Housing Finance Ltd. | Bajaj Housing Finance Ltd. | +113.79% | +113.79% |
| 2024-12-20 | 3 | Mamata Machinery Ltd. | Mamata Machinery Ltd. | +146.41% | +146.41% |
| 2024-12-23 | 2 | Senores Pharmaceuticals Ltd. | Senores Pharmaceuticals Ltd. | +51.34% | +51.34% |

## Limits that matter

- Daily subscription and GMP archive pages can have been edited later; not immutable investor-time evidence.
- GMP coverage is incomplete and selected. Quotes are at least a day older than the bid snapshot.
- Training is penultimate-session end-of-day, not a validated day-one or final-hour model.
- Live intraday or closing-day use is a stage extrapolation; prospective accuracy remains unmeasured.
- A residual band is empirical and can miss; the upper bound is not a cap or guarantee.
- Small simultaneous choice sets and repeated overlapping offer exposures limit inference.
- No paper numbers or final-book classifier were replaced by this experiment.

## Biggest return misses in the paired model

| IPO | Predicted central return | Actual opening return | Dated GMP | Model error |
|---|---:|---:|---:|---:|
| Unicommerce eSolutions Ltd. | +9.86% | +112.96% | +37.04% | 103.11 pp |
| Vibhor Steel Tubes Ltd. | +84.03% | +178.81% | +79.47% | 94.78 pp |
| Mamata Machinery Ltd. | +67.86% | +146.91% | +61.73% | 79.06 pp |
| JNK India Ltd. | -5.28% | +49.40% | +3.61% | 54.67 pp |
| P N Gadgil Jewellers Ltd. | +19.82% | +73.75% | +50.00% | 53.93 pp |
| Premier Energies Ltd. | +69.92% | +120.22% | +73.33% | 50.30 pp |
| Waaree Energies Ltd. | +116.07% | +69.66% | +100.47% | 46.41 pp |
| Orient Technologies Ltd. | -5.18% | +40.78% | +14.56% | 45.95 pp |

These are observed errors, not causal explanations. The model cannot infer demand shocks or opening-auction supply solely from earlier subscription multiples and an unofficial quote. The complete ledger retains every evaluated IPO, including these misses.
