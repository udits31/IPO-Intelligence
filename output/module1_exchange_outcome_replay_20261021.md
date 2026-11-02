# Module 1 — Pre-close return forecasts and IPO priority test

## What was built

A separate return regressor and early-book probability classifier, trained only on 2021–2024 IPOs. Inputs are penultimate-session QIB/HNI/retail, issue price and size. The paired variant learns a correction to a dated GMP quote. The existing final-book classifier is unchanged.

Candidates are fixed ridge, random forest and Huber boosting. Selection uses earlier-year 2022/2023 predictions; 2024 is retrospective confirmation, not a blind or prospective holdout. The 80% band uses earlier-year out-of-fold residuals; observed coverage is reported, not assumed.

## Results

| Track | 2024 IPOs | Return MAE (percentage points) | GMP-only MAE on same cases | 80% band coverage | Selection panels | Model minus GMP |
|---|---:|---:|---:|---:|---:|---:|
| bids_only (boosted_huber) | 75 | 27.68 | See paired track | 53.3% | 4 | +0.00 pp |
| bids_plus_gmp (boosted_huber) | 39 | 25.08 | 25.92 | 53.8% | 4 | +0.00 pp |

## Does it beat GMP-only selection?

A reliable advantage over GMP-only selection has NOT been established.

Selection compares only IPOs with the same snapshot date and a valid preceding-date GMP quote. Same-day GMP is excluded because its update clock is unknown. Both selectors use the same feature-supported IPOs; exclusions remain in the prediction ledger. Risk score = central return minus 0.5 × downside-to-zero at the lower band minus a 0.5 percentage-point cost scenario. The top positive score is selected, otherwise skip all. GMP picks its highest positive quote after the same cost hurdle. No parameters were tuned on 2024 selection profits.

These are conditional-on-allotment opening returns, NOT portfolio profits. The 0.5% cost is a sensitivity assumption, not an exchange fee schedule. No allotment probabilities, capital reuse or guaranteed execution are invented.

Paired return errors: model 25.08 pp versus GMP 25.92 pp on 39 IPOs; model improvement +0.84 pp, bootstrap 95% interval -3.97 to +5.49 pp.

## Every paired choice

| Date | IPOs | Model selection | GMP selection | Model return after scenario | GMP return after scenario |
|---|---:|---|---|---:|---:|
| 2024-02-08 | 3 | Rashi Peripherals Ltd. | Rashi Peripherals Ltd. | +8.66% | +8.66% |
| 2024-09-10 | 2 | Bajaj Housing Finance Ltd. | Bajaj Housing Finance Ltd. | +113.79% | +113.79% |
| 2024-12-20 | 3 | Mamata Machinery Ltd. | Mamata Machinery Ltd. | +146.41% | +146.41% |
| 2024-12-23 | 2 | Senores Pharmaceuticals Ltd. | Senores Pharmaceuticals Ltd. | +52.95% | +52.95% |

## Limits that matter

- NSE opening is used when listed on NSE at the IPO date; exact-code BSE opening otherwise.
- Canonical prices, labels and live model artifacts have not been silently rewritten.
- Offer prices and historical bid publication times still require independent point-in-time offer-document verification.
- Daily GMP quotes must precede the bid date because their update time is unknown.
- The final-book direction replay is not evidence of pre-application-cutoff accuracy.
- The exact original cohort includes the Ruchi Soya follow-on offer; retained for cohort parity, not relabelled as a new IPO.
- 2024 and later outcomes have been inspected previously; not a blind test.
- Conditional-on-allotment opening gains are not wallet returns or guaranteed executable profits.

## Biggest return misses in the paired model

| IPO | Predicted central return | Actual opening return | Dated GMP | Model error |
|---|---:|---:|---:|---:|
| Unicommerce eSolutions Ltd. | +8.78% | +117.59% | +37.04% | 108.81 pp |
| Vibhor Steel Tubes Ltd. | +82.34% | +181.46% | +79.47% | 99.12 pp |
| Mamata Machinery Ltd. | +65.32% | +146.91% | +61.73% | 81.59 pp |
| JNK India Ltd. | -5.93% | +49.64% | +3.61% | 55.56 pp |
| P N Gadgil Jewellers Ltd. | +21.41% | +72.92% | +50.00% | 51.51 pp |
| Waaree Energies Ltd. | +116.81% | +66.33% | +100.47% | 50.47 pp |
| Premier Energies Ltd. | +70.74% | +120.00% | +73.33% | 49.26 pp |
| Orient Technologies Ltd. | -4.97% | +39.81% | +14.56% | 44.77 pp |

These are observed errors, not causal explanations. The model cannot infer demand shocks or opening-auction supply solely from earlier subscription multiples and an unofficial quote. The complete ledger retains every evaluated IPO, including these misses.


## Verified-outcome direction sensitivity

This is the final-book classifier only; it does not validate application-time prediction.

| Years | Verified cases | Original LR vs verified outcomes | Same LR refit on verified outcomes |
|---|---:|---:|---:|
| 2023–2024 | 149 | 82.6% | 81.9% |
| 2025–2026 | 179 | 76.0% | 74.9% |

Coverage: {'training_n': 252, 'later_requested_n': 189, 'later_verified_n': 179, 'unverified_ids': [527, 565, 574, 605, 609, 649, 666, 668, 669, 676]}. Unverified IPOs are not counted as successful predictions.

## Original frozen return predictions, scored without retraining

On 39 paired IPOs, original forecasts have 25.17 pp return error versus 25.92 pp for GMP. Average improvement +0.75 pp; 95% descriptive interval -3.96 to +5.34 pp. These predictions were not refitted after observing corrected 2024 outcomes. This audit changes the scoring source, not the saved forecasts or deployed model.
