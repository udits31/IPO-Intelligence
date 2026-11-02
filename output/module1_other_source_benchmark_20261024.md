# Module 1: other-source recovery, temporal inputs and error diagnosis

Usable GMP coverage increased from 123 to 170; 47 additional eligible histories, 16 unresolved. All 63 prior gaps were attempted. Source failures and restrictions are retained, not silently dropped.

Remaining gap reasons: {"conflicting_predecision_quotes": 1, "no_verified_numeric_history": 10, "only_same_day_or_later_quotes": 5}. A source with different numeric quotes on the same day is not arbitrarily preferred because it improves the score.

Matched early bid-growth histories: 128; same-publisher pre-decision GMP slopes: 165.

7 selected GMP inputs are explicitly published ranges. Both endpoints are retained; the labelled midpoint is a derived feature, not a claimed single observed quote. Lower/upper endpoint return-error scenarios are in the JSON.

Selected on 2022/2023 only: direction **bids_offer_gmp**, returns **r_temporal_bids_gmp_huber**. Fixed thresholds and model parameters. 2024 has been inspected before; this is exploratory research, not blind confirmation.

Separate ranking selector: **gmp_pct**. 2022/2023 pairwise ordering accuracy, minimum 5 panels and 10 pairs; ties prefer fewer inputs then name; insufficient evidence falls back to raw GMP. This corrects the mismatch between choosing a model for average price error and choosing one for ordering IPOs; it is not a claim that ranking has improved.

| Test set | IPOs | GMP-sign accuracy | Combined accuracy | GMP return error | Combined return error | Fitted GMP-only return error |
|---|---:|---:|---:|---:|---:|---:|
| original_39 | 39 | 87.2% | 82.1% | 25.92 pp | 25.55 pp | 24.63 pp |
| previous_51 | 51 | 80.4% | 78.4% | 23.96 pp | 23.07 pp | 21.94 pp |
| expanded | 68 | 79.4% | 79.4% | 21.15 pp | 20.01 pp | 19.25 pp |

Return error means average absolute error in percentage points; lower is better. The original denominators remain separate.

## Every fixed direction candidate

| Test set | Inputs | Accuracy | Balanced accuracy | AUC |
|---|---|---:|---:|---:|
| original_39 | gmp_fitted | 87.2% | 50.0% | 0.685 |
| original_39 | bids_offer_gmp | 82.1% | 47.1% | 0.771 |
| original_39 | full_facts_gmp | 76.9% | 61.2% | 0.747 |
| original_39 | sector_relative_gmp | 84.6% | 65.6% | 0.782 |
| original_39 | temporal_bids_gmp | 79.5% | 45.6% | 0.694 |
| original_39 | gmp_sign | 87.2% | 50.0% | 0.500 |
| previous_51 | gmp_fitted | 80.4% | 50.0% | 0.679 |
| previous_51 | bids_offer_gmp | 78.4% | 52.6% | 0.749 |
| previous_51 | full_facts_gmp | 76.5% | 62.7% | 0.702 |
| previous_51 | sector_relative_gmp | 80.4% | 61.3% | 0.712 |
| previous_51 | temporal_bids_gmp | 76.5% | 51.3% | 0.688 |
| previous_51 | gmp_sign | 80.4% | 50.0% | 0.500 |
| expanded | gmp_fitted | 79.4% | 50.0% | 0.722 |
| expanded | bids_offer_gmp | 79.4% | 57.9% | 0.767 |
| expanded | full_facts_gmp | 77.9% | 62.3% | 0.729 |
| expanded | sector_relative_gmp | 80.9% | 61.5% | 0.726 |
| expanded | temporal_bids_gmp | 80.9% | 61.5% | 0.726 |
| expanded | gmp_sign | 79.4% | 55.3% | 0.553 |

## Paired uncertainty and ranking

### original_39

Direction difference: -5.13 points (95% interval -12.82 to +0.00); 0 GMP mistakes corrected; 2 new mistakes introduced.

Return-error improvement versus raw GMP: +0.37 pp (95% interval -3.94 to +4.67). Versus fitted GMP-only: -0.92 pp (95% interval -5.27 to +3.34).

Ranking: 4 identical-date choice sets. Reliable edge established: False.

Dedicated ranking selection: gmp_pct; pairwise ordering 100.0% versus GMP 100.0%; selection difference 0.0 pp. Reliable dedicated-ranking edge established: False.

| Decision date | IPOs | Combined choice | GMP choice | Actual combined opening return | Actual GMP opening return |
|---|---:|---|---|---:|---:|
| 2024-02-08 | 3 | Rashi Peripherals Ltd. | Rashi Peripherals Ltd. | +9.16% | +9.16% |
| 2024-09-10 | 2 | Bajaj Housing Finance Ltd. | Bajaj Housing Finance Ltd. | +114.29% | +114.29% |
| 2024-12-20 | 3 | Mamata Machinery Ltd. | Mamata Machinery Ltd. | +146.91% | +146.91% |
| 2024-12-23 | 2 | Senores Pharmaceuticals Ltd. | Senores Pharmaceuticals Ltd. | +53.45% | +53.45% |

### previous_51

Direction difference: -1.96 points (95% interval -9.80 to +3.92); 1 GMP mistakes corrected; 2 new mistakes introduced.

Return-error improvement versus raw GMP: +0.88 pp (95% interval -2.59 to +4.44). Versus fitted GMP-only: -1.14 pp (95% interval -5.07 to +2.48).

Ranking: 7 identical-date choice sets. Reliable edge established: False.

Dedicated ranking selection: gmp_pct; pairwise ordering 100.0% versus GMP 100.0%; selection difference 0.0 pp. Reliable dedicated-ranking edge established: False.

| Decision date | IPOs | Combined choice | GMP choice | Actual combined opening return | Actual GMP opening return |
|---|---:|---|---|---:|---:|
| 2024-02-08 | 3 | Rashi Peripherals Ltd. | Rashi Peripherals Ltd. | +9.16% | +9.16% |
| 2024-09-10 | 2 | Bajaj Housing Finance Ltd. | Bajaj Housing Finance Ltd. | +114.29% | +114.29% |
| 2024-09-18 | 2 | Arkade Developers Ltd. | Arkade Developers Ltd. | +36.72% | +36.72% |
| 2024-10-22 | 2 | Waaree Energies Ltd. | Waaree Energies Ltd. | +66.33% | +66.33% |
| 2024-12-12 | 2 | Sai Life Sciences Ltd. | Vishal Mega Mart Ltd. | +18.40% | +33.33% |
| 2024-12-20 | 4 | Mamata Machinery Ltd. | Mamata Machinery Ltd. | +146.91% | +146.91% |
| 2024-12-23 | 2 | Senores Pharmaceuticals Ltd. | Senores Pharmaceuticals Ltd. | +53.45% | +53.45% |

### expanded

Direction difference: +0.00 points (95% interval -5.88 to +5.88); 2 GMP mistakes corrected; 2 new mistakes introduced.

Return-error improvement versus raw GMP: +1.14 pp (95% interval -1.67 to +3.75). Versus fitted GMP-only: -0.76 pp (95% interval -3.70 to +2.14).

Ranking: 7 identical-date choice sets. Reliable edge established: False.

Dedicated ranking selection: gmp_pct; pairwise ordering 91.66666666666667% versus GMP 91.66666666666667%; selection difference 0.0 pp. Reliable dedicated-ranking edge established: False.

| Decision date | IPOs | Combined choice | GMP choice | Actual combined opening return | Actual GMP opening return |
|---|---:|---|---|---:|---:|
| 2024-02-08 | 3 | Rashi Peripherals Ltd. | Rashi Peripherals Ltd. | +9.16% | +9.16% |
| 2024-09-10 | 3 | Bajaj Housing Finance Ltd. | Bajaj Housing Finance Ltd. | +114.29% | +114.29% |
| 2024-09-18 | 2 | Arkade Developers Ltd. | Arkade Developers Ltd. | +36.72% | +36.72% |
| 2024-10-22 | 2 | Waaree Energies Ltd. | Waaree Energies Ltd. | +66.33% | +66.33% |
| 2024-12-12 | 3 | One Mobikwik Systems Ltd. | One Mobikwik Systems Ltd. | +57.71% | +57.71% |
| 2024-12-20 | 5 | Mamata Machinery Ltd. | Mamata Machinery Ltd. | +146.91% | +146.91% |
| 2024-12-23 | 3 | Senores Pharmaceuticals Ltd. | Senores Pharmaceuticals Ltd. | +53.45% | +53.45% |

## Where the selected direction model is wrong

| IPO | Dated GMP | Predicted positive probability | Actual opening return | Missing earnings / OFS | Out-of-training-range inputs |
|---|---:|---:|---:|---|---|
| Jana Small Finance Bank Ltd. | 16.9% | 72.8% | -4.35% | False / False | None |
| Capital Small Finance Bank Ltd. | 10.7% | 77.7% | -8.07% | False / False | None |
| Juniper Hotels Ltd. | 2.8% | 48.7% | +1.39% | False / True | None |
| GPT Healthcare Ltd. | 0.0% | 45.0% | +15.59% | False / False | None |
| R K SWAMY Ltd. | 31.2% | 86.2% | -13.19% | False / False | None |
| J.G.Chemicals Ltd. | 27.1% | 84.6% | -5.43% | False / False | None |
| Gopal Snacks Ltd. | 17.5% | 64.6% | -12.47% | True / True | None |
| Ola Electric Mobility Ltd. | 21.7% | 76.4% | +0.00% | True / True | None |
| Baazar Style Retail Ltd. | 32.1% | 94.4% | +0.00% | False / True | None |
| Kross Ltd. | 20.8% | 64.7% | +0.00% | True / True | None |
| Western Carriers (India) Ltd. | 36.0% | 89.3% | -0.58% | True / True | None |
| Hyundai Motor India Ltd. | 2.3% | 87.2% | -1.33% | False / False | log_offer_cap_price; log_issue_size |
| Deepak Builders & Engineers India Ltd. | 29.6% | 93.1% | -1.48% | True / True | None |
| Sagility India Ltd. | 10.0% | 38.7% | +3.53% | True / True | None |

## Largest return misses

| IPO | Predicted return | Actual opening return | Dated GMP | Absolute error |
|---|---:|---:|---:|---:|
| Unicommerce eSolutions Ltd. | +8.15% | +117.59% | +37.04% | 109.44 pp |
| Vibhor Steel Tubes Ltd. | +82.35% | +181.46% | +79.47% | 99.11 pp |
| Mamata Machinery Ltd. | +85.69% | +146.91% | +61.73% | 61.22 pp |
| SRM Contractors Ltd. | +56.88% | +2.50% | +30.95% | 54.38 pp |
| Waaree Energies Ltd. | +118.23% | +66.33% | +100.47% | 51.90 pp |
| JNK India Ltd. | -1.35% | +49.64% | +3.61% | 50.99 pp |
| Premier Energies Ltd. | +69.17% | +120.00% | +73.33% | 50.83 pp |
| Awfis Space Solutions Ltd. | +55.51% | +13.58% | +39.16% | 41.93 pp |
| P N Gadgil Jewellers Ltd. | +31.76% | +72.92% | +50.00% | 41.16 pp |
| R K SWAMY Ltd. | +25.63% | -13.19% | +31.25% | 38.83 pp |

The errors CSV contains every case, source clock, exact linear-model feature effects, missingness and training-range flags. Later final-demand fields are labelled diagnostic-only and never enter model training. These associations do not establish causation.

No production promotion or Module 2/paper changes. A numerical improvement is not called reliable when its interval includes no improvement.

- All 2024 outcomes have been inspected; this is exploratory retrospective research, not independent prospective proof.
- Different publishers can disagree. Conflicting same-date values are withheld, not averaged or selected by outcome.
- Only pre-decision public dated rows enter features. Paid history and access blocks are not bypassed.
- Final demand in the error CSV is post-cutoff diagnostic context only; never a prediction feature.
- Conditional opening returns are not executable/allotment-weighted portfolio profit.
