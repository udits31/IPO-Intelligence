# Module 1 completion: data coverage and GMP comparison

Module 1 only. No Module 2, selected-paper, saved-prediction or deployed-model changes.

## Data coverage

Universe: 441 issuers; 252 existing PDFs; 228 extracted annual EPS tables. The frozen early-book population has 186 IPOs. Dated GMP coverage increased from 102 to 123 (21 recovered); 63 remain unresolved.

At decision time: 98 usable earnings records; 78 usable OFS ratios; 48 IPOs have all three (earnings, OFS, GMP). 186 canonical closing dates disagree with the verified issue calendar; the verified dates are used.

The coverage CSV accounts for every issuer, including source absence, later document dates, conflicting offer amounts and uncovered bids. Missing data is not converted into an observed zero. Sector-relative financial features use retrospective sector metadata and are not publication-time certified.

## Fixed experiment and selections

Earlier-year matured labels only; 2022/2023 predictions select models. All candidates and all thresholds were fixed before this run. 2024 outcomes have been inspected before: these results are exploratory, not a fresh blind test.

Development-selected direction model: **bids_offer_gmp**. Return model: **r_bids_offer_gmp_huber**. GMP-only fitted return control: **r_gmp_fitted_huber**.

## Direction, return estimates and ranking

| Test population | IPOs | GMP-sign accuracy | Combined direction accuracy | GMP return MAE | Combined return MAE | Same-date ranking panels | Selection difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_primary | 39 | 87.2% | 84.6% | 25.92 pp | 24.50 pp | 4 | +0.00 pp |
| expanded_secondary | 51 | 80.4% | 80.4% | 23.96 pp | 21.43 pp | 7 | -2.13 pp |

MAE is average absolute return error in percentage points; lower is better. Ranking compares the highest positive central estimate with the highest positive GMP on exactly the same available IPOs, or skips all when no positive estimate exists. No feature-range filtering removes difficult cases. Ties use issuer ID, never realized results.

## Detailed paired results

### All fixed input combinations

For each input track below, the return algorithm is selected on development MAE only. These are not test-selected winners.

| Test population | Input track | Direction accuracy | Balanced accuracy | AUC | Return algorithm | Return MAE |
|---|---|---:|---:|---:|---|---:|
| original_primary | gmp_fitted | 87.2% | 50.0% | 0.685 | huber | 24.30 pp |
| original_primary | bids_offer_gmp | 84.6% | 48.5% | 0.753 | huber | 24.50 pp |
| original_primary | full_facts_gmp | 79.5% | 62.6% | 0.753 | huber | 24.56 pp |
| original_primary | sector_relative_gmp | 84.6% | 65.6% | 0.771 | huber | 24.56 pp |
| expanded_secondary | gmp_fitted | 80.4% | 50.0% | 0.679 | huber | 21.34 pp |
| expanded_secondary | bids_offer_gmp | 80.4% | 53.8% | 0.732 | huber | 21.43 pp |
| expanded_secondary | full_facts_gmp | 76.5% | 58.9% | 0.698 | huber | 21.52 pp |
| expanded_secondary | sector_relative_gmp | 80.4% | 61.3% | 0.700 | huber | 21.52 pp |

### original_primary

Direction difference versus GMP-sign: -2.56 points (paired bootstrap 95% interval -7.69 to +0.00); 0 GMP mistakes corrected and 1 new mistakes introduced.

Return-MAE improvement versus GMP: +1.42 pp (95% interval -3.38 to +6.20). GMP-only calibrated control MAE: 24.30 pp.

Ranking: 4 same-date choice sets, 8 non-tied IPO pairs. Reliable ranking advantage established: **no**.

Selection difference, model minus GMP: +0.00 pp (95% interval +0.00 to +0.00). Pairwise ordering: model 100.0% versus GMP 100.0%.

| Date | Eligible IPOs | Combined choice | GMP choice | Combined actual opening return | GMP actual opening return |
|---|---:|---|---|---:|---:|
| 2024-02-08 | 3 | Rashi Peripherals Ltd. | Rashi Peripherals Ltd. | +9.16% | +9.16% |
| 2024-09-10 | 2 | Bajaj Housing Finance Ltd. | Bajaj Housing Finance Ltd. | +114.29% | +114.29% |
| 2024-12-20 | 3 | Mamata Machinery Ltd. | Mamata Machinery Ltd. | +146.91% | +146.91% |
| 2024-12-23 | 2 | Senores Pharmaceuticals Ltd. | Senores Pharmaceuticals Ltd. | +53.45% | +53.45% |

### expanded_secondary

Direction difference versus GMP-sign: +0.00 points (paired bootstrap 95% interval -5.88 to +5.88); 1 GMP mistakes corrected and 1 new mistakes introduced.

Return-MAE improvement versus GMP: +2.52 pp (95% interval -1.54 to +6.44). GMP-only calibrated control MAE: 21.34 pp.

Ranking: 7 same-date choice sets, 14 non-tied IPO pairs. Reliable ranking advantage established: **no**.

Selection difference, model minus GMP: -2.13 pp (95% interval -6.40 to +0.00). Pairwise ordering: model 92.9% versus GMP 100.0%.

| Date | Eligible IPOs | Combined choice | GMP choice | Combined actual opening return | GMP actual opening return |
|---|---:|---|---|---:|---:|
| 2024-02-08 | 3 | Rashi Peripherals Ltd. | Rashi Peripherals Ltd. | +9.16% | +9.16% |
| 2024-09-10 | 2 | Bajaj Housing Finance Ltd. | Bajaj Housing Finance Ltd. | +114.29% | +114.29% |
| 2024-09-18 | 2 | Arkade Developers Ltd. | Arkade Developers Ltd. | +36.72% | +36.72% |
| 2024-10-22 | 2 | Waaree Energies Ltd. | Waaree Energies Ltd. | +66.33% | +66.33% |
| 2024-12-12 | 2 | Sai Life Sciences Ltd. | Vishal Mega Mart Ltd. | +18.40% | +33.33% |
| 2024-12-20 | 4 | Mamata Machinery Ltd. | Mamata Machinery Ltd. | +146.91% | +146.91% |
| 2024-12-23 | 2 | Senores Pharmaceuticals Ltd. | Senores Pharmaceuticals Ltd. | +53.45% | +53.45% |

## Remaining gaps and decision

63 early-book issuers still lack a verified, suitably dated GMP quote after the bounded recovery. Later/undated figures are not a repair. Missing earlier RHP versions and bid histories outside the frozen cohort remain source gaps, not values that can be inferred from listing outcomes.

Remaining GMP gap reasons: {"no_verified_dated_history": 56, "only_same_day_or_later_quotes": 7}. The native-cover date audit corrected 9 document dates; incorporation dates are no longer accepted as prospectus dates.

No automatic production promotion. Improvement is judged against the baselines and uncertainty above, not a requested headline accuracy. A negative result completes this comparison; it does not justify tuning the inspected test. The prediction and ranking ledgers retain every evaluated case.

- Retrospective, previously inspected 2024 outcomes; not blind or prospective proof.
- Recovered coverage is non-random. Original and expanded tests are reported separately.
- Dates on historical GMP/bid tables do not certify original intraday publication clocks.
- Sector labels are static canonical metadata without a verified publication-time history; sector-relative results are exploratory.
- Rankings compare only identical snapshot dates, not invented simultaneous access to different-day figures.
- Selection returns are gross conditional-on-allotment opening returns, not capital-weighted portfolio profits.
- This does not validate arbitrary intraday, first-day, or closing-hour use.
