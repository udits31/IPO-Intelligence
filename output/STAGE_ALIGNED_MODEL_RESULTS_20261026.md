# Stage-aligned IPO model tests — 26 October 2026

Training remains 2021–2025 (290 original IPOs); primary comparisons remain the same 69 earlier 2026 IPOs and 23 verified recent listing opens. Five experts and the C=0.5 log-odds stack are preserved. No probability calibration, new live model, ledger rewrite or git push.

## Module 1: actual comparison

| Candidate | Earlier 2026 correct | False positives among 23 losers | Recent correct | Promotion gate |
|---|---:|---:|---:|---|
| M1_V0_FROZEN_PRIOR | 56/69 (81.2%) | 5/23 | 15/23 (65.2%) | Fail |
| M1_V1_FROZEN_WITH_RECOVERED_INPUTS | 56/69 (81.2%) | 5/23 | 16/23 (69.6%) | Pass |
| M1_V3_QUOTA_DENOMINATORS | 52/69 (75.4%) | 8/23 | 16/23 (69.6%) | Fail |
| M1_V4_OFS_SPLIT | 54/69 (78.3%) | 5/23 | 15/23 (65.2%) | Fail |
| M1_V5_PROGRESSIVE_REGIME | 52/69 (75.4%) | 9/23 | 15/23 (65.2%) | Fail |
| Separate Day-2 track (273 train IPOs) | 55/69 (79.7%) | 10/23 | 17/23 (73.9%) | Fail |

Best eligible research result: the frozen weights with recovered financial/category inputs. These are retrospective recomputations, not new calls saved while bidding was open. Neither the retrained quota/OFS/regime variants nor the Day-2 candidate passed all gates. The existing deployed artifact and saved calls are retained.

### Why Option B cannot honestly be fitted yet

The 161 earlier matched-total rows do not provide 161 distinct Day-2-to-Day-3 pre-close pairs. Of those ratios, 159 equal 1.0: primarily Day-2 totals divided by the same Day-2 totals. The source audit found zero training pairs meeting the requested distinct-day, category-matched, closing-day-before-17:00 criteria. A final-book inflation offset likewise needs paired pre-close and post-close books, which are not present. The regularised forward/inverse bridge is implemented and input-guarded; no synthetic readings or arbitrary discounts were inserted.

Recovered Day-2 categories: 273/290 training IPOs, 69/69 earlier 2026 IPOs and 24/24 recent IPOs. The 22 recent total-ratio gaps were not 22 missing category-growth rows. The new replay computes growth from the current category book versus the recovered actual Day-2 categories.

The dedicated Day-2 evening model was fitted on 273 training issuers with a recoverable dated Day-2 table. Its macro data is restricted to the Day-2 date; same-day GMP with no publication clock is withheld. It is a research candidate and has not been routed into live predictions because its false positives increased.

### 24th issuer: separate supplied-outcome sensitivity

The task supplies NSE/BSE issue price Rs1,785 and opening Rs1,800 (+0.84%). There is no bound primary BSE opening record in the local outcome ledger. The JSON includes a separate 24-case sensitivity using those supplied numbers; it does not silently change the verified 23-case denominator.

### Capital-allocation results

The JSON includes APPLY precision, SKIP recall and allotment-weighted retail rupee scenarios for every eligible candidate. Reciprocal subscription multiples are approximations, not exact SEBI allotment probabilities. Original lot costs are used, not hard-coded 14/68-lot applications. Exact sNII/bNII outcomes remain unreported where dated split books and validated allotment basis/application counts are absent. Conditional allocation and full-allotment loss dependence remain unverified.

## Module 2: deterministic rules only

| Rules | Correct /132 | Winner sensitivity /55 | Loser specificity /77 | Mean net return, original protocol |
|---|---:|---:|---:|---:|
| M2_R0_OLD_BASELINE | 62/132 (46.97%) | 44/55 | 18/77 | -0.510% |
| M2_R1_WITH_DATED_FILINGS | 63/132 (47.73%) | 44/55 | 19/77 | -0.362% |
| M2_R2_FLOOR_AND_SMA20_GATE | 70/132 (53.03%) | 15/55 | 55/77 | -0.522% |
| M2_R3_PLUS_RELATIVE_STRENGTH | 69/132 (52.27%) | 14/55 | 55/77 | -0.548% |
| M2_R4_FULL_5_GATE_PROVE_IT | 70/132 (53.03%) | 13/55 | 57/77 | -0.460% |
| ALWAYS_SELL | 77/132 (58.33%) | 0/55 | 77/77 | -0.300% |

The full five-gate policy catches more losers but retains only 13 of 55 winners. It does not beat always-sell on accuracy or mean return. The learned classifier is not used by this evaluation or by the existing core monitor; the underperforming candidate is kept separate from the active deterministic monitor. Every gate exposes failed versus unknown evidence.

Protocol: decision at the 31st observed close; execution at the next observed close; direction measured over the next 30 sessions; original selling-cost treatment retained. The JSON also provides consistent exit costs on both sell and retained positions. Exchange trading dates determine retrospective tape eligibility; unclocked same-day EOD data is not presumed published by 15:30. Filings require eligible broadcast times; archived allotment dates remain an explicit assumption.

## Remaining data required for the bridge

1. For training issuers: category-level Day-2 readings paired with a distinct closing-day 14:00–before-17:00 reading, each with its own source time. Existing Day-2/Day-2 ratios cannot train this mapping.
2. For final-book deflation: pre-close and post-close category books for the same training issuer; no test issuer may fit the offset.
3. Recover Day-2 categories for the 17 training IPOs in module1_remaining_day2_category_gaps_20261026.csv. All 24 recent IPOs already have category readings.
4. Bind the supplied NSE opening to a primary BSE archive before including it in verified accuracy.
5. Exact HNI rupee evaluation needs dated sNII/bNII books and a validated basis of allotment, not approximate application counts.

These inspected cohorts have been used for repeated development comparisons. Any selected improvement needs a new prospective cohort before claiming independent generalisation.
