# Module 1: do extracted company facts improve prediction?

2022/2023 development folds; unchanged 2024 confirmation IPOs. Fixed C=0.5 and direction cutoff=0.5; no test-driven tuning.

## Same-test comparison

Every row below predicts positive versus flat/negative listing open on the same 75 IPOs. The test uses penultimate-session bids, not final bids or arbitrary intraday readings. The current-model baseline is refitted using its original method and labels on earlier years only; using today's deployed bundle on these 2024 IPOs would test on its own training records.

| Model | Correct / 75 | Accuracy | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|
| Current-model method, retrained on earlier years only | 48/75 | 64.0% | 64.1% | 0.719 |
| Final-book training, corrected listing-open labels | 42/75 | 56.0% | 64.7% | 0.726 |
| Final-book training, same training issuers as interim model | 39/75 | 52.0% | 67.7% | 0.735 |
| Interim-book training, bids only | 61/75 | 81.3% | 58.3% | 0.742 |
| Interim bids + offer price and issue size | 63/75 | 84.0% | 59.9% | 0.748 |
| Interim bids + offer terms + dated earnings, valuation and OFS | 60/75 | 80.0% | 62.9% | 0.719 |

Confirmation: 75 IPOs; 56 have dated annual EPS facts; 44 have a dated OFS ratio (pure or mixed offer).
All cases remain in every comparison. Missing facts are not a reason to remove a losing IPO.

Always predicting positive gets 61/75 correct (81.3%). Balanced accuracy gives equal weight to positive and nonpositive IPOs; ordinary accuracy is inflated by the positive majority.

## What changed and what it means

Offer-price/size model accuracy: 84.0%; full-fundamentals accuracy: 80.0%. Paired differences below distinguish stage matching from adding company facts. These retrospective scores do not establish general live accuracy or validate return-magnitude ranking.

- Interim-book training, bids only: +17.33 percentage points versus the current-model method (paired bootstrap 95% interval +5.33 to +29.33).
- Interim bids + offer price and issue size: +20.00 percentage points versus the current-model method (paired bootstrap 95% interval +6.67 to +33.33).
- Interim bids + offer terms + dated earnings, valuation and OFS: +16.00 percentage points versus the current-model method (paired bootstrap 95% interval +4.00 to +28.00).

## Source coverage

All 441 issuer records were accounted for; all 252 existing PDFs were attempted. Annual diluted EPS was extracted for 228 documents. 189 issuer records have no uniquely matched local PDF. A source dated after the decision is not backdated. Full issuer-level extraction and as-of exceptions are in the coverage CSV.

## GMP-only comparison (separate matched subset)

Only 39 of the 75 test IPOs have the required dated GMP observation. The GMP-above-zero rule gets 87.2% on these same 39 IPOs.

| Model on matched GMP subset | Accuracy | Change versus GMP |
|---|---:|---:|
| Current-model method, retrained on earlier years only | 71.8% | -15.38 points |
| Interim-book training, bids only | 82.1% | -5.13 points |
| Interim bids + offer price and issue size | 84.6% | -2.56 points |
| Interim bids + offer terms + dated earnings, valuation and OFS | 79.5% | -7.69 points |

This subset must not be compared directly with the full-cohort accuracy above.

## Deployment and limitations

Separate penultimate-session and final-book artifacts were saved as research-only. The live model and old saved predictions were not replaced.

- RHP dates and retrospective daywise tables do not certify original publication-time availability.
- Missing fundamentals use training-only median imputation and missingness indicators, never observed zero.
- All existing cohort documents were attempted; extraction and pre-decision coverage are reported separately.
- Penultimate-session training does not validate first-day, closing-hour or arbitrary intraday forecasts.
- 2024 outcomes have already been inspected in other research; not blind or prospective confirmation.
- Offer cap price and issue size come from canonical offer terms, not post-listing prices.
