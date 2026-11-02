# Stage-matched bids and dated fundamentals challenger

2022/2023 development folds; unchanged 2024 confirmation IPOs. Fixed C=0.5 and direction cutoff=0.5; no test-driven tuning.

| Same 2024 early-book IPOs | Accuracy | Balanced accuracy |
|---|---:|---:|
| legacy_final_fit_on_interim | 56.0% | 64.7% |
| stage_bids_only | 81.3% | 58.3% |
| stage_plus_offer_terms | 84.0% | 59.9% |
| stage_plus_dated_fundamentals | 82.7% | 59.1% |

Confirmation: 75 IPOs; 4 have dated annual EPS facts; 0 have an unambiguous pure-offer ratio.
All cases remain in every comparison. Missing facts are not a reason to remove a losing IPO.

Always-positive accuracy: 81.3%.

Separate penultimate-session and final-book artifacts were saved as research-only. The live model and old saved predictions were not replaced.

- RHP dates and retrospective daywise tables do not certify original publication-time availability.
- Missing fundamentals use training-only median imputation and missingness indicators, never observed zero.
- Bounded 48-document sample has sparse financial coverage; no broad valuation advantage is claimed.
- Penultimate-session training does not validate first-day, closing-hour or arbitrary intraday forecasts.
- 2024 outcomes have already been inspected in other research; not blind or prospective confirmation.
- Offer cap price and issue size come from canonical offer terms, not post-listing prices.
