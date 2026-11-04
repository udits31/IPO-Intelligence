# IPOPremium recovery and remaining data — 26 October 2026

Scope: 447 distinct IPOs across Module 1 and Module 2 training/evaluation cohorts, including 24 recent listed IPOs. No retraining was performed.

## Coverage after recovery

| Input | Training: before → after / 290 | 2026: before → after / 69 | Recent: before → after / 23 |
|---|---:|---:|---:|
| Dated pre-decision GMP | 270 → 270 | 69 → 69 | 23 → 23 |
| Fresh issue / OFS split | 45 → 188 | 5 → 69 | 6 → 23 |
| Category reservation denominators | 107 → 222 | 41 → 69 | 1 → 23 |
| Matched Day-2 and cutoff total | 161 → 161 | 69 → 69 | 1 → 1 |
| Pre-offer annual EPS | 159 → 175 | 59 → 69 | 20 → 23 |
| Pre-offer NAV / book valuation | 158 → 159 | 55 → 59 | 17 → 21 |
| Original IPO lot size | 216 → 272 | 69 → 69 | 22 → 23 |

Static offer facts and valuation fills use explicitly labelled IPO-context assumptions. This is recovery coverage, not proof of historical publication or a new accuracy result.

## Exact lists

- `IPO_PREMIUM_ALL_COHORT_COVERAGE_20261026.csv`: one row per IPO, with source/capture status.
- `IPO_PREMIUM_RECOVERED_FIELDS_20261026.csv`: each newly filled field and its source hash.
- `MISSING_DATA_ACTION_LIST_20261026.csv`: remaining model-input gaps, by company, reason and owner.
- `IPO_DATA_HELP_FROM_YOU_20261026.csv`: dated GMP quotes or compatible cutoff tables still useful from an existing export, plus optional independent pre-close archive evidence for the 24 listed cases. Public facts still recoverable from other sources remain agent work.

## Do not confuse these gaps

A blank day-wise table on an already-listed IPO is missing source coverage, not zero demand. A blocked/throttled page is an incomplete collection, not an absent fact. A final Day-3 table updated after 17:00 is historical closing evidence, not a saved application-time forecast. An approximate application count is not an official final allotment basis.

All original model files and prediction-ledger prefixes remain unchanged.
