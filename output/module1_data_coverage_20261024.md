# Module 1 data coverage and remaining exceptions

## What is covered

Issuer-universe coverage and usable pre-decision feature coverage are different. The frozen modeling protocol is 2021–2024 penultimate-session books; 2025–2026 rows are inventoried but not quietly substituted into this experiment.

| Listing year | Issuers | Existing PDFs | Extracted annual EPS | Eligible early books | Dated GMP at decision | Earnings at decision | OFS at decision |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 64 | 64 | 62 | 42 | 24 | 7 | 7 |
| 2022 | 39 | 39 | 36 | 28 | 18 | 10 | 6 |
| 2023 | 58 | 58 | 52 | 41 | 30 | 25 | 21 |
| 2024 | 91 | 91 | 78 | 75 | 51 | 56 | 44 |
| 2025 | 108 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026 | 81 | 0 | 0 | 0 | 0 | 0 | 0 |

## Repairs completed

- Retried all 84 missing-GMP early-book issuers; found 28 dated histories. Only 21 have a quote strictly before the decision and no more than seven days old. Usable GMP coverage: 102 to 123; 2024: 39 to 51.
- Corrected 9 cover dates using the prospectus title block, not incorporation narrative. Only changed records were re-extracted; this was native text, not OCR.
- Preserved the prior offer-placeholder fix and amount/share-count conflict guard. Conflicting ratios are withheld.
- Used verified exchange offer dates instead of 186 legacy fiscal-date placeholders in this early-book population. The canonical database, deployed model and saved predictions were not rewritten.

## Gaps that remain

- GMP: 63 unresolved. Reasons: {"no_verified_dated_history": 56, "only_same_day_or_later_quotes": 7}.
- Earnings at decision: 88 unavailable. Reasons: {"missing": 3, "not_available_before_decision": 74, "retrospective_dated_rhp": 11}. A later final Prospectus cannot be backdated into an earlier bidding decision; its earlier RHP is needed.
- OFS at decision: 108 unavailable. Reasons: {"conflicting_amount_and_share_count": 2, "missing": 106}.
- 189 universe issuers lack a uniquely matched local PDF. They remain in the coverage inventory; no data was fabricated for them.
- Daily GMP and bid histories were recovered retrospectively. A displayed historical date does not prove the original intraday publication time.
- Sector metadata is retrospectively recorded; sector-relative model results are exploratory.

## Deliverables

- module1_completion_20261024_coverage.csv: every issuer, source/date status and explicit gap reason.
- module1_completion_20261024.md: paired direction, return and ranking comparison.
- module1_completion_20261024_predictions.csv: every evaluated IPO and every fixed model prediction.
- module1_completion_20261024_ranking_choices.csv: exact eligible sets and paired choices.

No Module 2 or conference-paper changes. Remaining source gaps are not claimed solved.
