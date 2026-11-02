# Module 1: consistent exchange-target training candidate

This is a separate candidate, not a live promotion or paper revision. It uses
the same three log-subscription features, StandardScaler and logistic
regression (C=0.5), with all 252 original corporate training offerings from
2021-2024. No 2025-2026 IPO trains the model. No synthetic training rows,
additional websites or PDF review were used for this change.

## What was corrected

The exchange-opening audit already defined NSE open when eligible and
exact-scrip BSE open otherwise. The deployed bundle still learns from the
canonical labels rather than that audited target. Applying the target
consistently changes the classes for IDs 269, 275, 315, 346, 351 and 371.
These are target-source corrections, not evidence that the old values cannot
represent another exchange or another price convention. The candidate's
archived source hashes, dates, security identities and opens are rechecked
before fitting. Canonical files and the live bundle remain unchanged.

## Results on the same 179 verified later outcomes

| Evaluation | Accuracy | Balanced accuracy | APPLY precision |
|---|---:|---:|---:|
| Corrected-label candidate, fixed direction cutoff 0.50 | 74.9% | 70.5% | 80.0% |
| Corrected-label candidate, earlier-selected APPLY cutoff 0.53 | 76.5% | 73.0% | 82.0% |

The APPLY cutoff is selected only from chronological 2022-2023 predictions,
using the existing balanced-accuracy objective. It was not chosen by scanning
the 179 later outcomes. At that cutoff: TP 100, FP 22, FN 20 and TN 37.
Always-positive accuracy on these same cases is 67.0%. AUC is 0.8167.

The old fit's fixed-0.50 direction score is 76.0%; its deployed 0.82 APPLY
policy scores 73.2% with 90.0% APPLY precision on verified outcomes. The
candidate trades some APPLY precision for recall and overall accuracy. It
must not be presented as uniformly better or as a 90%-precision replacement.

Ten later offerings lack verified outcomes and remain explicitly pending,
not counted as correct. All 252 original training outcomes are verified.
These remain retrospective final-book measurements, not live pre-application
accuracy, investment profits or proof of better IPO ranking than GMP.

## Reproduction and protection

Candidate artifact: `module1_exchange_verified_candidate_bundle_20261022.json`.
The training helper requires an explicit exchange-verified option and rejects
writing that candidate over `ml/model_bundle.json`. The six targeted tests
passed, including exact corrected-label coefficient replay and protection
against silent live replacement. No manuscript, title, commit or push changed.
