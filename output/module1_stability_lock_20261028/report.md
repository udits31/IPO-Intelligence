# Module 1 stability and release decision

Fixed cutoff 0.50 for all candidates. Policies selected on 2024 and confirmed on 2025 before computing 2026 scores. Final training: 290 IPOs from 2021–2025. Test: identical 69 IPOs from 2026.

| Scoring policy | 2024 accuracy | 2025 accuracy | 2026 correct /69 | 2026 false positives |
|---|---:|---:|---:|---:|
| incumbent_C05 | 80.0% | 70.2% | 56/69 (81.2%) | 6 |
| regularized_C005 | 81.3% | 68.3% | 54/69 (78.3%) | 12 |
| equal_expert_probability_mean | 81.3% | 73.1% | 56/69 (81.2%) | 4 |

Selected before 2026: regularized_C005. Separate 2025 confirmation passed: False. Locked pre-test policy: incumbent_C05.
Actual frozen artifact: 56/69 (81.2%), 5 false positives.
Recommendation: retain_existing_frozen_artifact. No deployment or ledger changes.

A fixed threshold belongs to a fixed versioned scoring artifact. No daily threshold adjustment or automatic retraining is permitted by this release decision. New retraining creates a separate challenger, checked on chronological validation and new prospective outcomes before any future promotion.

2026 repeatedly inspected during development; not an independent new holdout
