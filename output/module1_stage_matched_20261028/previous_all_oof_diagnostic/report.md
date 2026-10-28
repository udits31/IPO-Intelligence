# Stage-matched Module 1 tests

Each stage gets its own fitted five-expert model. Train through 2025; test the same 69 2026 issuers when stage coverage permits. Fixed 50% cutoff. GMP and macro features rebuilt as of each source clock.

| Stage | Training IPOs | Test IPOs | Model (0.50) | GMP-only (0.50) | GMP-only (fixed 0.625) | Raw GMP > 0 (observed quotes only) |
|---|---:|---:|---:|---:|---:|---:|
| DAY1_EOD | 275 | 69 | 49/69 (71.0%) | 66.7% | 81.2% | 46/66 (69.7%) |
<!-- Observed dated GMP coverage: 66/69. -->
| DAY2_EOD | 273 | 69 | 52/69 (75.4%) | 66.7% | 78.3% | 46/68 (67.6%) |
<!-- Observed dated GMP coverage: 68/69. -->

Closing-day pre-17 training status: No eligible training stage
The all-case learned baseline uses training-only imputation for missing GMP. Raw GMP > 0 is not a valid all-case baseline where GMP is absent; use paired_observed_gmp in JSON for that comparison.
Retrospectively recovered source timestamps are not original immutable predictions. No model weights or forecast ledgers were replaced.
