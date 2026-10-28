# Stage-matched Module 1 tests

Each stage gets its own fitted five-expert model. Train through 2025; test the same 69 2026 issuers when stage coverage permits. Fixed 50% cutoff. GMP and macro features rebuilt as of each source clock.

| Stage | Training IPOs | Test IPOs | Model (0.50) | GMP-only (0.50) | GMP-only (fixed 0.625) | Raw GMP > 0 (observed quotes only) |
|---|---:|---:|---:|---:|---:|---:|
| DAY1_EOD | 275 | 69 | 50/69 (72.5%) | 66.7% | 81.2% | 46/66 (69.7%) |
<!-- Observed dated GMP coverage: 66/69. -->
| DAY2_EOD | 273 | 69 | 53/69 (76.8%) | 66.7% | 78.3% | 46/68 (67.6%) |
<!-- Observed dated GMP coverage: 68/69. -->

Closing-day pre-17 training status: No eligible training stage
The all-case learned baseline uses training-only imputation for missing GMP. Raw GMP > 0 is not a valid all-case baseline where GMP is absent; use paired_observed_gmp in JSON for that comparison.
Retrospectively recovered source timestamps are not original immutable predictions. No model weights or forecast ledgers were replaced.

## Pre-2026 checks

| Stage | 2024 model | 2024 GMP (0.625) | 2025 model | 2025 GMP (0.625) |
|---|---:|---:|---:|---:|
| DAY1_EOD | 81.3% | 81.3% | 68.0% | 68.0% |
| DAY2_EOD | 80.0% | 80.0% | 69.6% | 69.6% |

Do not promote either stage candidate: neither beats the fixed-0.625 stage-fitted GMP baseline on the 69-case test. Current deployed artifact and ledgers remain intact.

Initial all-OOF combiner refit did not match the frozen recent-two-year combiner recipe. Preserved as previous_all_oof_diagnostic; this correction is a retrospective implementation audit, not fresh model selection.
The final results include already-audited single dated GMP quotes where strictly earlier than the stage, in addition to the clocked quote graphs. Initial graph-only coverage was incomplete; these final rows supersede the preliminary graph-only figures.
