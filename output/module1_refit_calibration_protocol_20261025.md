# Refit-scale calibration check

Freeze the coverage experiment's selected model families and hyperparameters. No new model grid and no 2026-based selection. The question is whether a cutoff chosen from small expanding-fold fits transfers to a larger final fit's probability scale.

For the strict model, train/calibrate only on the 186 IPOs through 2024 and test the identical 155 later IPOs. For the updated model, train/calibrate only on the 290 IPOs through 2025 and test the identical 69 2026 IPOs.

Generate training-only out-of-fold scores with five stratified folds, shuffle seed 42, refitting imputation/scaling/model in every fold. Every row is scored without its own label in that fit. Use out-of-fold scores from the latest training year (2024 or 2025) to choose the cutoff on the fixed .35–.85 grid, maximizing overall accuracy then balanced accuracy and cutoff proximity to .5. This is retrospective training-time cross-fitting, not a claim that those internal 2024/2025 scores were available before their own decisions. All those labels were known before the actual 2025/2026 test window. No test issuer or test label enters these fits.

Keep the already-fitted final classifier unchanged, apply only the earlier-data-selected new cutoff, and compare its actual test errors with the original cutoff. Include an identical calibration check for the fixed GMP-only control. Archive the out-of-fold scores, thresholds and membership hashes before scoring. Do not deploy or reinterpret precision as accuracy. The repeated test remains exploratory evidence.
