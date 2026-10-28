# Frozen comparison: train 2021–2024, evaluate 2025–2026

Primary challenger: the previously selected bids_offer_gmp logistic model, C=0.5, seed 42, threshold 0.5. Inputs are log QIB/HNI/retail at the penultimate bidding session, log announced offer cap, log offer size, and the latest unconflicted dated GMP strictly before that session (maximum seven calendar days old). Fit preprocessing on training only. No later-year model or threshold selection.

Fit on all eligible 2021–2024 early-book/GMP rows whose listing outcomes preceded the first test decision. Evaluate only exchange-verified 2025–2026 openings with matched interim books and dated GMP. Audit issuer ID/name/symbol overlap. A positive opening means exchange open strictly greater than issue price; flat is nonpositive.

Compare on identical test IDs against the actual current deployed model using the same interim bids, raw GMP sign, and fitted GMP-only logistic regression. A separate final-book current-model result is diagnostic and does not represent an actionable pre-close forecast. Report counts, yearly accuracy, balanced accuracy, AUC, Brier and paired uncertainty, alongside all exclusions. Freeze all settings above before scoring.

These are reconstructed historical comparisons, not original forecasts saved before closing. Public dated tables can have been revised. Previously inspected 2025/2026 outcomes mean this is not an untouched or prospective test. Historical offer size in the canonical file is a public-offer descriptor, not a timestamp-certified capture. Do not promote the challenger automatically or edit saved calls, papers or Module 2.
