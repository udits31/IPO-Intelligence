# Pre-test recency weighting experiment

The larger-data and cross-fit cutoff experiments did not reach 80%. Retain them. Test the specific hypothesis that older IPO regimes dominate a model intended for the next year.

Use all 186 earlier interim books and all 104 2025 interim books, preserving missing GMP indicators. Keep exactly 155 strict-test IPOs and exactly 69 updated 2026 test IPOs. Do not add any 2026 training label.

Fixed candidates: raw six-input logistic C=.5; signed-log GMP/base+macro logistic C=.05 balanced; signed-log GMP/financial+macro logistic C=.5; the earlier trajectory forest (depth 5, min leaf 8, balanced); GMP-only logistic C=.5. Test four fixed training-age half-lives: none, two years, one year, six months. Weight is 2**(-(training_end_year-listing_year)/half_life), normalized to mean one. Fit weighted scaling/model only on training rows; imputation remains train-only. No new features or test-based family changes.

Generate expanding chronological validation. For each fit horizon, select candidate, half-life and cutoff using only the latest available validation year (2024 for strict fit; 2025 for updated fit), maximizing overall accuracy, then balanced accuracy, Brier, fewer features, cutoff closest to .5 and longer half-life. All missed gains and missed losses count equally. Save the selection before scoring tests. Archive all settings and errors, including rejected results. Any repeated-test gain is exploratory and needs prospective confirmation.
