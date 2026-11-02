# Fixed-view stacking experiment

Use five fixed complementary views: six-input logistic C=.5; signed-log GMP/base+macro balanced logistic C=.05; signed-log GMP/financial+macro logistic C=.5; trajectory balanced forest depth five/min leaf eight/250 trees; GMP-only logistic C=.5. Do not select a test winner among experts.

Use the previously saved expanding chronological validation predictions for each expert. For each training end year, use its latest two earlier validation years to fit a standardized-logit logistic stacker, C=.5, seed 42. Select a decision cutoff from training-only five-fold stratified cross-fitted stacker probabilities, using the latest training year only, maximizing overall accuracy then balanced accuracy and cutoff nearest .5. All cross-fitting and labels precede the actual test window. Internal calibration scores are not claimed as contemporaneous historical forecasts.

Refit each expert on all eligible data through the training end year, and fit the stacker on the earlier out-of-fold expert predictions. Freeze both through-2024 and through-2025 artifacts and cutoff records before scoring. Strict test remains all original 155 IPOs; updated test remains only the original 69 2026 IPOs. Include the same annual replay. No new views, weights or cutoffs may be chosen based on these test scores. Reused outcomes remain exploratory; no automatic deployment.
