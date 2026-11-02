# Module 1 completion protocol

Scope: coverage repair and research comparison only. No Module 2, paper, saved live forecast or deployed-model changes.

Frozen before the new experiment:

- Population: existing 186 eligible 2021–2024 penultimate-session IPO observations. Report all 441 corporate-universe issuer records separately.
- Primary confirmation: the same original 39 GMP-covered 2024 IPOs, with unchanged exchange-open outcomes. Expanded recovered coverage is secondary; never silently replace the primary denominator.
- Training: earlier listing years, with every training listing outcome strictly before the first test decision. 2022/2023 out-of-fold predictions select candidates; fixed 2024 confirmation. Previously inspected outcomes make all of this exploratory, not blind validation.
- GMP: latest explicit dated rupee quote strictly before the bid decision date, at most seven calendar days old. Never use same-day clockless, closing-book, listing-day or undated quotes. Conflicts are missing, never guessed or zero-filled.
- Recovery: retry all missing-GMP early-book issuers using publisher history with issuer, offer-era and offer-date/price checks. Store raw response and per-issuer failures. Additive research files only; preserve existing canonical inputs and prior results.
- Inputs: interim log bids, announced cap price and issue size, dated GMP; optional annual EPS-derived earnings yield/growth, loss flag, and OFS ratio. Missing fundamentals use training-only imputation and indicators. Source/date exceptions remain visible.
- Sector-aware variant: replace earnings yield and EPS growth with training-sector-relative, 5th/95th-percentile clipped values; broad financial/nonfinancial metadata is retrospective and not publication-time verified. No test-data statistics.
- Direction: three fixed logistic tracks (bids/offer terms/GMP; full facts/GMP; sector-relative facts/GMP), C=0.5 and cutoff=0.5. Compare with GMP sign and a fitted GMP-only logistic baseline. Select one combined track by pooled development Brier score only.
- Return: learn a residual correction to dated GMP. For each track use fixed Ridge(alpha=20) or Huber gradient boosting (100 trees, depth 2, minimum leaf 10, learning rate .04, seed 42). Select by pooled development MAE only. GMP percentage itself is the raw-return baseline; a fitted GMP-only correction is an additional control.
- Ranking: rank central return against GMP percentage on identical same-snapshot-date choice sets, all eligible cases retained. Deterministic issuer-ID ties; no invented allotment probabilities. Report panel counts, paired choices, actual opening returns, pairwise ordering and uncertainty. No pooling IPOs from different dates to manufacture choice sets.
- Uncertainty: paired issuer bootstrap for return error and direction differences; paired decision-date bootstrap for selections. With few panels, make no general ranking-edge claim.
- Completion: deliver coverage/exceptions, direction/return/ranking comparison, every test prediction and every paired choice, regression tests and an explicit promotion/no-promotion decision. A negative comparison is a completed experiment, not permission to keep tuning the inspected test.
