# Module 1 other-source recovery and temporal-input experiment

This extends—not overwrites—the completed 39/51-case experiment. All inspected outcomes remain retrospective/exploratory.

- Attempt all 63 remaining missing-GMP issuers against public Top10Stockbroker, IPO Ji, IPOwala, Top10Stock and A2ZIPO pages. Retain raw responses, identity/offer-date checks, source hashes, failures and conflicts. Do not bypass paid history or HTTP access restrictions. Distinct hostnames do not establish independent source families.
- Retry still-unresolved issuers against Investor Academy and A Digital Blogger public archives, retaining the same issuer/date qualification gates. Do not turn unspecified discounts or missing markers into invented numeric GMP.
- Where older pages omit opening/closing dates, require exact issuer identity plus the verified listing calendar date and offer cap (explicit band or the published minimum retail amount divided by the published minimum shares). Listing OPEN PRICES and realized returns are never used for this identity check or as predictors.
- Keep the original 39-case primary and previous 51-case comparison sets visible; report newly covered IPOs separately. Every model in a comparison uses identical training/test IPOs.
- Keep the original logistic C=.5/cutoff=.5 and fixed Ridge/Huber return candidates. Select only using 2022/2023 predictions; do not tune thresholds to 2024.
- Add one prespecified temporal input track: GMP quote age, same-publisher GMP change before the decision, and first-session-to-penultimate-session log bid growth. Compute only from pre-decision rows. Missing histories remain missing; no final-day bids enter model features.
- A reconstructed prior bid history must reproduce the existing penultimate-session bid values within 0.02x. If it conflicts, withhold growth features. Record final books solely in a separately labelled error diagnosis, never as features.
- Explicit within-publisher GMP ranges are retained with both endpoints. Their arithmetic midpoint is a labelled derived input, never represented as a single observed transaction. Between-publisher conflicting quotes remain withheld. Preserve original exact-quote test sets and disclose range-sensitive training inputs separately.
- Compare direction, opening-return error and same-date ranking with raw GMP and a fitted GMP-only correction. A beat in one metric does not imply a beat in the others.
- Diagnose the return-error/ranking-objective mismatch separately: choose a dedicated ranking variant using only 2022/2023 pairwise ordering accuracy with at least five decision-date panels and ten non-tied pairs. Ties prefer fewer inputs then lexical name; insufficient coverage falls back to raw GMP. Preserve the MAE-selected ranking result alongside this selector. This is an exploratory extension, not an untouched confirmation test.
- Diagnose each direction error and the largest return errors using missingness, training-range support, quote staleness, financial inputs and model contributions where applicable. Diagnostics are not causal claims.
- No live promotion, Module 2 changes, paper changes, rewriting saved forecasts or forced success claim.
