# Module 2: ownership evidence is not the same as trade volume

The current price-only baseline replay calls the monitor with empty
disclosures and news fetching disabled. Its 50.0% score is therefore not a
test with verified anchor/promoter-sale evidence. The separate dated-news
replay tested available historical headlines but demonstrated no binary
sell/hold improvement under the current governance.

The rule engine has an anchor-sale exit and a coverage-gated accumulation
branch. These require actual ownership/disclosure evidence. Delivery volume,
large average trades, and announcement subject counts do not identify an
anchor or promoter. `promoter_sell_detected` is also calculated in the core
but does not currently drive the verdict; changing this policy without dated
evidence would not establish higher forecasting accuracy.

## Bounded public-source pilot, not another full collection

NSE's structured PIT feed was queried once per first five sorted development
IPO IDs, independent of their outcomes. Original responses and SHA-256
metadata are preserved in `data/archive/module2_pit_pilot`. Re-runs reuse
these snapshots rather than fetching them again.

| Security | Response rows | Admissible interpretation |
|---|---:|---|
| CMSINFO | 5 | Modes are not explicit open-market purchases/sales; not sale signals |
| SUPRIYA | 0 | No returned rows; not proof that nobody sold |
| HPAL | 0 | No returned rows; not proof that nobody sold |
| DATAPATTNS | 0 | No returned rows; not proof that nobody sold |
| MEDPLUS | 1 | Market purchase of 5,000 shares; role unverified, not an inferred promoter or anchor |

The Medplus record's transaction date is 23 December 2021, but the feed's
publication clock is 24 December 2021 at 23:07. It cannot be used as public
information at the 23 December close. The code checks both clocks and
retains unknown categories as unknown. PIT is a separate channel from bulk
deals and anchor-allocation identity; this pilot establishes neither complete
anchor coverage nor a predictive edge. Three real-source checks passed.

No full-cohort expansion, PDF review, synthetic transaction, live policy
replacement or paper-score change followed from this pilot.

Official reference: [NSE insider-trading disclosures](https://www.nseindia.com/companies-listing/corporate-filings-insider-trading).
Machine-readable pilot: `module2_pit_pilot_20261022.json`.
