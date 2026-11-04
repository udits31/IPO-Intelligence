# Live-data review — 26 October 2026

As of 18:19 IST. Models and policies were not changed.

## Module 1

655 saved records across five distinct IPOs. Their listing outcomes remain pending. All current records are provisional; repeated captures do not create additional independent test IPOs.

| IPO | First score today | Latest score | Stage | Application screen |
|---|---:|---:|---|---|
| BLS Polymers | 69.9% | 69.3% | pre_bidding_estimate | DATA_PENDING |
| Bombay Coated & Special Steels | 53.3% | 54.6% | pre_bidding_estimate | DATA_PENDING |
| EAAA India Alternatives | 58.6% | 46.7% | pre_close_forecast | DATA_PENDING |
| M.K.Sons Fine Jewels | 50.1% | 50.5% | pre_bidding_estimate | DATA_PENDING |
| SFC Environmental Technologies | 62.1% | 60.9% | pre_bidding_estimate | DATA_PENDING |

Four upcoming IPOs have no bidding categories yet. EAAA has category readings but lacks Day-2 growth history. OFS ratio, retail quota and total-book multiple remain missing from the application-screen inputs; these are pipeline/coverage gaps, separate from the model direction score.

## Module 2

19704 distinct minute candles, 19705 stored versions, 93 alerts across 26 IPOs. Current-day closing-price events exist for 46 IPOs. Historical morning candles were downloaded; prospective alerts begin only in the afternoon.

### Exploratory 15-minute follow-through

| Signal | Saved alerts | Evaluable | Direction followed | Mean subsequent price move |
|---|---:|---:|---:|---:|
| REVIEW_EXIT | 20 | 10 | 8/10 | -0.264% |
| VWAP_LOST | 39 | 23 | 11/23 | +0.163% |
| VWAP_RECLAIMED | 34 | 19 | 4/19 | -0.170% |

Overall follow-through: 23/52; always-up direction: 18/52 on the same event pairs. 41 alerts lack the exact next-minute/15-minute horizon within captured session data.

This is a post-hoc, one-afternoon diagnostic, not validated Module 2 accuracy or a profit backtest. The sampled alert events overlap and repeat issuers. Execution prices are candle-close proxies; transaction costs, spreads and inability to sell at a circuit are not included. No result changes a locked benchmark.

All per-event numbers, exact timestamps, exclusions, feature changes and missing inputs are retained in the companion JSON.
An always-down direction would match 32/52 on the same event pairs, versus the mixed alert directions at 23/52. Therefore the aggregate diagnostic does not beat both constant-direction baselines. No trading edge is established.
