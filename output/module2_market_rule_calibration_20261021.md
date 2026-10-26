# Module 2 — Observed market-regime rule calibration

The existing loss/relative-lag/trailing rules are retained. The experiment tests a small fixed set of tighter/default/wider settings conditioned only on the observed Nifty return, not the listing year or future market direction. Settings are selected on matured 2021–2024 IPO outcomes.

| Same later cases | 30-session accuracy |
|---|---:|
| Current rules | 50.00% |
| Earlier-data-selected market settings | 53.03% |
| Always sell | 58.33% |

Cases: 132; selected settings: {'regime_threshold_nifty30_pct': 0.0, 'weak_market': 'wider', 'strong_market': 'tighter'}. Balanced accuracy: 51.95%. 75–80% target established: False.

No live policy or paper was changed. The goal remains incomplete unless both full requirements are met.

## Limitations

- Market co-movement and nonstationarity limit transfer from earlier years.
- Later outcomes were inspected in previous experiments; not a blind holdout.
- No news, transaction-cost execution guarantee or cash-dividend total return is inferred.
- This rule experiment is not a replacement learned Module 2 classifier or a paper-result update.
