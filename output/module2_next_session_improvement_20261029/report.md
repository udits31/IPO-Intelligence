# Module 2: dated-news recovery and execution-aligned retraining

No sufficiently supported advantage over both rules and buy-and-hold; keep challenger in research.

{'train': {'rows': 5115, 'issuers': 136, 'first': '2021-02-01', 'last_label': '2024-02-14'}, 'validation': {'rows': 2632, 'issuers': 72, 'first': '2024-02-15', 'last_label': '2024-12-31'}, 'final_train': {'rows': 7958, 'issuers': 207, 'first': '2021-02-01', 'last_label': '2024-12-31'}, 'test': {'rows': 5163, 'issuers': 159, 'first': '2025-02-03', 'last_label': '2026-10-16'}}

| Policy | Mean net issuer return | Median | Mean drawdown | Buy fills | Direction accuracy |
|---|---:|---:|---:|---:|---:|
| execution_aligned_price | 1.768% | 0.000% | -8.480% | 649 | 56.25% |
| execution_aligned_flow | 1.628% | -0.347% | -11.941% | 468 | 54.66% |
| execution_aligned_events | 1.983% | -0.363% | -11.384% | 496 | 52.86% |
| current_rules | 2.599% | 0.000% | -11.145% | 198 | — |
| always_hold | 1.942% | -0.172% | -14.337% | 159 | — |
| always_cash | 0.000% | 0.000% | 0.000% | 0 | — |
| momentum | -0.519% | -2.599% | -10.211% | 566 | — |
| reversal | -2.246% | -1.945% | -11.183% | 1312 | — |

Selected using 2024 only: execution_aligned_price / boosting; policy ('return_hysteresis', 0.35, 0.0).
Paired net gain vs current rules: {'mean_gain_pct': -0.8302293785435984, 'bootstrap95_pct': [-2.866499754050048, 1.0827020409698787], 'improved_issuers': 84, 'worsened_issuers': 71, 'ties': 4, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}.
News/filing coverage: {'train': {'rows': 7958, 'news_rows': 1010, 'filing_rows': 1630}, 'test': {'rows': 5163, 'news_rows': 1561, 'filing_rows': 3364}}.

{'training': '2021-2023 listings; 2024 validation; final 2021-2024; test 2025-2026. Disjoint issuer IDs and later-label maturity purges.', 'forecast': 'After completed session, predict next observed close / current close - 1.', 'execution': 'Signal at 18:00 IST, execute at next open. Learned trading target: following open / next open - 1. Carry overnight; change positions only on signals. Liquidate at last common test close.', 'news': 'Verified issuer headlines only, trailing seven days, previous calendar dates only. Exchange events require source publication timestamp <= decision. Current fetch time never supplies a historical clock.', 'costs': '35bps per round trip (17.5bps each buy and sell), stress tested at 15/60bps. No shorting or leverage.', 'selection': 'Maximum 2024 mean net issuer return among candidates with >=10 validation buys. No choice based on later results.', 'caveats': ['Previously inspected later years are retrospective development tests, not blind proof.', 'Historical EOD bars assumed available by 18:00; source publication clocks are not exhaustive.', 'Windows begin at session 21; early listing-day forecasts are not evaluated.', 'RSS archive results are incomplete; empty results do not establish absence of news.', 'Equal initial capital per issuer, varying windows, no annualisation; circuit fills, slippage beyond assumed costs and shared calendar dependence need prospective verification.']}
