# Module 2 next-session forecast and trading evaluation

{'decision': '18:00 IST after completed session', 'forecast_target': 'next observed close / current close - 1', 'trading_target': 'next observed close / next observed open - 1', 'execution': 'Long/cash, entry next open, exit same close, no shorting.', 'round_trip_cost_bps': 35, 'sensitivity_bps': [15, 35, 60], 'feature_sessions': '21st through 61st observed session', 'selection': '2021-2023 train, 2024 validation; final 2021-2024; 2025-2026 test, disjoint issuers; outcomes mature strictly before next split.', 'same_day_eod_assumption': 'Historical daily bars and market closes assumed available by 18:00; actual archive publication clocks are not universally recorded.', 'headline_policy': 'Issuer-verified archived headlines, preceding calendar dates only, trailing 7 days; empty coverage remains missing.', 'news_limit': 'Published timestamp permits retrospective reconstruction; retrieval was later, and archive coverage is sparse and not exhaustive.'}

| Model | Test rows / IPOs | Direction accuracy | Close-return MAE | Net per opportunity at 35bps | Trades |
|---|---:|---:|---:|---:|---:|
| price | 5163 / 159 | 54.97% | 2.132% | -0.0004% | 10 |
| price_volume_market | 5163 / 159 | 55.20% | 2.140% | 0.0078% | 38 |
| price_volume_market_news | 5163 / 159 | 54.64% | 2.135% | -0.0006% | 7 |

Selected using 2024 only: price, alpha 10.0, entry threshold 0.35%.

| Baseline (same test rows) | Net per opportunity at 35bps | Trades |
|---|---:|---:|
| always_cash | 0.0000% | 0 |
| always_buy_next_open | -0.5187% | 5163 |
| current_rules_next_open | -0.3641% | 3978 |
| momentum_ret5_positive | -0.2452% | 2400 |
| reversal_ret1_negative | -0.3087% | 2819 |

## Continuous holdings companion

HOLD carries the position overnight; costs apply only to actual position changes. Each issuer starts in cash and ends liquidated. Same test rows, varying issuer windows.

| Strategy | Mean issuer window return after costs | Buy fills | Sell fills |
|---|---:|---:|---:|
| learned_model | 0.051% | 8 | 8 |
| current_rules | 2.599% | 198 | 198 |
| always_hold | 1.942% | 159 | 159 |
| always_cash | 0.000% | 0 | 0 |
| momentum | -0.519% | 566 | 566 |
| reversal | -2.246% | 1312 | 1312 |

Issuer-bootstrap interval for mean net improvement vs rules: [0.14240739122988197, 0.34434707492789796] percentage points.

SELL means exiting an existing holding or staying in cash; short selling is not modeled. All strategies buy at next open and liquidate at the same close for an identical horizon. The existing rules were designed for longer holding periods; their next-session result is a new comparison.

No live policy promotion. Repeated sessions are correlated, not independent IPOs. Archive coverage and equal-capital availability limit interpretation; fills at circuits, market impact, taxes and overnight execution risk need prospective testing.
