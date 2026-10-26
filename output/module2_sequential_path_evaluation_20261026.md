# Module 2: sequential daily monitoring, not just one checkpoint

Locked strategy uses Python close[31] (Session32 execution) through close[61] (Session62). Direction labels still use Session31 to Session61. Literal one-based Session31-to61 is separately reported and has a different always-sell return because next-close slippage is included.

Decisions use only closes through that session and verified filings published by 15:30 IST. Signals at Sessions 31–60 execute at the next observed close. No locked benchmark, model artifact or prediction ledger is changed.

## LOCKED_EXECUTION_ALIGNED

| Policy | Mean A | Median A | Mean B | Median B | Worst-decile drawdown B |
|---|---:|---:|---:|---:|---:|
| M2_R0_CONTINUOUS | -1.015% | -0.300% | -1.188% | -0.300% | -24.080% |
| M2_R1_CONTINUOUS | -0.867% | -0.300% | -1.038% | -0.300% | -24.080% |
| M2_R1_STATIC_31 | -0.362% | -0.300% | -0.593% | -0.300% | -25.564% |
| ALWAYS_HOLD_31_TO_61 | -1.134% | -5.458% | -1.430% | -5.742% | -28.001% |
| ALWAYS_SELL_AT_31 | -0.300% | -0.300% | -0.300% | -0.300% | -0.300% |
| M2_R0_ALL_SUCCESSIVE_CALLS | -0.494% | -1.092% | -0.721% | -1.389% | -24.709% |
| M2_R1_ALL_SUCCESSIVE_CALLS | -0.346% | -0.753% | -0.571% | -1.051% | -24.709% |

Of the 58 eventual losers initially held, 22 exited later. Of the 44 eventual winners initially held, 6 also exited later. These winner exits are reported with actual return forfeited or saved, rather than automatically called premature.

- held_eventual_losers: mean exited return -9.81997835213325; same-case gross terminal hold return -13.412920021735328; mean benefit after comparable exit costs 3.8527029095368728 percentage points.
- held_eventual_winners: mean exited return -5.257749558778765; same-case gross terminal hold return 19.023133693056838; mean benefit after comparable exit costs -23.923813850756428 percentage points.

Continuous R1 does not beat always-sell and does not beat static R1 on mean net return under consistent liquidation costs. The JSON compares mean, median and worst-decile drawdown separately; one improved metric is not labelled an overall edge.

## LITERAL_SESSION_31_TO_61

| Policy | Mean A | Median A | Mean B | Median B | Worst-decile drawdown B |
|---|---:|---:|---:|---:|---:|
| M2_R0_CONTINUOUS | -0.488% | -1.110% | -0.661% | -1.332% | -24.160% |
| M2_R1_CONTINUOUS | -0.352% | -1.110% | -0.524% | -1.332% | -24.038% |
| M2_R1_STATIC_31 | 0.199% | -0.847% | -0.033% | -1.050% | -25.274% |
| ALWAYS_HOLD_31_TO_61 | -0.449% | -3.322% | -0.748% | -3.612% | -27.448% |
| ALWAYS_SELL_AT_31 | 0.255% | -0.241% | 0.255% | -0.241% | -4.017% |
| M2_R0_ALL_SUCCESSIVE_CALLS | -0.116% | -1.578% | -0.343% | -1.814% | -24.789% |
| M2_R1_ALL_SUCCESSIVE_CALLS | 0.020% | -1.578% | -0.205% | -1.814% | -24.768% |

Of the 58 eventual losers initially held, 22 exited later. Of the 44 eventual winners initially held, 6 also exited later. These winner exits are reported with actual return forfeited or saved, rather than automatically called premature.

- held_eventual_losers: mean exited return -9.430866200188516; same-case gross terminal hold return -12.464928112361985; mean benefit after comparable exit costs 3.2966671278363804 percentage points.
- held_eventual_winners: mean exited return -3.1492790464202725; same-case gross terminal hold return 20.088744155757336; mean benefit after comparable exit costs -22.877756969710333 percentage points.

Continuous R1 does not beat always-sell and does not beat static R1 on mean net return under consistent liquidation costs. The JSON compares mean, median and worst-decile drawdown separately; one improved metric is not labelled an overall edge.

## Limits

- Retrospective corrected price series and archived summaries, not contemporaneously logged daily predictions.
- Verified filing publication times gate every step; unavailable filings do not become neutral evidence.
- Anchor calendar warnings are retained, not promoted to unverified hard exits.
- No rule thresholds or model weights were selected using these outcomes.
- Each IPO is evaluated at equal initial notional; this is not an overlapping, capital-constrained portfolio backtest.
