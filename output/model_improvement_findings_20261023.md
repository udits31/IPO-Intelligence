# Model accuracy: what failed, what changed, and what is not solved

## Exact explanation of the Module 2 drop

The 8% price-pullback branch previously said SELL. It now flags a review,
not an automatic exit. The binary back-test interprets no exit as holding.
That is an evaluation convention, not proof that a review alert predicts a rise.

Across the same 132 IPOs, this changes 32 calls. Eighteen subsequently fell
and fourteen rose. The old calls got eighteen right, the new no-exit proxy
fourteen: four fewer successes, from 66/132 (50.00%) to 62/132 (46.97%).

| Current branch | Cases | Correct under the binary convention |
|---|---:|---:|
| SELL / REDUCE | 29 | 18 |
| HOLD / CONSOLIDATION | 42 | 16 |
| HOLD (WATCH) | 29 | 14 |
| WATCH / PRICE PULLBACK | 32 | 14 |

The largest weakness is assuming no exit means the stock will rise.
The current back-test has eleven sells before a rise and fifty-nine no-exits
before a flat/down outcome. Merely restoring the old 8% sell instruction would
return to 50%, not solve the requested 70% problem.

## A new learned Module 2 challenger

The reconciled development file keeps roughly the first 62 trading sessions
per IPO. This limits most training issuers to one matured 30-session label.
A price-history challenger using those records scored 66/132 (50%).

The original, real secondary-provider histories contain longer earlier-year
series. A separate experiment used 2,643 completed checkpoints from 214
2021–2024 issuers, with equal total weight per issuer. It uses price momentum,
mean reversion, volatility and same-date market/sector comparisons. Four
candidate configurations were selected using earlier confirmation cases;
2025–2026 outcomes did not choose the configuration or its 50% threshold.
Training labels must have completed by the end of 2024.

| Same 132 later IPOs | Correct | Accuracy |
|---|---:|---:|
| Current rule/no-exit proxy | 62 | 46.97% |
| Learned extended-history challenger | 79 | 59.85% |
| Always sell | 77 | 58.33% |

The challenger corrected 41 current-rule mistakes but introduced 24 new ones.
Its paired improvement over the current rules is +12.88 percentage points,
with a descriptive 95% bootstrap interval of +1.52 to +24.24 points.
Its improvement over always selling is only +1.52 points, with an interval
of -9.09 to +12.12 points. A reliable advantage over that baseline is NOT
established. Prior later-year results have already been inspected, so these
are exploratory comparisons, not blind validation.

Development prices in the extended experiment are retrospective Yahoo
histories, not fully exchange-reconciled extended histories. The test prices
and all 132 outcome definitions remain the original reconciled exchange cases.
No synthetic observations were used. No candidate was deployed.

## Module 1

The deployed 3-input model remains 136/179 correct (76.0%) at the direction
threshold. Its misses are 24 predicted-positive IPOs that opened flat/down,
and 19 predicted-flat/down IPOs that opened positive.

A new bounded test used the same three inputs and all 252 original training
issuers, with consistently exchange-verified labels. It compared four recency
weightings using only 2022–2023 chronological predictions. The selected
two-year half-life scored 74.9% on the same 179 later cases: no improvement
over the unweighted corrected-label candidate and below the deployed model.
It was not deployed. No threshold was tuned on the 179 later answers.

## What would justify the requested scores

Neither 70% Module 2 nor 80–85% Module 1 is demonstrated by these experiments.
The remaining work is better dated predictive evidence, not changing a
headline: reconcile extended development prices, add verifiable as-published
earnings/valuation and event evidence, and test an earlier-selected challenger
on future untouched IPOs. Price-based risk limits and return forecasts must be
separate outputs. Review alerts cannot silently become confident HOLD forecasts.

The live monitor, Module 1 bundle, manuscript and existing performance scores
were not replaced by these research candidates. All misses are retained in
the accompanying per-company diagnostic and challenger prediction ledgers.
