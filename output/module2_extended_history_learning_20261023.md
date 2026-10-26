# Module 2: policy drop and learned daily-history challenger

Same 132 later IPOs. Changed 32 old SELL calls to review alerts; old calls were correct 18 times, the no-exit proxy 14 times.

| Current call | IPOs | Correct under binary proxy |
|---|---:|---:|
| HOLD (WATCH) | 29 | 14 |
| HOLD / CONSOLIDATION | 42 | 16 |
| SELL / REDUCE | 29 | 18 |
| WATCH / PRICE PULLBACK | 32 | 14 |

Challenger selected on earlier confirmation: shallow_forest.
Training: 2643 real checkpoints from 214 earlier issuers, each issuer total weight equal.
Confirmation n=61: 62.30% accuracy; later n=132: 59.85% accuracy, 58.57% balanced accuracy.
Always-sell later baseline: 58.33%.

No live model, monitor policy or paper was changed.

- Research after previous inspection of later results; not a blind test.
- Feature fitting and candidate choice use only completed earlier IPO outcomes.
- Repeated earlier checkpoints are issuer-weighted, not independent samples.
- No as-published news/filing forecast or transaction-cost advantage established.
- WATCH is not a HOLD forecast; the legacy binary comparison interprets no exit as hold.
