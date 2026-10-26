# Module 2: unsupported trailing cutoff is an alert, not an automatic exit

The manually chosen 8% trailing cutoff no longer independently produces
SELL / REDUCE or EXIT / CUT LOSS. A breach above issue price now produces
WATCH / PRICE PULLBACK and explicitly says NO AUTOMATIC EXIT. The old level
remains visible only as an unvalidated reference. It is not replaced by an
equally unsupported new percentage or a positive HOLD forecast.

Each monitor response now includes a primary decision reason, method and
limited-history note, displayed beside the verdict. The execution thesis no
longer contradicts this alert by instructing an immediate exit. Other existing
loss, ownership and accumulation rules remain heuristics; this change does
not establish their predictive accuracy or optimize any threshold.

Real-source regression: HD Fire Protect, 22 October close 276.700012 versus
highest prior observed close 306.75; legacy reference 282.21. This now returns
WATCH / PRICE PULLBACK with the explicit manual-cutoff warning. The actual
archived Yahoo response, including the next day's quote, is replayed at the
22 October cutoff without using that later quote.

Policy version: monitor-risk-alert-governance-20261023-v1. Previous scores
are frozen old-policy evaluations, not accuracy for this new action policy.
No old prediction ledgers, model bundles or manuscript results were rewritten.
Sixteen targeted backend tests, seven frontend checks and the production
frontend build passed. Browser visual inspection was not performed.
