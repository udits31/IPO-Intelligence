# Open and upcoming IPO offer-field coverage — 26 October 2026

Offer extraction and application-screen wiring only. Frozen model weights, historical benchmarks and screen rules are unchanged. Existing forecast rows remain immutable.

| IPO | OFS share | Retail reservation | Latest published total | Screen |
|---|---:|---:|---|---|
| BLS Polymers | 0.00% | 35% | Bidding not started | DATA_PENDING |
| SFC Environmental Technologies | 77.53% | 35% | Bidding not started | DATA_PENDING |
| EAAA India Alternatives | 100.00% | 35% | 0.110973x | HARD_SKIP |
| Bombay Coated & Special Steels | 0.00% | 35% | Bidding not started | DATA_PENDING |
| M.K.Sons Fine Jewels | 20.00% | 35% | Bidding not started | DATA_PENDING |

Static sources: identity-checked IPOji pages already archived by the live collector, with SHA-256 and original fetch timestamps in the companion JSON. OFS comes from reported fresh/OFS/total amounts; absence alone is never interpreted as zero. Retail reservation is independent of the remaining QIB book after anchor allocation.

EAAA total comes from the NSE total row: 588,123 bid shares / 5,299,684 offered shares = 0.1109732203x; publisher update 26 October 2026 at 16:51 IST, captured at 16:55 IST. This is the last captured interim bid position, not a fresh evening reading or certified closing book.

Upcoming IPO demand remains not started. Its absence is not an extraction failure. The screen waits for actual demand; no zero subscription was fabricated.

Live Run analysis now appends a current frozen-model assessment from the available ledger and connects the static offer facts and same-book total. Previous saved calls are not rewritten.