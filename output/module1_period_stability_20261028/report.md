# Chronological stability versus GMP-only

2023–2025 expanding causal base/meta fits; 2026 same actual frozen release and fitted GMP expert

Both direction thresholds fixed at 0.50. Primary rows require observed dated GMP and verified Day-2 stage. All arms use identical issuer IDs and listing outcomes.

| Test year | IPOs | Five-expert correct | Learned GMP-only correct | Raw GMP > 0 rule | Always positive | Model minus learned GMP |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 38 | 27 (71.1%) | 32 (84.2%) | 84.2% | 84.2% | -13.1 pp |
| 2024 | 68 | 54 (79.4%) | 54 (79.4%) | 79.4% | 79.4% | +0.0 pp |
| 2025 | 96 | 67 (69.8%) | 65 (67.7%) | 67.7% | 67.7% | +2.1 pp |
| 2026 | 64 | 54 (84.4%) | 45 (70.3%) | 70.3% | 70.3% | +14.1 pp |

## Quarterly checks

| Quarter | IPOs | Model accuracy | GMP-only accuracy |
|---|---:|---:|---:|
| 2023Q1 | 2 | 100.0% | 100.0% |
| 2023Q2 | 3 | 66.7% | 66.7% |
| 2023Q3 | 16 | 68.8% | 100.0% |
| 2023Q4 | 17 | 70.6% | 70.6% |
| 2024Q1 | 18 | 66.7% | 66.7% |
| 2024Q2 | 9 | 100.0% | 100.0% |
| 2024Q3 | 21 | 81.0% | 81.0% |
| 2024Q4 | 20 | 80.0% | 80.0% |
| 2025Q1 | 10 | 80.0% | 80.0% |
| 2025Q2 | 13 | 61.5% | 61.5% |
| 2025Q3 | 46 | 71.7% | 69.6% |
| 2025Q4 | 27 | 66.7% | 63.0% |
| 2026Q1 | 17 | 76.5% | 29.4% |
| 2026Q2 | 8 | 75.0% | 62.5% |
| 2026Q3 | 39 | 89.7% | 89.7% |

All-69 2026 sensitivity: model 56/69 (81.2%), GMP-only 46/69 (66.7%).

{"years_beating_gmp": 2, "years_tied_gmp": 1, "years_losing_gmp": 1}

Previously examined retrospective data; small quarters are descriptive, not independent validation
Scores for other bidding stages remain provisional. Historical Day-2 results cannot certify intraday or closing-day performance. The learned GMP-only control and the raw GMP-sign rule are both reported in JSON; they are different baselines.

The expert architecture was chosen during earlier development. Earlier-year folds refit weights causally but are retrospective diagnostics of that fixed design, not untouched architecture-selection holdouts.

Actual app GMP-only artifact at its separately locked 0.625 cutoff: 55/69 (79.7%). This additional 2026-only comparison preserves the app baseline; it is not retrospectively applied as a selected policy to earlier validation years.
