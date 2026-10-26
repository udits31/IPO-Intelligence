# Module 2 — Volatility-aware calibration of the existing rules

Unchanged 31st-close checkpoint, 30 later sessions, all eligible IPOs; daily volatility from 20 past close-to-close returns only.

| Same later IPOs | 30-session direction accuracy |
|---|---:|
| Default monitor | 50.00% |
| Earlier-data-selected volatility thresholds | 46.97% |
| Always sell | 58.33% |

n=132; balanced accuracy 48.83%. Requested target established: False.

Reference volatility: 2.7001% per day, frozen using known pre-2025 outcomes. Selected: {'preset': 'wider', 'exponent': 0.0, 'lower_scale': 1.0, 'upper_scale': 1.0}.

No live rules, paper, original prices or frozen decisions were changed.

## Limitations

- 2025–2026 outcomes have been inspected previously; this is not a blind holdout.
- Only thresholds are scaled; no claim that volatility alone predicts future direction.
- Earlier outcomes, including their future 30-session windows, must mature before selection.
- Cash dividends, execution slippage and complete issuer news/disclosure coverage are not supplied.
- Later results are not a promotion instruction and cannot justify repeated tuning on those outcomes.
