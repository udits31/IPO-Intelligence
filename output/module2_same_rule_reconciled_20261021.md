# Module 2 — Same-model threshold calibration

The paper’s deterministic monitor family is unchanged. This experiment calibrates only its existing loss, relative-lag loss and trailing-stop thresholds; it does not introduce a new classifier.

Data: 244 real exchange-price development cases; 132 later cases. One 31st-close checkpoint and the same 30-session target per IPO. No synthetic IPOs, price interpolation, future news, post-2024 parameter fitting, excluded losing trades or changed horizons.

Final threshold freeze: 31 December 2024. 214 IPO outcomes were known; 30 development IPO outcomes had not matured and could not choose parameters.

| Evaluation | Default accuracy | Calibrated accuracy | Always sell | Always hold |
|---|---:|---:|---:|---:|
| 2024 confirmation (n=91) | 46.15% | 48.35% | 58.24% | 41.76% |
| 2025–2026 later replay (n=132) | 50.00% | 46.97% | 58.33% | 41.67% |

Selected final thresholds: {'loss_pct': 8.0, 'lag_loss_pct': 0.0, 'trail_pct': 12.0}. Later balanced accuracy: 47.53%. Requested 75–80% accuracy established: NO.

The goal remains incomplete unless both modules satisfy their own full, timing-correct evidence requirements. An improvement below the requested target is progress, not success. The live monitor and paper are not changed.

## Limitations

- 2025–2026 outcomes were inspected previously; not a blind test.
- Price-only replay does not measure the full live news/disclosure monitor.
- Dated NSE bonus/split ratios reconciled; cash dividends and executable trading returns are not measured.
- Changing only thresholds may not supply missing predictive information.
- Bootstrap intervals are descriptive; market co-movement can reduce effective sample size.
