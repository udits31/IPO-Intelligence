# Module 1: overall accuracy and forward-year training update

| Experiment | Training IPOs | Test IPOs | Correct | Accuracy | Loss/flat caught | Gains caught |
|---|---:|---:|---:|---:|---:|---:|
| Fit through 2024 | 170 | 155 | 114 | 73.5% | 14/49 | 100/106 |
| Fit through 2025 | 256 | 69 | 51 | 73.9% | 17/23 | 34/46 |
| Annual chronological replay | 170, then 256 | 155 | 115 | 74.2% | 25/49 | 90/106 |

**The through-2025 model is tested only on 2026. Its training IPOs are never counted as test predictions.**

## What changed

- The objective is overall accuracy. Each missed positive and missed negative counts once; there is no forced negative-recall requirement.
- Selection uses the latest two earlier validation years. The strict model uses 2023/2024; the updated model uses 2024/2025.
- Historical market conditions are joined strictly before the decision date, alongside prior bid/GMP trajectories and available IPO-context financial facts.
- Corrected 12 of the previous 20 missed positive openings in the annual replay.

## Selected settings

- Through 2024: {'family': 'logistic', 'transform': 'raw', 'C': 0.5, 'class_weight': None, 'name': 'financial_macro__candidate_02', 'feature_set': 'financial_macro'}; cutoff 0.45; earlier validation accuracy 84.9%.
- Through 2025: {'family': 'logistic', 'transform': 'signed_log', 'C': 0.05, 'class_weight': None, 'name': 'financial_macro__candidate_06', 'feature_set': 'financial_macro'}; cutoff 0.65; earlier validation accuracy 79.9%.

## Full evidence

- Original/current scores, opening prices, dates, cutoffs and correctness flags are stored in the per-IPO CSVs.
- The complete candidate ranking and chronological training memberships were saved before computing later-year scores.

## Limits

- Reused test outcomes make this exploratory retrospective development, not independent proof.
- Training/selection through 2025 is legitimate only for the 2026 test; 2025 rows cannot be counted as its holdout.
- Market closes strictly precede each decision; financial period gates do not certify original publication dates.
- Same issuer IDs, original interim bids and original opening labels are retained. No test winner is substituted.
