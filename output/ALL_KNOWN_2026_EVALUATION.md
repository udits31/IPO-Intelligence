# Frozen-model evaluation: all locally known 2026 IPOs

Model training ends in 2025. No fitting, threshold search or artifact changes were performed. All 2026 canonical IDs are disjoint from training IDs. Source recovery and evaluation are retrospective.

## Coverage

The canonical dataset has 81 records dated 2026: 76 corporate IPOs and five REIT/InvIT investment trusts outside this classifier's corporate training scope. All 76 corporate IPOs now have an evaluated prediction and a verified exchange opening. The earlier 69-case set omitted seven because of data/identity gaps; all seven now have recovered penultimate-session bids. Manika's opening was recovered from its genuine NSE BE-series row.

The separate app cache contributes 24 additional issuer records. All 24 have replay predictions; 23 have verified opens, and NSE's opening remains unverified in this pipeline. No exact normalized issuer-name overlaps were found across the two caches. This gives 105 inventoried records, 100 corporate-model predictions, 99 verified corporate outcomes and five explicitly excluded trusts. This is an audit of local known records, not an exchange-certified complete market census.

## Results

| Evaluation | Model | Fitted GMP-only | Always positive |
|---|---:|---:|---:|
| Previous subset, penultimate stage | 56/69 (81.16%) | 55/69 (79.71%) | 46/69 (66.67%) |
| Expanded canonical corporate set, penultimate stage | 60/76 (78.95%) | 59/76 (77.63%) | 49/76 (64.47%) |
| Recent listings, closing-day stage | 15/23 (65.22%) | 13/23 (56.52%) | 17/23 (73.91%) |
| All known verified corporate outcomes, mixed-stage diagnostic | 75/99 (75.76%) | 72/99 (72.73%) | 66/99 (66.67%) |

The expanded stage-matched result is **60/76 = 78.95%**, rather than 56/69 = 81.16%. On that cohort, the improvement over fitted GMP is only one issuer; it is not proof of a reliable edge. The aggregate **75/99 = 75.76%** combines penultimate-session and closing-day readings and must not be marketed as a uniformly timed or prospective accuracy.

## Seven previously omitted corporate IPOs

| IPO | Model call | Probability of positive | Actual open | Opening return | Result |
|---|---|---:|---:|---:|---|
| Manika Plastech Ltd. | Positive | 57.8% | 43.00 | +0.00% | Wrong |
| Asset Reconstruction Co.(India) Ltd. | Positive | 59.6% | 139.00 | +0.00% | Wrong |
| Kanohar Electricals Ltd. | Positive | 95.4% | 685.50 | +8.47% | Correct |
| Rays of Belief Ltd. | Positive | 50.3% | 239.00 | +0.00% | Wrong |
| Dhoot Transmission Ltd. | Positive | 88.9% | 1200.00 | +37.77% | Correct |
| SBI Funds Management Ltd. | Positive | 77.4% | 613.30 | +6.85% | Correct |
| Waterways Leisure Tourism Ltd. | Flat / negative | 35.4% | 681.00 | -15.72% | Correct |

## Why this is not overtraining

Inference on extra IPOs does not train the model. The frozen artifact SHA-256 is checked before and after evaluation and remains unchanged. No 2026 labels are passed into fitting because this script has no fitting step. Repeatedly modifying features, thresholds or model choice in response to these same test outcomes would nevertheless overfit the evaluation process. These already-inspected cases are diagnostic; a separate prospectively recorded cohort is still necessary.

## Trusts reported outside scope

- Cube Highways Trust
- Bagmane Prime Office REIT
- Citius Transnet Investment Trust
- Property Share Investment Trust-Propshare Celestia
- Raajmarg Infra Investment Trust

## Evidence

Full ledger: `output/all_known_2026_frozen_evaluation.csv`. Input features, raw-source hashes, recoveries, exclusions and group metrics: `output/all_known_2026_frozen_evaluation.json`. Reproduction: `.venv/Scripts/python.exe -m ml.evaluate_all_known_2026`. Model fingerprint: `bf07f15c86fce9984751b36cda7636f3c52875202492199a0017c8842406ff38`. Some missing non-bid features still use the frozen training imputation; their exact names remain in each record.
