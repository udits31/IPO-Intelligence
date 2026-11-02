# Module 1: why 36 positive predictions were wrong, and what changed

**Result: the new combination gets 118/155 right (76.1%). It has not reached 80%.**

All fits and model/cutoff choices use 2021–2024 only. All results below use the exact same 155 IPOs from 2025–2026. No losses or flat openings were dropped, no final bids were substituted, and no saved original forecasts were rewritten.

## What the 36 misses show

- 10 were flat openings; 26 opened below issue price. Flat is correctly counted as a missed positive call.
- 35/36 had positive unofficial GMP, so positive GMP was not sufficient evidence of an opening gain.
- 21/36 had GMP of 5% or less. The earlier logistic model often let QIB/HNI evidence outweigh that weak sentiment.
- Training had 77.6% positive openings; this test had 68.4%. The model learned an optimistic base rate that did not transfer fully.
- Independent source checks found nine material bid disagreements. Scoda had 646x retail on one page versus 6.99x on another. These are source conflicts, not a licence to invent decimal corrections. Consistent whole-cohort source substitution did not improve the previous model’s 74.8% accuracy.

## Actual tests

| Experiment | Correct / 155 | Accuracy | Flat/negative caught | Positive openings caught |
|---|---:|---:|---:|---:|
| Previous combined model | 116/155 | 74.8% | 13/49 | 103/106 |
| GMP sign baseline | 106/155 | 68.4% | 3/49 | 103/106 |
| Fitted GMP-only baseline | 106/155 | 68.4% | 0/49 | 106/106 |
| Earlier-validation-selected bid/GMP fix: rejected | 112/155 | 72.3% | 7/49 | 105/106 |
| Financial extension: rejected | 104/155 | 67.1% | 24/49 | 80/106 |
| Demand-trajectory model | 115/155 | 74.2% | 32/49 | 83/106 |
| Original + trajectory combination | 118/155 | 76.1% | 32/49 | 86/106 |

The rejected results remain in their reports. The final combination was selected from three fixed mixture weights using only earlier-year chronological validation; its later-year score was not used to choose its weight or cutoff.

## The final candidate

- Adds how QIB/HNI/retail demand changed from the first bidding day to the decision date, plus earlier GMP movement from the same publisher. It never uses the final day’s bids.
- Combines the original logistic score with the balanced demand-trajectory forest. Earlier-year validation selected 75% trajectory weight and a 47.5% decision-score cutoff. This is a learned call policy; its score is not a certified calibrated probability.
- Fixes 19 of the original 36 wrong positive calls, leaving 17 wrong positive calls. It newly misses 17 previously correct positive openings. Total correct rises by two.
- Paired accuracy change versus previous: +1.29 percentage points; 95% paired bootstrap interval [-6.451612903225806, 9.032258064516128]. This does not establish a reliable overall accuracy improvement.
- Negative recall rises from 26.5% to 65.3%; positive recall falls from 97.2% to 81.1%. Precision among positive calls is 83.5%, which must not be reported as 83.5% overall accuracy.

## Remaining gaps

- Training contains only 170 usable matched interim-book/GMP IPOs; just 38 are nonpositive. The expanding validation cohort contains only 28 nonpositive IPOs. Additional model complexity did not solve that information limit.
- Trajectory fields are missing for 45 earlier-year training IPOs. Financial valuations are observed for 61/170 training IPOs versus 138/155 test IPOs. That unequal coverage limits how well these features can be learned.
- The 2025–2026 outcomes have been examined repeatedly. These are exploratory retrospective results, even though no later-year label entered model fitting or selection. A new prospective test is needed before claiming that a small improvement will persist.
- Public historical pages provide dated rows and fiscal periods, but do not certify original publication times. Original saved pre-close forecasts remain distinct.
- No candidate was deployed automatically; the existing live model, paper and Module 2 remain unchanged.

## Files

- Per-IPO review of all 36 original false positives: `module1_downside_investigation_20261024_36_error_review.csv`.
- All 155 original/new calls and actual exchange opens: `module1_blended_fix_20261024_predictions.csv`.
- Reproducible research model: `ml/artifacts/module1_blended_fix_research_20261024.joblib`.
- Model-selection traces, source hashes and every rejected experiment are retained alongside this report.
