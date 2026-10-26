# Module 2: what limits the current accuracy

All 132 later IPOs retain the same checkpoint and 30-session target. No live policy is changed.

## Measured experiment

| Policy | Actual later accuracy |
|---|---:|
| Current monitor | 50.00% |
| Earlier-data-selected macro/momentum branch | 49.24% |
| Always sell | 58.33% |

The new branch did not improve the score and is not promoted.

## Why threshold tuning alone is insufficient

The three presets used by the initial earnings experiment can be right on at most 81 of 132 cases (61.36%), even if a hindsight oracle chooses the best preset for every IPO.

The broader, fixed set of 75 price-threshold combinations can be right on at most 94 cases (71.21%) under the same unrealistic assignment.

These are impossibility diagnostics for those specified rule ranges, NOT achieved model accuracy, forecasts or an upper bound on all possible rule-based systems. No hindsight assignment enters prediction features or earlier-data parameter selection.

The 75-80% objective therefore needs genuinely useful evidence-linked decisions beyond those tested threshold ranges. More source rows alone cannot remove this structural limitation. Real earnings/news collection remains useful for testing such decisions, but coverage must not be advertised as predictive improvement.

## Limitations

- Hindsight ceilings use later outcomes only to falsify insufficient rule ranges; they are not achievable scores and never choose actions.
- A limit below 75% applies only to the specified tested thresholds, not to every possible deterministic monitor.
- Later outcomes have been inspected repeatedly; this is exploratory, not blind or prospective validation.
- New risk-off logic is research-only and is not silently written into the deployed monitor.
- Nifty history is a retrospective secondary-provider series, not independently immutable as-published exchange records.
- Direction accuracy is not a guaranteed executable return or investment recommendation.
