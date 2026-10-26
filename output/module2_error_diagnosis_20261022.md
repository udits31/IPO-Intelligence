# Why Module 2 is not improving

No new model, thresholds, PDF extraction or data acquisition. Same primary evaluation.

| Rule branch | Earlier n | Earlier accuracy | Later n | Later accuracy |
|---|---:|---:|---:|---:|
| loss_stop | 32 | 56.2% | 23 | 65.2% |
| relative_lag_exit | 6 | 66.7% | 6 | 50.0% |
| trailing_exit | 79 | 51.9% | 32 | 56.2% |
| hold | 97 | 54.6% | 71 | 42.3% |

Later errors: 25 sell calls followed by a rise, and 41 hold calls followed by a flat/down outcome.
Only 18 of 66 errors have an absolute subsequent move at most 5%; 26 have moves above 10%.

These are explanatory outcome diagnostics, not selectable trading features or new success criteria.

- Descriptive after-outcome audit, not a new forecast or blind validation.
- Reason-group rates are not probabilities for new IPOs; small groups are unstable.
- No alternate horizon, flat-price exclusion, abstention or cohort filtering improves the quoted score.
