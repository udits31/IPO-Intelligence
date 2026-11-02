# Module 1: two separate scorecards

| Cohort | Model right | Model missed (FP / FN) | Apply winners / selected | Losers admitted | Losers avoided | Winners skipped |
|---|---:|---:|---:|---:|---:|---:|
| test_69 | 56/69 | 13 (5 / 8) | 18/18 | 0 | 23/23 | 28/46 |
| recent_23 | 16/23 | 7 (5 / 2) | 12/14 | 2 | 4/6 | 5/17 |
| recent_24_verified | 17/24 | 7 (5 / 2) | 12/14 | 2 | 4/6 | 6/18 |

## Rupee scenarios per application

| Cohort | Raw positive calls | Total raw scenario | Average raw | Apply calls | Total filtered scenario | Average filtered |
|---|---:|---:|---:|---:|---:|---:|
| test_69 | 43 | Rs23,537.83 | Rs547.39 | 18 | Rs15,014.49 | Rs834.14 |
| recent_23 | 20 | Rs3,099.46 | Rs154.97 | 14 | Rs2,111.76 | Rs150.84 |
| recent_24_verified | 21 | Rs3,160.48 | Rs150.50 | 14 | Rs2,111.76 | Rs150.84 |

The 20 and 21 raw-positive denominators come from TP+FP in the frozen records. The supplied 19/20 denominators were inconsistent with those confusion matrices. These are assumed-allotment scenarios, not realised profits.

Every per-IPO JSON record reports Model Call, Model Outcome, Application Screen and Screen Result independently. Skipped winners are opportunity cost; admitted losers remain visible. No locked benchmark, model, capital rule or forecast row was changed.
