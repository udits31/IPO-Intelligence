# OpenBell — frozen model and two-layer application screen

Module 1 uses the original frozen five-expert artifact with recovered inputs. Layer 2 is a separate deterministic application policy; its verdict never overwrites the raw direction call. Module 2 uses R1 price rules plus eligible adverse exchange filings. R2–R4 remain diagnostics.

| Cohort | Raw correct | Raw false positives | Tier 1 | Tier 2 | Skip | APPLY precision | Losers skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| test_69 | 56/69 (81.2%) | 5 | 3 | 15 | 51 | 100.0% | 100.0% |
| recent_23 | 16/23 (69.6%) | 5 | 8 | 6 | 9 | 85.7% | 66.7% |
| recent_24_verified | 17/24 (70.8%) | 5 | 8 | 6 | 10 | 85.7% | 66.7% |

## Same-cohort direction baselines

| Cohort | Frozen model | Always positive | GMP sign | Frozen GMP-only classifier |
|---|---:|---:|---:|---:|
| test_69 | 56/69 | 46/69 | 46/69 | 55/69 |
| recent_23 | 16/23 | 17/23 | 15/23 | 13/23 |
| recent_24_verified | 17/24 | 18/24 | 16/24 | 13/24 |

## Rupee scenarios, not realised portfolio returns

- test_69: raw-positive selection Rs23,537.83; filtered selection Rs15,014.49. Original lot costs, reciprocal allotment, 0.35% friction and Rs9 per application assumed.
- recent_23: raw-positive selection Rs3,099.46; filtered selection Rs2,111.76. Original lot costs, reciprocal allotment, 0.35% friction and Rs9 per application assumed.
- recent_24_verified: raw-positive selection Rs3,160.48; filtered selection Rs2,111.76. Original lot costs, reciprocal allotment, 0.35% friction and Rs9 per application assumed.

## Module 2

| Policy | Correct /132 | Winners retained /55 | Losers exited /77 | Youden J | Mean net return |
|---|---:|---:|---:|---:|---:|
| M2_R0_OLD_BASELINE | 62 | 44 | 18 | +0.034 | -0.510% |
| M2_R1_WITH_DATED_FILINGS | 63 | 44 | 19 | +0.047 | -0.362% |
| M2_R2_FLOOR_AND_SMA20_GATE | 70 | 15 | 55 | -0.013 | -0.522% |
| M2_R3_PLUS_RELATIVE_STRENGTH | 69 | 14 | 55 | -0.031 | -0.548% |
| M2_R4_FULL_5_GATE_PROVE_IT | 70 | 13 | 57 | -0.023 | -0.460% |
| ALWAYS_SELL | 77 | 0 | 77 | +0.000 | -0.300% |

## Data and verification

- Eligible recovered Day-2 category coverage: 287/290 training, 69/69 earlier 2026, 24/24 recent.
- Gap-by-gap dispositions and source hashes: final_day2_gap_disposition_20261026.csv. A table published after the original decision is not backfilled into that decision.
- NSE/BSE outcome: verified_primary_bse. The original 16/23 comparison stays locked; the expanded exchange-verified result is 17/24 (70.8%).
- The stage bridge remains unfitted: 161 matched totals did not constitute 161 distinct Day2/preclose pairs.
- Existing model artifacts and forecast rows remain intact. R0 source is archived under its original hash; R1 is separately versioned.
- Pytest results are recorded in final_lock_pytest_run_20261026.log. The default suite now targets automated tests; exploratory scraping scripts are not collected and the network smoke test is opt-in.

## Allotment and financing corrections

- Mainboard allocation rules and minimum-plus-residual NII allotment verified against SEBI ICDR, consolidated 21 March 2026, regulation 32 and Schedule XIII. Source: https://www.sebi.gov.in/sebi_data/attachdocs/mar-2026/1774592300989.pdf
- Minimum sNII/bNII application lots strictly exceed Rs2 lakh/Rs10 lakh: floor(threshold/lot_cost)+1, including exact-boundary cases.
- A 35% retail quota does not trigger the >=50% retail-heavy flag. Share multiples alone do not prove valid-application lottery odds or full allotment.
- Financed interest is charged on the actual borrowed principal for every application. The 9.5% / five-day setting is a scenario, not a verified lender quote. Dynamic parcel ratio is used instead of hard-coding 5.
- Missing dated bNII demand means financing viability is false/unestablished, never inferred from total NII.
- No prospective profitability or zero-overfitting claim is made. Full assumptions and per-IPO decisions are in the JSON.
