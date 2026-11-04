# OpenBell progressive-engine experiment — 26 October 2026

No candidate replaced the frozen model. All original saved prediction rows and protected model hashes are unchanged.

## Module 1: same five experts, fixed 50% direction rule

Training: 290 IPOs from 2021–2025. Tests: the same 69 2026 IPOs and 23 recent verified openings. No issuer overlaps training and test. No post-hoc calibration was added.

| Variant | Correct / 69 | Wrong positive / 23 losers | Correct / 23 recent | Pass all gates? |
|---|---:|---:|---:|---|
| V0_FROZEN | 56 | 5 | 15 | No |
| V1_TIME_DECAY | 55 | 4 | 15 | No |
| V2_FAST_REGIME_GMP | 53 | 14 | 16 | No |
| V3_QUOTA_MIN_CAT | 55 | 5 | 16 | No |
| V4_OFS_CONVICTION | 59 | 3 | 14 | No |
| V5_CHAMPION | 56 | 5 | 15 | No |

The OFS/conviction candidate reached 85.5% on the older cohort but worsened the recent cohort. It was rejected. None of the four isolated components passed all requested gates, so V5 retains V0. Combining rejected components would violate the requested selection rule.

## Coverage before fitting

| Feature | Training 290 | 2026 test 69 | Recent 23 |
|---|---:|---:|---:|
| ofs_ratio | 45 | 5 | 6 |
| qib_quota_pct | 194 | 69 | 22 |
| nii_quota_pct | 178 | 69 | 22 |
| retail_quota_pct | 207 | 69 | 22 |
| anchor_domestic_mf_pct | 0 | 8 | 0 |
| lot_size_shares | 216 | 69 | 22 |
| quota_weighted_book_mult | 107 | 41 | 1 |
| log_min_category_mult | 290 | 69 | 23 |
| day2_to_cutoff_conviction | 161 | 69 | 1 |
| rolling_10_hit_rate | 289 | 69 | 23 |
| rolling_10_gmp_slippage | 289 | 69 | 23 |

Static terms are admitted under an explicit presumed-original-offer assumption. Fetch time is retained separately; the assumed opening-day availability is not presented as independently verified publication. Dated total-demand rows must match the frozen category readings. Missing/conflicting totals remain unavailable.

## Module 2: same 132 matured IPOs

The 31st-close checkpoint, 31→61 direction label and next-observed-close execution remain unchanged. Archived delivery uses only earlier sessions; filings use actual broadcast timestamps. Allotment calendars recovered later are explicitly assumed historical calendars.

| Policy | Correct / 132 | Accuracy | Winner recall | Loser recall | Governance return |
|---|---:|---:|---:|---:|
| Existing monitor | 62 | 46.97% | 80.00% | 23.38% | -0.510% |
| Always sell | 77 | 58.33% | 0.00% | 100.00% | -0.300% |
| Prove-it rules | 70 | 53.03% | 23.64% | 74.03% | -0.460% |
| Verified-clock sensitivity | 77 | 58.33% | 0.00% | 100.00% | -0.300% |

Applying the same 0.30% exit cost to both immediate exits and eventual held exits gives: prove-it -0.535%, existing monitor -0.744%, always-sell -0.300%. The original governance return is also retained above for exact comparison.

The prove-it policy does not beat always-sell. With strictly verified historical clocks, required evidence gaps make every call defensive; that is the always-sell baseline, not a predictive edge.

## Capital scenarios and live tracking

The capital calculator uses actual price × lot size and category thresholds. The proposed 6.5% loss, 0.35% friction, Rs9 fee and seven financing days are assumptions. Reciprocal subscription odds are labeled proxies. A separate full-allotment-loss stress can reject funded HNI exposure even when the retail scenario looks attractive. No orders are placed.

The backend regime tracker updates every 15 minutes. The next quarterly review is recorded, but no automatic model refit or promotion is enabled after all four components failed the gate. A future expanding-window refit must reserve a new future test cohort.

## Listing-day timing correction

A first-15-minute price-floor strategy cannot be evaluated at 09:45 during IPO special pre-open. NSE order entry ends by 09:45, followed by matching and a buffer to 10:00. The guarded implementation uses the first 15 completed continuous minutes, normally 10:00–10:15. Historical intraday bars are unavailable, so no intraday accuracy or fill is claimed. [NSE session rules](https://www.nseindia.com/static/products-services/equity-market-special-pre-open-session).

Targeted checks passed: 38. Original ledger/model integrity preserved: True.

These rules and ablations were proposed after the later outcomes had been inspected. They are exploratory comparisons; a gate pass would still need prospective confirmation.
