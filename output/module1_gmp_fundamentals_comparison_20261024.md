# Module 1: combining dated GMP, interim bids and company facts

Fixed logistic C=0.5, fixed cutoff=0.5; paired training/test identities for all variants; earlier-year matured labels only.

All models use the same 39 2024 test IPOs and the same eligible earlier-year training IPOs. All 39 GMP quotes are positive: GMP-sign is therefore identical to always predicting positive.

| Inputs | Correct | Accuracy | Balanced accuracy | AUC | Brier (lower is better) |
|---|---:|---:|---:|---:|---:|
| bids | 34/39 | 87.2% | 50.0% | 0.753 | 0.1057 |
| bids_offer_terms | 34/39 | 87.2% | 50.0% | 0.776 | 0.1062 |
| bids_all_facts | 32/39 | 82.1% | 64.1% | 0.771 | 0.1432 |
| gmp_logistic | 34/39 | 87.2% | 50.0% | 0.685 | 0.1082 |
| bids_gmp | 34/39 | 87.2% | 50.0% | 0.753 | 0.1061 |
| bids_offer_terms_gmp | 33/39 | 84.6% | 48.5% | 0.771 | 0.1077 |
| bids_all_facts_gmp | 32/39 | 82.1% | 64.1% | 0.759 | 0.1479 |
| gmp_sign | 34/39 | 87.2% | 50.0% | 0.500 | 0.1282 |
| always_positive | 34/39 | 87.2% | 50.0% | 0.500 | 0.1282 |

## Development-selected candidate

Selected using 2022/2023 probability error only: **bids_all_facts_gmp**. Against GMP-sign on 2024: -5.13 percentage points (paired bootstrap 95% interval -17.95 to +7.69); 2 GMP mistakes corrected, 4 new mistakes introduced.

## Why the previous model lost to GMP-sign

The quote-covered test sample has many more positive IPOs than flat/discount ones. A rule that never warns scores highly. Those previous models used a larger non-GMP training pool; the table above refits every candidate on identical paired training IPOs.

- Previous stage_plus_offer_terms: caught 0 GMP mistakes but missed 1 positive IPOs.
- Previous stage_plus_dated_fundamentals: caught 2 GMP mistakes but missed 5 positive IPOs.

A systematic extraction defect was also corrected: an undisclosed OFS amount could incorrectly pick up the following employee-reservation amount. All cited offer pages were checked with the same fix; contradictory amounts/share counts are withheld, not guessed. Annual EPS tables were reused, not re-extracted. This correction was prompted by inspected test errors; these results remain exploratory.

## GMP-sign mistakes

| IPO | Dated GMP | Actual listing return | Selected model probability |
|---|---:|---:|---:|
| Jana Small Finance Bank Ltd. | 16.9% | -4.35% | 0.8% |
| Capital Small Finance Bank Ltd. | 10.7% | -8.07% | 32.5% |
| R K SWAMY Ltd. | 31.2% | -13.19% | 95.1% |
| Gopal Snacks Ltd. | 17.5% | -12.47% | 67.2% |
| Baazar Style Retail Ltd. | 32.1% | +0.00% | 96.9% |

## Where the combined model goes wrong

The selected fit has 63 training IPOs, including only 12 flat/negative openings, for 14 transformed inputs. Its feature effects are learned associations, not economic laws. The following are positive IPOs it incorrectly warned against; negative contributions lower its predicted probability. They exactly reproduce the saved model, not a guessed explanation.

| Missed positive IPO | Probability | Two strongest negative contributions (log-odds) |
|---|---:|---|
| Rashi Peripherals Ltd. | 6.7% | earnings_yield_pct: -4.65; missingindicator_ofs_ratio: -0.22 |
| Mukka Proteins Ltd. | 5.0% | earnings_yield_pct: -5.52; log_issue_size: -0.43 |
| Ceigall India Ltd. | 47.3% | earnings_yield_pct: -1.37; log_qib: -0.74 |
| Sanathan Textiles Ltd. | 18.3% | earnings_yield_pct: -2.05; log_qib: -0.62 |

## Scope

Dated GMP exists for 102 of 186 early-book records and 39 of 75 2024 test IPOs. Uncovered IPOs are listed explicitly in the JSON; the 39-case score is not a score for all 75 IPOs. No live model, paper metric or saved prediction was replaced. The errors CSV includes every paired test IPO.

- All 2024 outcomes have been inspected before; this is retrospective, not new blind or prospective validation.
- Only IPOs with a genuine preceding-date GMP are in this paired experiment. No missing GMP is replaced with zero.
- Daily source dates are not verified intraday publication clocks; later historical revisions remain possible.
- GMP-sign is a hard direction baseline; its 0/1 Brier score is not a calibrated market probability.
- Direction accuracy does not establish return ranking, allotment-adjusted profit or a tradable edge.
