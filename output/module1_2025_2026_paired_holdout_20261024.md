# Module 1: 2025–2026 paired comparison

Training: 2021–2024 only. Combined training: 170 IPOs; deployed training: 252.
Test: 155 identical IPOs out of 179 exchange-verified later-year IPOs. No ID, company-name or symbol overlap.
Latest training outcome: 2024-12-30; first test decision: 2025-01-01.

The combined model and settings were selected on earlier development years and fixed before this evaluation. Positive means opening above issue price; flat counts as nonpositive. Direction cutoff is 50%; this is different from the deployed APPLY cutoff.

| Model | Correct / IPOs | Accuracy | Balanced accuracy | AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Current deployed model, same interim bids | 87/155 | 56.1% | 65.7% | 0.7665 | 0.248 |
| Bid-only model retrained on interim bids | 111/155 | 71.6% | 56.2% | 0.768 | 0.188 |
| GMP-only: positive GMP predicts positive opening | 106/155 | 68.4% | 51.6% | 0.5165 | 0.3161 |
| GMP-only fitted logistic control | 106/155 | 68.4% | 50.0% | 0.8004 | 0.1795 |
| New combined model | 116/155 | 74.8% | 61.9% | 0.8057 | 0.169 |
| Current model on later final bids (diagnostic only) | 119/155 | 76.8% | 71.5% | 0.8329 | 0.155 |

## Combined accuracy change on the same IPOs

- Versus Current deployed model, same interim bids: +18.71 percentage points; paired bootstrap 95% interval [7.096774193548387, 30.32258064516129]. Corrected 61 errors and introduced 32.
- Versus GMP-only: positive GMP predicts positive opening: +6.45 percentage points; paired bootstrap 95% interval [1.2903225806451613, 11.612903225806452]. Corrected 14 errors and introduced 4.
- Versus GMP-only fitted logistic control: +6.45 percentage points; paired bootstrap 95% interval [1.2903225806451613, 11.612903225806452]. Corrected 13 errors and introduced 3.

## By year

| Year | IPOs | Current, interim | GMP sign | Fitted GMP | Combined |
|---|---:|---:|---:|---:|---:|
| 2025 | 86 | 54.7% | 69.8% | 69.8% | 70.9% |
| 2026 | 69 | 58.0% | 66.7% | 66.7% | 79.7% |

## What changed and what still fails

The current final-book model receives much smaller bid counts at the interim stage and misses many positive openings. Retraining a bid-only control at the correct stage raises accuracy from 56.1% to 71.6%; adding GMP and offer descriptors raises it to 74.8%. The additional gain over that stage-matched control is 3.23 points, with a paired interval spanning zero.
The combined model still misses 3 positive openings and incorrectly predicts positive for 36 flat/negative openings. It catches only 26.5% of nonpositive listings. Its AUC is 0.8057, versus 0.8004 for fitted GMP. This is not evidence that ranking or return forecasting has improved.
All 39 wrong combined calls are saved in `module1_2025_2026_paired_holdout_20261024_errors.csv`.

## Coverage

- paired_eligible: 155
- no_verified_daywise_book: 15
- missing_verified_offer_dates: 5
- interim_book_missing_prior_gmp: 4

## Interpretation limits

- Retrospective reconstruction; not forecasts originally saved before applications closed.
- Later-year outcomes were previously inspected. No model or threshold selection used this new comparison.
- Only the matched coverage subset supports this comparison; excluded IPOs are not assumed correct.
- Public historical daily tables can be revised; GMP must precede the interim decision date.
- Offer size is a canonical historical offer descriptor, not a timestamp-certified capture.
- Current-model final-book diagnostic uses later information and is not an actionable pre-close forecast.
- Accuracy is direction at a fixed 0.5 cutoff, not APPLY precision or guaranteed investment profit.

Per-IPO probabilities, actual exchange opens and correct/wrong flags: `module1_2025_2026_paired_holdout_20261024_predictions.csv`.
The existing live model and saved forecasts were not changed.
