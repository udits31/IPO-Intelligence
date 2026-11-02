# Why Bonfiglioli missed, six-IPO comparison, and training through 2025

## Frozen six-IPO calls

| IPO | QIB | Retail | HNI | Saved positive probability | Actual opening return | Direction correct? |
|---|---:|---:|---:|---:|---:|---|
| Zetwerk Manufacturing Business | 4.7100x | 0.68x | 1.43x | 45.16% | +5.46% | No |
| Fusion CX | 0.0019x | 1.55x | 0.46x | 26.07% | +5.54% | No |
| Sahajanand Medical Technologies | 34.6961x | 3.03x | 14.64x | 79.44% | +7.89% | Yes |
| HD Fire Protect | 1.4918x | 11.31x | 48.66x | 60.92% | +23.62% | Yes |
| Bonfiglioli Transmissions | 120.7840x | 5.91x | 40.40x | 91.21% | +0.00% | No |
| Tonbo Imaging India | 293.5569x | 31.07x | 170.94x | 97.26% | +65.71% | Yes |

## Bonfiglioli: exact score, not a guess about its price move

The archived bid response hash and saved inputs match. The exact saved probability replays to 91.21%.
Linear score = 1.491374 intercept +0.953446 QIB -0.161582 Retail +0.055936 HNI.
The intercept is the model score at mean transformed training features, not the unconditional positive-listing rate.
All three bid coefficients are positive. Higher demand mechanically increases the probability; it does not measure whether the issue price is attractive.
GMP, valuation, profitability, OFS/fresh-capital allocation, issue size and market conditions are NOT inputs to this deployed classifier.
A 91% positive probability is not a 91% expected gain. One miss does not prove miscalibration, but the miss must count.

## Timing problems across the six

HD Fire and Fusion use 14 October interim snapshots, although their offers closed later. The six-call comparison mixes interim and post-close book stages. The final-book model is not validated on interim observations. Do not rewrite those saved calls with later numbers.
The other four captures are post-close. Bonfiglioli and Tonbo both have very strong subscriptions, yet opened flat and +65.71% respectively. A blanket penalty based solely on strong bids or OFS would not resolve this distinction.
Current webpage GMP or other facts retrieved after listing are not substituted as decision-time evidence.

## Calibration diagnostic on 179 later historical IPOs

| Score band | IPOs | Mean predicted positive | Actually positive |
|---|---:|---:|---:|
| 0–50% | 54 | 38.3% | 35.2% |
| 50–80% | 35 | 65.8% | 57.1% |
| 80–90% | 29 | 86.4% | 82.8% |
| 90–100% | 61 | 94.0% | 93.4% |

## Teacher-requested test: add 2025, evaluate only 2026

No model settings or thresholds were selected on 2026. Same 75 exchange-verified 2026 outcomes in every row.

| Training label source | Trained through | Training cases | 2026 direction accuracy |
|---|---:|---:|---:|
| canonical | 2024 | 252 | 80.0% |
| canonical | 2025 | 356 | 78.7% |
| exchange_verified | 2024 | 252 | 77.3% |
| exchange_verified | 2025 | 356 | 78.7% |

Four newly added 2025 trust instruments are excluded because this is a corporate IPO model. Six 2026 records lack verified opens (including trust instruments) and are listed explicitly in the JSON coverage report.
The 75-case 2026 score must not replace or be confused with the 179-case 2025–2026 score.

## Fix direction

1. Separate interim-book forecasts from final-book forecasts, and validate each stage using dated observations.
2. Select probability calibration using chronological earlier-fold predictions, not these six answers.
3. Add valuation, earnings and offer-structure features only when their pre-decision publication evidence is available. Test incremental benefit against bids-only and GMP-only on identical IPOs.
4. Keep all misses, retain immutable saved probabilities, and require paired evidence before promoting a new model.

No live bundle, saved forecast, manuscript or performance headline was changed.

- Six cases are too few to establish calibration or select new parameters.
- All six saved forecasts were created after bidding; this is not live pre-close accuracy.
- Final-book training does not validate interim subscription probabilities.
- 2026 outcomes have been inspected before: this is retrospective research, not untouched prospective validation.
- Missing features are plausible omissions, not proven causes of an individual price move.
