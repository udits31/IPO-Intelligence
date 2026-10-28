# Overall-accuracy and forward-year update protocol

Keep the original 155 IPO IDs, actual exchange opening labels, original interim bids and GMP observations. Positive means exchange opening strictly above issue price; flat is nonpositive. Do not drop hard cases or replace predictions with their outcomes.

Report three distinct experiments:
1. Strict fixed fit on 170 eligible 2021–2024 IPOs, score all original 155 IPOs from 2025–2026.
2. Updated fit on 256 eligible 2021–2025 IPOs, score the original 69 IPOs from 2026 only. The 86 IPOs used for 2025 training are not test observations in this result.
3. A chronological 155-IPO annual replay: each 2025 score uses the strict model; each 2026 score uses the updated model. This is a changing annual model, not the fixed-2024 experiment.

Feature sets: original six bid/offer/GMP fields; those plus bid/GMP trajectory; base plus dated market context; trajectory plus market context; base plus financial fields and market context; trajectory plus financial fields and market context. Financial fields: public IPO-context annual profit margin, PAT/assets, revenue growth, loss indicator, financial-sector flag, pre-IPO EPS/offer-cap yield and log IPO price/book. Fiscal periods must precede decisions, but original publication times remain uncertified. Never extract current stock quotations or listing results as features.

Market features use Nifty five-/20-session returns, 20-session realised daily volatility, India VIX and VIX five-observation change. Select the last historical close strictly before the decision date, require no more than five calendar days' staleness, and retain missing rather than insert a neutral value. Use frozen existing observations through 2024 and new downloaded closes only from 2025 onward. Archive source URLs and hashes. Trajectories remain pre-cutoff and same-source.

Use the previous 24 predeclared logistic/shallow boosting/forest configurations, plus XGBoost depth 1/2, 120 trees, learning rate .04, min_child_weight 5, lambda 10, subsample/column sample .8, seed 42, no test early stopping. Fix the candidate set before scoring. Threshold grid remains .35–.85 in .025 steps.

Use expanding chronological validation, with training outcomes strictly before the first validation decision. For the strict model select using the latest two available validation years 2023/2024; for the through-2025 model select using 2024/2025. Primary objective is overall accuracy, then balanced accuracy, Brier, fewer features and simpler family, then cutoff nearest .5. No required negative recall: missed gains count equally with missed losses. All preprocessing is fitted inside each earlier fold. Save selections and fit memberships before scoring 2026. Do not select a later-year winner or enlarge the grid based on its score.

2025–2026 outcomes have already been inspected during debugging; this remains exploratory retrospective evidence. No 2026 labels may enter fit, calibration or selection. More than 80% is a target, not permission to tune to the test set or relabel 83% positive-call precision as overall accuracy. Do not change live bundles, old ledgers, Module 2, paper or Git remotes automatically.
