# OpenBell / Capstone: complete engineering and research context

Snapshot: 25 October 2026, India Standard Time. Prepared from Git history, local source code, saved model artifacts, evaluation reports, SQLite records and running API responses. This is a factual handoff for further research, not a claim that the system has demonstrated profitable trading.

## 1. Purpose and intended workflow

OpenBell is an Indian IPO research and post-listing monitoring application. It has two distinct modules.

- **Module 1:** help compare IPOs while applications are still open, estimate listing direction and potentially listing return, and rank opportunities. Predictions must use information available by the decision time and must stop being updated for application decisions at the configured closing cutoff.
- **Module 2:** monitor a listed holding continuously using dated prices, market/sector movement, corporate events, disclosures and news. Explain hold, review, reduce or accumulate indications.

The intended product is useful to ordinary retail investors, rather than a prettier collection of scraped figures. Whether the model actually adds value beyond simple GMP or demand rules is an empirical question and remains unresolved.

There is no single valid “overall application accuracy.” Module 1 direction, return magnitude, opportunity ranking, and Module 2 future holding outcomes require separate measurements.

## 2. Repository and pushed-versus-local status

Repository: https://github.com/Titan-M/Capstone

Workspace: `C:/Projects/Capstone`.

Verified against the remote main branch during this handoff:

- Local HEAD, local `origin/main`, and the remote `refs/heads/main` all point to `fb26c4b895bb0d8dad26de725cd468922087d8a0`.
- Latest pushed commit is dated **18 October 2026**.
- Much of the important work described below is **local and uncommitted/untracked**, including the latest prospective stack, historical input recovery, backtest comparison and UI cleanup.
- The user explicitly requested no further push without asking. This handoff does not commit or push anything.
- A fresh clone of GitHub will therefore not reproduce the current local application without transferring the missing local changes, data and artifacts.

### Pushed development timeline

| Date | Commit | Main contribution recorded in Git |
|---|---|---|
| 28 Aug 2026 | `d219b70` | Initial scraper utilities/data collection and frontend dependencies |
| 28 Aug 2026 | `c5162db` | IPO analytics engine, original hybrid ML, news sentiment and evaluation API |
| 29 Aug 2026 | `f9b38e1` | FastAPI dual-module API and AI copilot integration |
| 29 Aug 2026 | `53d4a09` | Dashboard, backend service and monitoring engine |
| 23 Sep 2026 | `96798c5` | Dynamic census dataset, modern IPO surveillance and ticker resolution |
| 23 Sep 2026 | `7fdbcf9` | Multiple GMP sources, historical archive mapping and dual GMP metrics |
| 23 Sep 2026 | `00cee7d` | Reconciled contradictory verdict/explanation narratives |
| 23 Sep 2026 | `e4cf01a` | Replaced internal database-ID badges with useful issue-status labels |
| 24 Sep 2026 | `2f68e2e` | Attribution/monitoring engines, GMP service, API and UI development |
| 25 Sep 2026 | `66cb1d6` | Trendlyne integration, monitoring improvements and tests |
| 18 Oct 2026 | `fb26c4b` | Provenance safeguards, validated model bundle, subscription capture, NSE integration, monitoring evaluation, tests and conference revision sources |

The first visible commit is not proof that the project had no earlier planning or work outside this repository.

## 3. Application architecture

- Frontend: Next.js 14 / React 18, JavaScript/JSX, CSS/Tailwind dependencies and Lucide icons.
- Backend: Python/FastAPI/Uvicorn.
- Data work: pandas, NumPy, scikit-learn, XGBoost, joblib, BeautifulSoup, HTTP clients and yfinance.
- Local persistence: CSV/JSON datasets, compressed source archives, model files and SQLite telemetry at `data/telemetry/snapshots.db`.
- Supabase and Google AI dependencies/integrations exist; their presence does not establish a deployed managed production system.
- Important code areas: `api/`, `core/`, `services/`, `scrapers/`, `ml/`, `frontend/`, `tests/` and `output/`.
- Local frontend is served on port 3000; backend on port 8000. During verification, the API reported healthy, version 5.1.0. The API also reported both subscription and Module 2 collectors running.
- Background Python threads depend on the backend process being alive. They are not evidence of an independently managed, always-on cloud scheduler.

Features include an IPO opportunity board, company/security search, Module 1 analysis, watchlists, performance comparisons, portfolio/holding monitoring, market context, charts and an AI research assistant. The assistant is not the numerical classifier and should not invent model scores.

## 4. Data universe and cohort definitions

The local canonical dataset contains **689 IPO records from 2010 through 2026**. This universe is not the training set of every model.

`data/subscription_observations.csv` also has 689 rows in this snapshot. A final historical subscription row is not equivalent to a time-stamped live bid snapshot.

### Distinct cohorts that must not be mixed

1. **Legacy final-book training:** 254 canonical listings in 2021–2024, excluding two non-corporate investment trusts, leaving **252 corporate IPOs**.
2. **Early-book research:** a day-wise research file has 196 source rows; the return-model audit retained 186 eligible corporate rows from 2021–2024, with 102 paired GMP-supported rows in that original experiment.
3. **Expanded feature frame:** 359 rows: 42 from 2021, 28 from 2022, 41 from 2023, 75 from 2024, 104 from 2025 and 69 from 2026.
4. **Current frozen stack training:** **290 IPOs from 2021–2025**; the 69 2026 rows are outside this training-year set.
5. **Strict earlier-only comparison:** a separate 2021–2024-trained research result uses **155 IPOs from 2025–2026**. It must not be confused with the 69-case 2026-only result.
6. **Most recent listing replay:** **24 locally listed records**, with recovered closing-day bid evidence; **23 independently verified listing opens**. NSE's opening is still unverified in this evaluation pipeline and excluded from accuracy.
7. **Module 2 current price-rule evaluation:** **132 matured IPOs from 2025–2026**, selected from 175 available cases; 43 lack enough future sessions.

There are different ID systems: canonical numerical issuer IDs and scraped/source identifiers. Matching must use verified company/security identity rather than assuming an ID number means the same thing across sources.

## 5. Sources and provenance

Sources integrated or used in recovery include NSE/BSE issue information and price archives; SEBI/RHP/DRHP offer documents; Chittorgarh and InvestorGain; IPOJI; The IPO List; Groww; IPORise; IPOmetrics; 5paisa; Choice; financial/news sites; Yahoo Finance; and supplementary Trendlyne context.

Authority differs by field. Exchange opening prices have priority for listing results. GMP is unofficial grey-market information, not an exchange quote. Chittorgarh and InvestorGain belong to the same source family and do not provide independent confirmation merely by agreeing.

For important observations the system retains issuer identity, observation time where supplied, retrieval time, source URL, compressed raw response and SHA-256 hash. A hash establishes integrity of a saved file, not the truth of its contents or that it was public at an earlier date.

The system distinguishes missing, zero, stale, interim, final, post-cutoff and recovered-later data. A missing QIB figure must not become 0x. After a failed refresh, the last genuine reading can remain visible with its original time.

## 6. Module 1 model evolution

### A. Original hybrid design

The original design used engineered subscription/market/offer features and logistic regression, XGBoost and histogram gradient boosting, with classification and return-regression components. Historical-analog comparisons and heuristic narrative/decision layers were also present. Some histogram models were fitted at application initialization in the early implementation.

An original model's impressive-looking headline is not automatically a reproducible current result. Experiments changed features, splits, folds, thresholds, data coverage and target handling over time. A larger fold count does not itself establish better generalization. Any historical “nine folds” claim must be tied to its exact script/commit and cohort; it is not the current prospective protocol.

### B. Retained legacy final-book classifier

Artifact: `ml/model_bundle.json`.

- Version: `module1-corporate-subscription-lr-2021-2024-v2`.
- Logistic regression, standardized `log1p` QIB, retail and HNI multiples.
- Trained on 252 corporate IPOs from 2021–2024.
- Inputs are **final subscription books**, generally available too late to represent an investor's pre-application decision.
- Binary target: exchange listing open strictly greater than issue price; flat is nonpositive.
- Direction threshold: 0.5. Separate conservative application policy threshold: 0.82. These answer different questions.
- GMP, VIX and offer facts are not features of this particular three-input bundle.

It remains loaded by the legacy backend predictor. It has not disappeared simply because newer cards use another service.

### C. Return and prioritization research

`ml/return_pipeline.py`, `services/return_forecast_service.py`, and the `preclose_return.joblib` artifact support a separate experimental opening-return estimate and ranking path.

The reported training stage is penultimate-session end. Five core inputs are log QIB/HNI/retail, issue price and issue size; there are bids-only and bids-plus-GMP tracks. The original selected return family was boosted Huber regression. Estimated bands come from historical residuals; they are not automatically calibrated live confidence intervals.

Example of the limits: on 75 2024 confirmation rows, bids-only return MAE was about **26.85 percentage points**, and an intended 80% band covered about **54.7%**. On 39 paired rows, the bids-plus-GMP MAE was about **24.26 pp**, versus **24.96 pp** for GMP. This does not establish reliable trading superiority.

A research ranking rule uses central return minus half of downside below zero, minus a disclosed 0.5 pp cost scenario. It does not calculate actual allotment-adjusted expected profit.

### D. Current local frozen direction stack

The visible prospective/latest historical comparison uses:

`ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib`

Protocol: `output/prospective_module1_protocol.json`.

Model SHA-256: `bf07f15c86fce9984751b36cda7636f3c52875202492199a0017c8842406ff38`.

Training: 290 issuer IDs through 2025; standard positive-direction cutoff 0.5. The original artifact is explicitly research-only. It was subsequently wired into the local frozen forecast service; this is distinct from replacing the legacy JSON bundle or formally validating a public production release.

The five expert views are:

| View | Estimator | Main configuration |
|---|---|---|
| Base | Logistic regression | C=0.5; raw GMP representation |
| Base + macro | Logistic regression | C=0.05; balanced classes; signed-log GMP |
| Financial + macro | Logistic regression | C=0.5; signed-log GMP |
| Trajectory | Random forest | Maximum depth 5; balanced classes |
| GMP only | Logistic regression | C=0.5 |

Each expert produces a probability. Probabilities are clipped, converted to log-odds and fed to a standardized logistic meta-model with C=0.5. Missing-value transformations/imputation are fitted from training data.

The union of input fields is **23 engineered features**, grouped as:

- Demand/offer: log QIB, log retail, log HNI, log offer cap price, log issue size, GMP percentage.
- Trajectory: log growth in QIB/HNI/retail, GMP quote age, GMP slope in percentage points per day.
- Macro: Nifty 5-session return, 20-session return, 20-session volatility, India VIX, 5-session VIX change.
- Financial: profit margin, return on assets, transformed revenue growth, annual-loss flag, financial-sector flag, IPO earnings yield and log IPO price/book.

Each view uses a subset. This is not an LLM prediction, a single XGBoost model, or a model using every number displayed in the UI.

### Current-model routing is fragmented

The API health response names the legacy three-input logistic predictor. The main newer forecast card is supplied separately by `services/prospective_module1.py`; listed-IPO replay uses `services/listed_backtest_service.py`. Experimental return/ranking and historical analog outputs are additional paths. Calling all of these simply “the model” has caused confusion and is an outstanding architecture/documentation problem.

## 7. Module 1 performance: keep the denominators

| Evaluation | Result | Interpretation |
|---|---|---|
| Legacy final-book model, exchange-verified 2025–2026 set | 136/179 = **76.0%**, AUC **0.8141** | Final-book retrospective direction; not pre-close live accuracy |
| Same legacy set, conservative application policy | **90.0% precision** among APPLY calls | Not 90% overall accuracy; overall policy classification is 73.2% |
| Strict 2021–2024-trained research comparison | 118/155 = **76.1%** | Separate earlier-only experiment |
| Updated stack, trained through 2025; 69 2026 cases | 56/69 = **81.2%**, AUC **0.879**, Brier **0.1516** | Retrospective research; standard-cutoff comparison added after other test failures were inspected |
| GMP control on those 69 | 55/69 = **79.7%** | Small one-case difference; not proof of sustained superiority |
| Latest recovered closing-day replay, frozen stack | 15/23 = **65.2%** | Current recent diagnostic cohort |
| Fitted GMP-only model on those same 23 | 13/23 = **56.5%** | Separate frozen GMP classifier, threshold 0.625 |
| Simple GMP > 0 rule on those same 23 | 15/23 = **65.2%** | Different baseline from the fitted GMP classifier |
| Always-positive baseline on those same 23 | 17/23 = **73.9%** | Beats the frozen combined model in this recent cohort |
| Latest simpler candidate on those 23 | 16/23 = **69.6%** | Not promoted |

The 69-case historical 81.2% must never be presented as the accuracy of the recent 23 cases, the whole application, an intraday forecast, or Module 2. AUC is discrimination, not a percentage of correct labels. Precision is not overall accuracy.

## 8. Latest recovery and diagnosis, 25 October

All 24 currently listed records now have recovered closing-day subscription readings timestamped before the configured 17:00 IST cutoff. All 24 also have dated GMP evidence. Annual financial panels were recovered for 23. Some sector/valuation fields remain missing; NSE's annual financial inputs and verified opening remain unresolved in this pipeline.

This work used public HTML tables and cached archives. It did not require OCR for every IPO document.

Two concrete extraction defects were fixed:

1. Multiple different GMP observations on one day were previously treated as conflicts and discarded. The new parser retains the times and chooses an eligible quote available by the decision reading.
2. Annual financial extraction assumed March year-ends. December year-ends, needed for Bonfiglioli, are now supported; June interim figures are not relabelled annual.

Bid trajectories, GMP age/slope, financial facts and preceding-session macro observations are connected to the frozen inference path. Quotes after the decision are excluded. A quote with only a date must precede the decision day; its date alone does not establish a pre-cutoff intraday time.

The initial incomplete recent replay predicted every case positive and got 17/23, matching always positive. Adding genuine inputs reduced its score to 15/23. It now calls 21 positive and two nonpositive; the two nonpositive calls, Nityas and Sonaselection, are false negatives.

All six original flat/negative misses remain positive calls:

| IPO | Recovered GMP % used | Combined positive score | Actual opening return |
|---|---:|---:|---:|
| Vishal Nirmiti | 9.09% | 50.8% | -2.27% |
| Acevector | 3.12% | 55.5% | -11.50% |
| Runwal Enterprises | 4.59% | 62.9% | 0.00% |
| Elevate Campuses | 0.69% | 59.8% | -1.91% |
| ArMee Infotech | 5.33% | 59.3% | 0.00% |
| Bonfiglioli Transmissions | 16.21% | 97.7% | 0.00% |

Interpretation: high demand/GMP is insufficient to distinguish all flat openings; missing-input imputation masked weak information; and the stack was studied at penultimate-day close but this replay uses closing-day readings. Missingness and stage shift are substantive problems. Changing the label to day-one high just to rescue errors would answer a different question.

A restricted candidate check used 2021–2024 fitting and 2025 Brier score selection, then refitted through 2025. Three families were considered: regularized logistic, less-regularized logistic and shallow histogram boosting. Regularized logistic was selected by older-year Brier score and reached 16/23 later. It was not substituted for the frozen artifact. Candidate design followed inspection of recent failures, so even that result is diagnostic, not a fresh untouched test.

## 9. Saved forecasts versus historical reconstruction

The audit found no genuine application-time saved probability for the 24 already-listed records. Six archived scores had been recorded before listing but after bidding had ended. They are not pre-close calls.

Recovered historical subscriptions can support a backtest calculated now. They cannot prove that the app issued the resulting probability earlier. Existing archived calls are preserved rather than overwritten or backdated.

The latest UI explicitly labels listed-company results **Historical analysis** and shows model estimate, actual opening and direction matched/missed. All-IPO comparison, source details, baselines, missing inputs and method explanations are expandable. Recovered GMP is also connected to the deal snapshot instead of incorrectly displaying unavailable.

The research target remains listing **open versus issue price**. Listing open, day-one high, closing price and achievable sale proceeds are different outcomes. The intraday high is a hindsight maximum and cannot be treated as an automatically obtainable sell price. An executable exit strategy needs a predeclared order/time/target/stop rule and realistic fills/costs. This handoff does not claim such an intraday strategy has been validated.

## 10. Prospective capture now running locally

`services/prospective_module1.py` freezes the model/protocol and appends original predictions, inputs, hashes, times and later independent outcomes to SQLite.

- Scheduled model cycles: roughly every 15 minutes, increasing to 5 minutes during 14:00–17:00 IST.
- Forecast issuance checks actual completion time against the offer cutoff. A run finishing after the cutoff cannot masquerade as a pre-close call.
- Pre-bidding estimates and bidding-period forecasts are distinguished.
- Closing a daily session is not the same as closing the IPO offer.
- Post-cutoff reconciliation and actual listing outcomes do not rewrite earlier inputs or probabilities.
- The freeze currently uses a 17:00 IST closing-day convention; broker/application and mandate deadlines can differ and need appropriate product treatment.

At inspection, five upcoming issuers had genuine new frozen estimates and none had a verified listing outcome. Prospective accuracy is therefore **not yet measurable**. The five are BLS Polymers, SFC Environmental Technologies, EAAA India Alternatives, Bombay Coated & Special Steels and M.K.Sons Fine Jewels. Some are pre-bidding estimates, not complete subscription forecasts.

Telemetry row counts at this snapshot: 424 subscription captures, 162 frozen forecast rows across five issuers, 326 historical subscription-evidence rows, and six older after-bidding/pre-listing archives. These counts will grow while collection runs; forecast rows are not unique evaluated IPO counts.

## 11. Module 2: what is actually implemented

The local live Module 2 is primarily a **deterministic risk-rule monitor**, not a validated trained 30-session return classifier.

It combines dated stock closes, issue-price-relative gain/loss, peak drawdown, 10/30-session and since-listing stock/sector/Nifty attribution, verified corporate-action adjustments, lock-in information, supplied selling disclosures and eligible news.

Examples of current fixed rules include a severe issue-price loss threshold around -12%; combinations of loss and sector-specific lag; stricter conditions for accumulation using verified unlock/disclosure coverage and relative strength; and trailing references around 8% or 25% for different branches. These percentages are configured heuristic rules, not numbers learned by the current Module 1 model.

The 8% pullback rule was changed from implying an automatic validated exit to a **review alert with no automatic exit support**. A trailing reference cannot guarantee execution at that level during gaps or illiquidity.

News is used, but currently through a versioned lexical/rule scorer with contextual treatment of profit/revenue changes, issuer filtering and date checks—not a fine-tuned neural sentiment predictor. A score at or below -0.30 can block an accumulation indication and cause an event-risk caution. Positive news adds context. Missing or stale news is not positive news.

Unverified autocomplete quotes do not drive the close-based action. Stale or unverified price/share-adjustment evidence pauses advice. Confirmed splits/bonuses are handled without applying adjustments twice; cash dividends are not included in the cited price-return tests.

Collectors refresh/reassess evidence while the backend runs: news-related checks are approximately hourly during weekday market hours and every two hours otherwise within configured daytime/evening hours; a separate close-based cohort capture is attempted after 17:30 with retries. This is not tick-by-tick trading or guaranteed live exchange coverage.

## 12. Module 2 performance and experiments

Current matching governance evaluation: `output/module2_holdout_accuracy_2025_2026_governance_20261023.json`.

The report's monitor source hash matches the local `core/monitor_engine.py` at this snapshot. It evaluates a **price-only historical subset**, not the entire live news/disclosure-enabled module.

Protocol: one decision at the 31st observed close, next-observed-close execution, future direction 30 sessions after decision, and strategy performance 30 sessions after execution. Assumed selling cost is 0.30%. Source prices are retrospectively retrieved/reconciled; there is no complete as-published historical news/filing input set for this test.

| Measurement on 132 matured IPOs | Current rule monitor | Relevant baseline |
|---|---:|---:|
| Next-session direction after the checkpoint | **49.24%** | Always hold 51.52%; 8% trailing baseline 53.03% |
| 30-session future direction | **46.97% (62/132)** | Always sell 58.33% (77/132); 8% trailing baseline 49.24% |
| Illustrative mean net strategy return | **-0.51%** | Always hold -1.134%; always sell -0.30% |

“First day” in this particular report means the first future session after the checkpoint, **not the IPO listing day**.

An older 128-case report showed 50.78% 30-session accuracy. Different source reconciliation, maturity coverage and rule governance produced the newer 132-case result. These are versioned evaluations, not interchangeable scores.

Research alternatives include price models, macro/risk governance, delivery information, dated news overlays, financial/earnings events, disclosure extraction, rule calibration and daily learning. A 19-feature extended-history shallow-forest experiment used 1,665 earlier daily training examples from 142 issuers and reached **79/132 = 59.85%**, versus always sell at 58.33%. Its development prices had secondary-source limitations and it remains **not promoted**. Repeated daily rows are not independent companies, and this small retrospective difference does not establish a stable tradable edge.

There is currently no established 70%+ live Module 2 accuracy.

## 13. Problems faced: engineering and data

| Problem | Work done | Remaining qualification |
|---|---|---|
| Missing subscription data rendered as 0x | Missing-value flags, preserved last reading, source/time labels | Provider failures and category availability still vary |
| Late or absent Chittorgarh readings | Multiple-source recovery and NSE bid feeds | Exchange bid position is not necessarily certified final subscription |
| Collector gaps when servers stopped | Startup refresh/catch-up, ledgers and scheduled collectors | Local machine/process availability still matters |
| Offer “closed” confused with session end | Offer-calendar checks and timestamped stages | Broker deadlines and market holidays need disciplined handling |
| Final books used as if investors could apply on them | Separate early/live/final/reconciliation tracks | Model stage alignment remains a central research problem |
| Source timestamps confused with fetch timestamps | Retain both clocks; reject future/post-cutoff observations | Recovered publisher times lack immutable historical certification |
| Companies, tickers or issue dates mismatched | Identity/date/cap-price matching, NSE/BSE resolution | One recent opening remains unverified |
| GMP sources missing, related or conflicting | Source-family tracking and timed historical parser | GMP remains unofficial; disagreements are preserved |
| Financial tables inaccessible or malformed | HTML/native-document extraction, targeted recovery, December year-end fix | Remaining sector/valuation facts are not fabricated |
| Excessive document/OCR work | Reuse archives and prefer structured HTML/native PDF text | OCR should be targeted, not applied indiscriminately |
| Stale prices/news shown as current | Refresh checks; retain timestamps; pause stale decisions | No full low-latency market-data infrastructure |
| Splits/bonuses corrupted returns | Explicit event reconciliation and share-basis checks | Unknown events pause analysis; dividends excluded from price-only returns |
| Arbitrary 8% rule presented too strongly | Explain rule and downgrade to review-only alert | Not validated as an optimal sell rule |
| Blank charts, dense cards and contradictory descriptions | UI iterations, price-context labels, cleaner main card and collapsed audit sections | Broader product usability still needs systematic review |
| Performance page failed after evidence changed | SHA-matched evidence gate instead of stale constants | Reports must be rebuilt for changed inputs/code |
| Research-paper title/content changed incorrectly | User required original title and preservation of substantive material | Paper is a separate artifact and is not automatically synchronized to every experiment |
| LaTeX preview/compiler trouble | Existing source preserved; document/PDF work kept in conference-revision output | An old PDF may not reflect the current model state |
| Large local worktree diverged from GitHub | Verified pushed versus local status in this handoff | Packaging/reproducibility and a controlled future commit are still needed |

## 14. Problems faced: ML and evaluation

- Small effective sample sizes compared with feature/model complexity.
- High positive-listing prevalence makes accuracy alone misleading.
- Three different numbers were repeatedly confused: probability, APPLY precision and accuracy.
- More complex models, more folds, larger datasets or more extracted facts do not guarantee improvement.
- Mixing final and interim bids creates a train/decision-time mismatch even if the issuer train/test split is correct.
- Training through 2025 requires treating 2025 cases as training/calibration, never as holdout for that model.
- Repeated inspection of the same misses creates test-set reuse and post-hoc selection risk.
- Missing-input imputation can produce extreme probabilities that are poorly calibrated for live missingness patterns.
- A direction classifier does not directly estimate gain magnitude, rank expected profit, or decide an exit time.
- The day's maximum high is not an executable hindsight-free strategy return.
- A saved-after-listing replay is not proof of a saved-before-close forecast.
- Module 2 future-session maturity, costs, corporate actions and absent historical news constrain its score.
- One better retrospective result is not evidence of persistent profit or superiority over a strong baseline.

## 15. Tests, reproducibility and current limits

Tests cover cutoff boundaries, rejection of post-cutoff data, preservation of genuine zero bids, issuer matching, UTC-to-IST conversion, intraday GMP parsing, annual versus interim periods, immutable prediction records and listed-opening comparisons. The latest targeted group passed **19 tests**, and the latest frontend production build passed. This is not a claim that the entire repository test suite was rerun for this handoff.

Historical source hashes, model hashes, training issuer IDs and cohort-level ledgers are available. API `/api/v1/performance/evidence` returned `verified_current_snapshot` when checked. That evidence endpoint principally describes the older final-book/monitor evaluation; it does not replace the separate current historical-stack replay.

Known unresolved work includes stage-matched pre-close modeling, weak recent discrimination, overconfidence, incomplete financial/sector facts, one unverified opening, prospective outcome maturity, full news-aware Module 2 evaluation, realistic execution/portfolio-allotment modeling, production hardening and consolidation of multiple model paths. The requested 80–85% recent Module 1 and 70% Module 2 performance has not been established.

The user reports that the conference paper has been selected and has asked to prioritize the application rather than keep changing the paper. Conference selection is not proof that every later model experiment is validated.

## 16. Key artifacts for a researcher

Paths below are relative to the repository root. A researcher receiving only this brief cannot access the local files unless they are separately supplied.

| Area | Files |
|---|---|
| Legacy model | `ml/model_bundle.json`, `ml/model_pipeline.py`, `ml/train_module1.py` |
| Frozen newer model | `output/prospective_module1_protocol.json`, `services/prospective_module1.py`, `ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib` |
| 69-case stack research | `output/module1_standard_classifier_audit_20261025.json`, `ml/research_module1_stacking.py` |
| Latest recovery | `ml/recover_listed_closing_books.py`, `ml/recover_listed_model_inputs.py`, `output/listed_recovered_model_inputs.json` |
| Recent comparison | `services/listed_backtest_service.py`, `output/listed_model_baseline_comparison.csv`, `output/listed_closing_day_backtest.json` |
| Diagnosis/candidate | `output/listed_model_input_diagnosis.md`, `ml/validate_listed_input_fix.py`, `output/listed_input_fix_candidate.json` |
| Saved-call audit | `services/saved_preclose_service.py`, `output/listed_subscription_recovery_audit.json`, `data/telemetry/snapshots.db` |
| Return/ranking research | `ml/return_pipeline.py`, `services/return_forecast_service.py`, `output/module1_return_benchmark.json` |
| Module 2 live policy | `core/monitor_engine.py`, `core/attribution_engine.py`, `services/news_service.py`, `services/module2_decision_ledger.py` |
| Module 2 evidence | `output/module2_holdout_accuracy_2025_2026_governance_20261023.json`, `output/module2_extended_history_learning_20261023.json` |
| Performance evidence API | `services/performance_service.py`, `output/performance_evidence.json` |
| Frontend | `frontend/src/app/page.jsx`, `frontend/src/app/PerformanceReview.jsx`, `frontend/src/app/PredictionResults.jsx` |
| Conference source | `output/conference_revision/revised_manuscript.tex` |

## 17. Research request to Gemini

Use this as an engineering/research audit, not a request to manufacture an attractive accuracy headline. Please:

1. Challenge the target definition: listing open, day-one achievable return and later holding outcomes are different targets.
2. Propose stage-matched, leakage-resistant features and training/evaluation procedures for application-time decisions.
3. Explain the six flat/negative misses and two false negatives without adjusting a threshold to memorize these 23 cases.
4. Compare combined ML against fitted GMP, simple GMP sign/threshold rules, QIB-only and class-prior baselines on identical inputs, timestamps and issuers.
5. Distinguish calibrated probability from an uncalibrated model score and address missingness patterns explicitly.
6. Evaluate return magnitude and ranking with MAE, ranking/selection utility and realistic allotment/cost assumptions, not direction accuracy alone.
7. Design an executable sell/hold evaluation using predeclared actions and time horizons rather than selling at the hindsight day high.
8. Explain what additional dated evidence would be necessary to validate a news-aware Module 2 model.
9. Propose a staged production architecture that consolidates model paths, automates dated capture and remains usable when feeds fail.
10. Preserve the separation between verified pushed work, current local implementation and unpromoted research. Ask for missing artifacts rather than assuming the brief proves more than it does.

## 18. Addendum: expanded all-known-2026 evaluation

After the main brief was prepared, all seven previously omitted 2026 corporate IPOs were recovered and tested without retraining. The canonical corporate cohort is now 76/76 evaluated: 60 correct (78.95%), versus fitted GMP-only 59/76 (77.63%) and always-positive 49/76 (64.47%). Five REIT/InvIT trusts remain explicitly outside this corporate classifier's scope. Combined with the separate recent cache, 105 total records are inventoried; 99 corporate outcomes are verified, one is pending, and five trusts are excluded. The mixed-stage pooled diagnostic is 75/99 (75.76%), versus GMP 72/99 (72.73%) and always-positive 66/99 (66.67%). Do not treat this mixed-stage aggregate as prospective or uniformly timed accuracy. See `output/ALL_KNOWN_2026_EVALUATION.md` and `output/all_known_2026_frozen_evaluation.json`. The 81.2% figure remains the earlier 69-case subset, not the expanded result.
