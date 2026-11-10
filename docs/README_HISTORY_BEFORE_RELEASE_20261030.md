# Indian IPO Intelligence System

IPO research and post-listing monitoring for Indian mainboard offerings.

## Module 1 model contract

The deployed final-book model is a three-input logistic classifier trained on 252
corporate IPOs listed in 2021–2024. Two 2021 investment trusts are excluded
from the 254-record historical universe because they are not corporate IPOs
and lack retail subscription books. No training bid is imputed. Its inputs are final QIB, retail, and HNI subscription
multiples. It estimates the probability that the exchange opening price exceeds
the IPO issue price. It does **not** predict an exact opening price or return.
GMP, VIX, and offer-document facts are shown separately as context, not silently
used as model features. Before the book closes, or when any final subscription
input is unverified, that final-book model does not display a final-book probability.
Live provisional bid forecasts are saved separately and cannot borrow the
final-book model's reported historical accuracy.

## Pre-close return forecasts and research priorities

Module 1 now has a separate experimental return challenger. It forecasts a
central opening return and empirical lower/upper scenarios, with a separate
early-book probability classifier. It does not turn a positive-opening
probability into a return estimate. Training uses only matched 2021–2024
penultimate-session bids; 2025–2026 records are not added to training.

Reproduce the historical GMP collection with `python -m ml.collect_preclose_gmp`,
then train and benchmark with `python -m ml.return_pipeline`. The trainer writes
`ml/artifacts/preclose_return.joblib` and its hashed metadata; these are local
build artifacts, not a replacement for `ml/model_bundle.json`. The full report
and per-IPO prediction ledger are in `output/module1_return_benchmark.md`,
`output/module1_return_benchmark.json` and `output/module1_return_prediction_ledger.csv`.

Model family selection uses earlier-year 2022/2023 replay, followed by a 2024
retrospective confirmation. GMP-only and model selectors use identical dated,
feature-supported candidate sets. A daily GMP quote must precede the bid date;
unknown same-day publication clocks are not assumed to precede an application.
Raw matched GMP articles are archived under `data/archive/preclose_gmp/`.

`GET /api/v1/module1/priorities` ranks fresh saved pre-close return estimates by
central return minus half the lower-band loss below zero, minus a disclosed
0.5 percentage-point cost scenario. It can skip all. These are research
priorities, not allotment-adjusted portfolio profits. Old books, future clocks,
closed offers and inputs outside training support cannot receive a live rank.
`GET /api/v1/module1/return-benchmark` reports the historical and separate
prospective return scorecards. No reliable superiority to GMP has been
established; the empirical bands under-cover 2024 outcomes and are NOT valid
80% live prediction intervals.

The collector saves each new return estimate inside the existing append-only
forecast input record and freezes the joint priority/GMP comparison in
`module1_priority_v1`. It never adds this new model's predictions to old records
or rewrites a post-close computation as a pre-close prediction. Future official
listing opens are matched separately. Missing outcomes remain pending.

Module 2 refreshes dated price/news evidence on access and in its existing
collector. Stale news retains its previous headlines but cannot provide a fresh
sentiment score; stale daily prices cannot issue a current holding action.
Unfinished same-day candles are excluded from close-based rules. Unverified
autocomplete quotes remain supplementary context, not decision prices.

The live monitor now reconciles explicit split/bonus ratios before comparing
historical prices with the issue price. Yahoo history already incorporates
splits, so it is not adjusted twice. The newest quote is unchanged; the price
waterfall labels a share-adjusted issue price and keeps the original offer
price visible. Unknown capital events pause advice rather than inventing a
ratio from a price drop. Cash dividends are excluded from price returns.
Portfolio P&L requires confirmation that entered quantities and average prices
already include any detected share changes.

Canonical IDs, company names and eligible additional listings use the same
core monitor policy, not a separate fallback trading tree. For supported BSE
issues, the provider ticker is resolved from the exact scrip in a hash-checked
BSE archive; a numeric scrip code is not assumed to be a provider ticker.
BSE share events currently have secondary-provider authority, explicitly
distinguished from NSE cross-checked events. Missing identity/price evidence
does not become a guessed quote or a pre-listing verdict.

`ml/model_bundle.json` is the promoted model and its matching chronological
replay. Rebuild it with `python -m ml.train_module1` from the repository root,
then run `python -m unittest discover -s tests -p "test_model_bundle_contract.py"`
and `python -m unittest discover -s tests -p "test_module1_lifecycle.py"`.
The bundle records the cohort size, input SHA-256 hashes, features, coefficients,
threshold, and every replayed result. The 2022–2023 replay selected the 0.82
decision threshold. 2024 is a retrospective confirmation; 2025–2026 is a
later-year retrospective check, **not** a blind prospective test. Historical
source-publication timestamps still need independent verification before any
claim of point-in-time backtesting.

### Exchange-outcome and monitor data audits (21 October 2026)

`python -m ml.audit_module1_listing_opens` preserves the original 252 training
records while checking opening outcomes against final exchange archives. It
uses NSE at the IPO date, or exact-code BSE when not listed on NSE then, and
resolves old symbols from official dated change records. It does not overwrite
the canonical dataset. `python -m ml.replay_module1_exchange_outcomes` separately
rescores the unchanged return forecasts and tests a source-corrected refit.
The app exposes only the matching, hashed **unchanged-forecast** source audit;
refit research scores are not represented as deployed-model performance.

Module 2 data recovery and replay scripts are research-only:

- `python -m ml.recover_module2_be_closes`: recover genuine initial BE-series
  prices even when delivery fields are missing; no zero delivery is invented.
- `python -m ml.extend_module2_late2024`: recover the missing 2025 outcome
  sessions of late-2024 IPOs, with source-window overlap and calendar checks.
- `python -m ml.reconcile_module2_corporate_actions`: apply explicit NSE bonus
  and split ratios on their actual ex-dates, and resolve listing-close conflicts
  using raw final daily archives. Cash dividends are not included.
- `ml.evaluate_module2_holdout` accepts an audited `--prices` snapshot and a
  `--listing-close-audit` ledger. Its drawdown is peak-to-trough, not merely the
  minimum price relative to entry. Original snapshots remain unchanged.
- `ml.calibrate_module2_rules` tests 75 fixed settings of the same monitor rules.
  Only 2021–2024 IPO outcomes known at the freeze date can select settings.

The latest corrected Module 2 replay has 132 matured later IPOs: 50.0% correct
30-session direction, versus 58.3% always-sell. Earlier-data calibration made
this worse (47.0%) and was **not promoted**. This is price-only retrospective
evidence, not an accuracy measurement of the full live news monitor. The
75–80% goal and a pre-close selection advantage over GMP remain unestablished.
See `output/model_validation_status_20261021.md` for the current evidence and
the exact reports; older research snapshots must not override corrected results.

### Presentation-friendly performance page (22 October 2026)

The Model performance screen now separates conclusions, negative results,
tested improvements, six design challenges and future validation. Presenter
readouts explain the numbers without substituting slide constants. Its API,
`GET /api/v1/performance/evidence`, checks the saved report against the active
model, input hashes, replay and per-candidate ledger; mismatches hide the scores.

Rebuild evidence with `.venv\Scripts\python.exe -m ml.build_performance_evidence`
after rerunning `ml.evaluate_module2_holdout` with the audited share-reconciled
prices, listing-close audit and `--tag presentation_20261022`. The comparison
uses 20 candidates, the original 252 corporate training records and the same
179 verified later outcomes. Runtime/dependency versions are recorded.
`output/performance_presenter_notes_20261022.md` contains matching speaking notes.

The 82% operating cutoff has 73.2% policy accuracy and 90.0% APPLY precision;
the distinct 50% direction cutoff has 76.0% accuracy on those same later cases.
These are retrospective final-book results, not investor-time return accuracy.
The Module 2 market-rule research score increased from 50.0% to 53.03% on the
same 132 IPOs, but its interval includes no gain and it trails always-sell.
It is labelled research-only, not a promoted or reliable live improvement.

`python -m ml.refresh_live_evidence` checks recoverable calendar/detail, demand,
GMP, market, news, listing-open and recent-monitor sources. The 22 October run
archived four complete demand-source responses for two offers and checked 47
recent listings. Actual afternoon captures stopped at 12:45; those original
missing times were not backfilled. Recovery data has its actual download clock.
The refresh report is `output/live_data_catchup_20261022.json`.

#### Dated news and issuer quality

The current keyword sentiment family is retained, but profit/revenue mentions
alone no longer count as positive. Financial direction matters (for example,
profit decline versus profit growth), and a narrower loss has the opposite
meaning to a wider loss. The scorer records `keyword_financial_context_v2_1` in
the monitor evidence. This is still heuristic sentiment, not a newly trained
language model or a validated causal return signal.

`python -m ml.collect_news_issuer_master` captures the official NSE name/symbol
list for disambiguation. Shared brands require a distinguishing issuer name;
general Indian Railways news is not automatically IRFC news, and Kalpataru
Projects is not Kalpataru Limited. Existing cached headlines are rechecked too.
The master identifies issuers only; current market data is not inserted into
historical prediction features.

For a retrospective news test, run `ml.collect_module2_dated_news`, then
`ml.reparse_module2_news` and `ml.replay_module2_news_overlay`. Every raw RSS
response is archived and hashed. Same-day historical headlines are excluded
when their original publisher clock is unverified. A successful empty search
is not evidence that there was no issuer news. The monitor's optional
`historical_news` input requires an explicit matching historical cutoff and
cannot fetch present-day RSS in its place. Research overlays are not promoted
automatically. The separately recorded news and market-rule experiments still
do not establish the 75–80% goal.

The UI compares both correct risk warnings and false warnings against the
always-positive baseline. Historical return bands summarize earlier-year IPOs
in each score band; they are not company-specific return forecasts. A model
warning is not a certainty, and position sizing remains the investor's decision.

`python -m ml.model_search` runs a read-only comparison of regularized linear,
nonlinear, tree, boosted, SVM, discriminant, and blended candidates using the
same chronological folds. Do not promote a candidate merely because it fixes a
known later-year miss. The active model remains the regularized three-bid
classifier because no tested alternative showed a consistent improvement in
ranking, probability quality, and risk-warning trade-offs. The UI reports a
year-stratified bootstrap refit sensitivity range to show how the score changes
with training-sample composition; it is **not** a statistical confidence or
prediction interval. Listed IPOs show their actual opening outcome and the
model's retrospective replay separately.

## Live subscription pilot

Start the backend from the repository root with
`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`. Keep the machine
awake and the backend running during 10:00–17:00 IST on each bidding weekday.
The backend discovers open IPOs hourly and attempts an IST-aligned capture every
15 minutes until 14:30, then every 5 minutes through 17:00. The source may
publish later than our poll, and a final exchange-verified book is not implied
by a 17:00 fetch.

Check `GET /api/v1/telemetry/snapshots` for missing slots, failed attempts, and
the last verified observation. For day-by-day comparison use
`GET /api/v1/telemetry/subscription-captures?date_ist=YYYY-MM-DD`.
Each attempt is append-only in `data/telemetry/snapshots.db` under
`subscription_capture_v2`, including failures and compressed source HTML.
`fetched_at_ist` is our clock; the source's publication time is **unknown**
unless independently verified. The older `pre_bidding_snapshots` rows are
retained for audit but must **not** be used for model evaluation: some have
stale or unsourced values. Four or five collection days validate the feed and
timing; they cannot establish prospective model accuracy.

#### Dated financial results: research inputs, not an accuracy claim

The financial-source audit now verifies 254 current-quarter result cases
(144 development, 110 later). Sixty-eight quarterly rows were recovered only
after the exact NSE filing detail resolved malformed WEB-XBRL exports that
gave quarter and year-to-date facts identical dates. The original exports are
preserved. A publication after the monitor checkpoint cannot become an input.
Profit margin alone did not improve the same 132-case later score: 50.0%,
versus 58.33% always-sell. No financial rule was promoted.

Original result PDFs and local OCR are being used to recover actual year-earlier
quarter comparisons. `ml.collect_module2_financial_pdfs` archived 176 matching
original PDFs; unavailable source cases remain in the evaluation. Run OCR in
the isolated `.venv-financial-ocr` environment using the pinned
`ml/requirements-financial-ocr.txt`, not in the deployed predictor environment:

```
.\.venv-financial-ocr\Scripts\python.exe -m ml.extract_module2_financial_comparatives
```

An optional `--cohort later` collection worker has its own checkpoint file and
shares the content-addressed raw page cache. It partitions collection only,
not evaluation or parameter selection. `python -m
ml.recheck_module2_financial_comparatives` independently reparses available
caches and checks each accepted table against its current-quarter source and
an explicit full-page visual review. This audit always retains all 376
development/later cases. A raw OCR result is never automatically training data.
Merged quarter headings are resolved using the entire column layout, not
whichever caption is closest. Annual, YTD, owner-only and comprehensive-income
figures must not replace total after-tax quarter profit. Prior losses retain
their signs, and a recovery from a loss is not labelled conventional growth.

`ml.replay_module2_earnings_rules` declares seven fixed policies conditioning
only the existing loss/relative-lag/trailing thresholds on verified earnings
changes. Parameter selection requires 2021-2024 outcomes matured before 2025;
all 132 later cases keep the original decision and 30-session horizon. Missing
comparatives retain the default rule, not zero earnings or exclusion. The
source extraction is still incomplete; neither this infrastructure nor more
source coverage establishes the requested 75-80% accuracy. No paper or live
model is changed by these research scripts.

The reachability audit in `ml.research_module2_risk_governance` found that the
three earnings-conditioned presets have a 61.36% pointwise hindsight ceiling
on the unchanged 132 later cases. Even the broader 75 tested price-threshold
settings have a 71.21% ceiling. These are **not model scores**: the diagnostic
unrealistically chooses a different setting after seeing each outcome. It
proves only that those finite threshold ranges cannot reach 75%, not that
every possible deterministic monitor is incapable of doing so. Hindsight
flags in the audit ledger are explicitly marked NOT_A_FORECAST and never
become model inputs. A nine-option macro/momentum governance experiment,
selected only on matured pre-2025 outcomes, scored 49.24% and was rejected.
The next research needs evidence-linked action logic beyond threshold tuning;
it must retain the deterministic family, all cases and the original horizon.

`python -m ml.extract_module2_native_financials` additionally checks exact
embedded text in every available original PDF. It does not assign a made-up
OCR confidence: native-token confidence is explicitly null and the review gate
replays text and coordinates from the original PDF before accepting anything.
The first complete sweep recovered eight source-matching tables, seven not
previously recovered through OCR. Image-only or garbled text still requires
the independent OCR/visual-review path. Collection outputs are not, by
themselves, accepted training features.

`ml.research_module2_earnings_events` tests thirteen fixed event-governance
policies in the same deterministic family, with settings chosen only from
matured 2021-2024 outcomes. A negative result needs an observed loss or verified
deterioration plus adverse price confirmation; an improving result can defer
a profit-taking exit only above issue price, not an underwater structural
stop. All 132 later cases remain. The first 24 fully checked comparisons
(14 development, 10 later) did not establish an improvement; earlier-data
selection retained the unchanged policy and later accuracy remained 50.0%.

Original-filings review also identified two opposite-direction before/after-tax
comparisons. Their reported profits/losses remain unchanged. Explicit reviewed
before-tax facts qualify the comparative event signal without removing the
IPO or manufacturing normalized earnings. This is not a claim that every
other accounting distortion has been detected. No live policy or paper result
is promoted from this incomplete-source experiment.
