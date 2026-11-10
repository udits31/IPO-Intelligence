# OpenBell — IPO Research and Post-Listing Monitoring

Current release: 30 October 2026. This is the current project guide. Older designs in `docs/README_HISTORY_BEFORE_RELEASE_20261030.md` are historical context, not the active architecture.

## Application and workflow

The Next.js dashboard provides Market Today, Module 1 research, IPO comparison, performance evidence and Module 2 monitoring. FastAPI resolves issuers, collects dated sources, records forecasts, reconciles official openings and serves evidence. Gemini provides optional explanations; it does not replace the statistical model or determine benchmark scores.

Module 1 follows an IPO through bidding, saves its probability and inputs before the application cutoff, then compares the preserved call with the verified listing opening. Listed IPOs without an actual saved call can display a labelled historical benchmark; that replay never enters the prospective ledger. Module 2 monitors listed shares using eligible prices and news/filings, and records dated actions for later outcome checks.

## Module 1: active model

Configuration: `M1_V1_FROZEN_WITH_RECOVERED_INPUTS`.

Artifact: `ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib`.

SHA-256: `bf07f15c86fce9984751b36cda7636f3c52875202492199a0017c8842406ff38`.

Training uses 290 original IPOs from 2021–2025. Five views are Base, Base + Macro, Financial + Macro, Trajectory and GMP-only. Four experts use logistic regression and Trajectory uses a random forest. The standardized expert-log-odds meta-model is logistic regression with C=0.5. The union has 23 input features; individual experts use subsets. The binary cutoff is fixed at 0.50, without post-hoc probability calibration. Recovering inputs does not retrain weights. Exact feature lists and parameters are exposed by `services/model_lineage.py`.

Inputs include category subscription demand, dated GMP, offer price/size, eligible market context, financial facts and demand trajectory. Missing values follow the frozen pipeline and remain disclosed. Probability of a positive opening is not predicted return or a calibrated guarantee.

| Cohort | Raw direction correct | Interpretation |
| --- | --- | --- |
| Earlier 2026, 69 IPOs | 56/69 (81.2%) | Repeatedly inspected retrospective development cohort |
| Recent verified 24 | 17/24 (70.8%) | Recent historical replay |
| Original recent 23 | 16/23 (69.6%) | Denominator before the additional verified opening |

These figures do not establish future accuracy. Day-2 historical performance cannot be assigned to arbitrary live intraday stages. New genuinely saved pre-close forecasts are evaluated separately. Flat openings count as nonpositive.

### Direction and application decision are different

The frozen model predicts whether the listing open exceeds the issue price. `services/retail_hni_economics.py` separately checks demand, OFS, quota, GMP and a disclosed cost/allotment scenario to produce Apply Tier 1, Apply Tier 2 or Skip. The screen never rewrites a wrong model direction.

On the earlier 69 cases it selected 18 winners out of 18 applications, but skipped 28 of 46 positive openings. On the recent 24 it selected 12 winners out of 14 applications, admitted two losers and skipped six winners. Selected APPLY precision is not overall accuracy or proof of greater portfolio profit. Reciprocal subscription odds are approximations and scenario rupees are not realised returns.

ESDS illustrates the distinction: a positive model call matched its Rs757 opening versus Rs429 issue price, but the application screen skipped it because Day-2 QIB demand was below 1x. Direction correct; winner skipped. It is not an overall successful recommendation.

### Historical versus saved evidence

A genuine saved call retains its original model, inputs and timestamp. A newer historical replay stays separate. The research page now uses the same locked probability and dated inputs as Model Performance for 93 benchmark IPOs, with verified opening comparisons. All 93 frozen replays matched their locked probabilities. Historical tests are clearly labelled and never called live pre-close predictions.

## Module 2: active monitor and separate ML research

The production policy is `M2_R1_WITH_DATED_FILINGS`, a deterministic daily price/filing monitor. It is not the separately trained next-day research classifier. It combines eligible price levels, pullback/trend context, dated news/filings and supply-calendar warnings. Identity conflicts, unknown price basis or missing evidence can pause advice. Split/bonus adjustments are checked before comparisons with issue price.

The frozen governance benchmark uses the 31st observed close, next-close execution and a 30-session outcome on 132 matured 2025–2026 IPOs. R1 scored 63/132 (47.73%), retained 44/55 winners and identified 19/77 losers. Mean net return was -0.362% under the original protocol. Always-sell scored 77/132 (58.33%) and -0.300%. This test has not demonstrated a trading edge.

Sequential research follows daily state changes rather than forcing every initial hold to remain held. One-way stops can miss rebounds; following successive calls slightly improved the static result but did not establish superiority over full exit. Costs and conventions are in `output/module2_sequential_path_evaluation_20261026.json`.

The next-session learned research accuracy was about 56.25% versus 54.91% for its majority baseline. Five-session HGB research used 206 earlier training issuers and 159 later evaluation issuers. Mean window return was 2.688% versus 2.599% for rules, but the paired improvement interval included zero and 2026 trailed rules. It remains research/shadow logging. Do not combine different horizons/cohorts into one accuracy figure.

### Saved call versus actual price chart

Module 2 shows dated calls, historical prices and Matched/Missed/Pending outcomes. Select a call to inspect its action, preceding known close, later verified close and return. Horizons are 1, 5 or 30 trading sessions. Example API: `/api/v1/module2/call-outcomes?ipo_id=610&horizon=1`.

The chart reads genuine saved decisions and checksum-verified price archives. It has no weekend display restriction. Saturday can show Friday and earlier evidence. An outcome requiring a future unobserved session remains pending. A directional match is not execution profit, and a rule-based call is not labelled as a learned forecast.

## Data, collection and storage

Sources include NSE/BSE issue information and price archives, offer documents, IPOPremium/IPOJI and other archived offer facts, unofficial GMP providers, Yahoo price history and dated news/filings. Issuer identity, source clocks and hashes are preserved. Two related aggregators are not independent confirmations. Later archive downloads are not live captures recorded in the past.

The API process starts subscription, frozen Module 1, daily Module 2, intraday, maintenance, news and research-shadow workers. These require a continuously running backend. Opening a frontend alone does not start collection. Repeated attempts are recorded; training readers deduplicate without deleting raw captures. Missing readings never become fabricated zero values. Closed offers stop receiving application-time calls; listing outcomes are reconciled separately.

Local development uses SQLite in `data/telemetry/` with a Supabase mirror. Production uses Supabase itself through the native API ledger adapter. Writes request sync; startup reconciles it. Inspect conflicts and failed syncs. Supabase mirroring does not automatically replace every local archive, model, CSV and cache used by the application. Never commit secrets or private telemetry.

### Weekend behaviour

Pages, historical analysis, saved calls, benchmarks and archived charts can be served every day. Several automatic market collectors explicitly skip weekends: intraday collection follows weekday exchange hours, daily Module 2 and research-shadow workers have weekday schedules, and Module 1 checks offer dates and cutoffs. There is no weekend login/application block.

Do not remove market-hour checks just to show a presentation or invent Saturday prices. Show the latest completed session and genuine older calls. Special exchange sessions/holidays need separate calendar verification; weekday checks alone are not a complete exchange calendar.

## Local setup

Use Python 3.11 and Node.js/npm. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-runtime.txt
# Configure .env from .env.example, entering secrets locally.
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal:

```powershell
cd frontend
npm ci
npm run build
npm start
```

Visit http://localhost:3000/ and check http://127.0.0.1:8000/api/v1/health. One backend worker is deliberate: multiple workers duplicate in-process collectors. Do not start a second backend against the same ledgers as a deployment shortcut.

## Vercel multi-service deployment

Use the repository-root `vercel.json`. The layout has a public Next.js `frontend`, an internal FastAPI container `app`, and a leased `collector` daemon. The frontend runtime API handler reaches `app` using the injected `OPENBELL_APP_URL` binding. Leave `NEXT_PUBLIC_API_BASE_URL` unset; the browser calls same-origin `/api/v1`. Do not set the binding variable manually. There is no build-time localhost rewrite.

See [Vercel service configuration and outstanding production requirements](docs/VERCEL_DEPLOYMENT.md) for the exact routing, confirmation checklist, local `vercel dev` workflow and production blockers.

**Production storage now uses the existing Supabase API credentials.** Thirty-six typed ledger views expose the existing cloud records without duplicating raw history. Production does not open SQLite. Required evidence files are stored privately in Supabase Storage; a separately leased daemon owns scheduled collection. Live database checks passed. The final Vercel image build, bindings and deployed URL still require the manual smoke test; no deployed uptime or improved model accuracy is claimed from this migration.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
node --test tests/*.test.mjs
npm run build
```

Some full-suite tests require local archived research fixtures. Release preparation reran 23 targeted Python tests and 17 frontend tests successfully. Earlier frontend production builds passed. The cloud configuration still needs authenticated build and end-to-end verification. A targeted run is not a full-suite pass.

## Source and evidence map

| Location | Purpose |
| --- | --- |
| `api/main.py` | API routes and worker lifecycle |
| `services/prospective_module1.py` | Frozen inference and actual pre-close logging |
| `services/module1_benchmark_lookup.py` | Shared historical research/performance results |
| `services/retail_hni_economics.py` | Application screen and rupee scenarios |
| `services/module2_decision_ledger.py` | Saved daily monitor calls |
| `services/module2_call_outcomes.py` | Saved call/outcome matching |
| `services/module2_intraday.py` | Indicative minute-price monitoring |
| `services/module2_shadow_forecast.py` | Research-only learned forecasts |
| `services/sqlite_supabase_sync.py` | Ledger reconciliation |
| `output/FINAL_SYSTEM_LOCK_20261026.json` | Locked release accounting |
| `output/module1_two_track_evidence_20261026.json` | Direction versus application results |
| `output/module1_research_benchmark_link_audit_20261030.json` | 93 replays, zero mismatches |
| `output/module2_multihorizon_20261029/` | Five-session research and shadow protocol |
| `docs/AUTOMATIC_DATA_LIFECYCLE.md` | Collection lifecycle |

## Development history and limitations

The project evolved from an earlier subscription-only classifier and static monitor to a five-expert direction stack, separate application screen, dated source recovery and chronological comparisons. It added identity checks, pre-close logging, preserved call/outcome records, Supabase mirroring, offer/financial recovery, intraday context, model lineage and matched comparisons. Cached Market Today responses refresh in the background; atomic cache writes prevent concurrent truncation.

Problems addressed include late/missing books, interim/final and Day-2/live mismatches, unclocked GMP, post-listing data incorrectly used as pre-listing context, ambiguous symbols, split/bonus price bases, sparse financials, duplicate training observations, misleading combined success labels and historical performance disconnected from Research. Missing reliable evidence remains a limitation instead of being invented.

For faculty documentation, state the cohort, model version, horizon, cutoff and historical/prospective status. Do not claim guaranteed future accuracy, calibrated probabilities, exact allotment odds, guaranteed profit, proven cloud uptime or a demonstrated Module 2 trading edge.
