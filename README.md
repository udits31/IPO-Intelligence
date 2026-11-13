# IPO Intelligence

**OpenBell** — an IPO research and post-listing monitoring platform for Indian equity markets (NSE/BSE).

It does two things:

- **Module 1 · Pre-listing research.** Follows an IPO through its bidding window, saves a frozen model's probability of a positive listing open *before* the application cutoff, and later checks that saved call against the verified exchange opening price.
- **Module 2 · Post-listing monitoring.** Watches newly listed shares day by day using eligible prices, dated news and exchange filings, records a dated Hold/Sell call, and matches each call with the price that actually followed.

Every forecast is stored with its inputs, model version and timestamp, so historical replays and genuine pre-close calls are never mixed.

---

## Features

- **Market Today**: open, upcoming and recently listed IPOs with subscription, GMP and offer details
- **IPO research and comparison**: per-issue probability, inputs, comparable past IPOs and an application screen (Apply Tier 1 / Tier 2 / Skip)
- **Model performance**: locked benchmark results, cohort-by-cohort accuracy and model lineage
- **Module 2 monitor**: daily calls, intraday context and a saved-call vs. actual-price chart (1, 5 and 30 sessions)
- **Data provenance**: issuer identity checks, source clocks and SHA-256 hashes on archived evidence

## Architecture

```mermaid
flowchart LR
    U[Browser] --> F[Next.js dashboard<br/>frontend/]
    F -- "/api/v1/* proxy" --> A[FastAPI backend<br/>api/main.py]
    A --> S[services/ · core/<br/>models and monitors]
    A --> L[(Ledgers<br/>SQLite locally · Supabase in production)]
    W[Collector worker<br/>services/production_worker.py] --> L
    W --> X[NSE · BSE · price history · news · filings]
    S --> M[ml/artifacts<br/>frozen models]
```

| Layer | Stack |
| --- | --- |
| Frontend | Next.js 14, React 18, Tailwind CSS |
| Backend | Python 3.11, FastAPI, Uvicorn |
| Modelling | scikit-learn, pandas, NumPy (joblib artifacts) |
| Storage | SQLite (local), Supabase Postgres and Storage (production) |
| Deployment | Docker, Vercel Services |

## Models and results

**Module 1** (`M1_V1_FROZEN_WITH_RECOVERED_INPUTS`) is a stacked classifier trained on 290 IPOs from 2021–2025. It has five expert views (Base, Base + Macro, Financial + Macro, Trajectory, GMP-only) and a logistic-regression meta-model over 23 features, with a fixed 0.50 cutoff. The artifact is `ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib`, and its exact features and parameters are exposed by `services/model_lineage.py`.

| Cohort | Direction correct |
| --- | --- |
| 2026 development cohort (69 IPOs) | 56/69 (81.2%) |
| Recent verified replay (24 IPOs) | 17/24 (70.8%) |

The application screen (`services/retail_hni_economics.py`) is separate from the direction model. On the 69-IPO cohort it applied to 18 IPOs, all of which opened positive, but it also skipped 28 of the 46 winners.

**Module 2** (`M2_R1_WITH_DATED_FILINGS`) is a deterministic daily price and filing monitor. On 132 matured 2025–2026 IPOs it scored 63/132 (47.7%), versus 77/132 (58.3%) for always-sell. **It has not demonstrated a trading edge.** Learned next-session and five-session models are kept as research and shadow logging only.

> These are historical evaluations, not guarantees of future accuracy. Probabilities are not calibrated return forecasts, and nothing here is investment advice.

## Repository layout

| Path | Contents |
| --- | --- |
| `api/` | FastAPI app, routes and worker lifecycle |
| `core/` | Feature engineering, predictor, similarity, attribution and monitor engines |
| `services/` | Data feeds, Module 1/2 services, forecasting, evidence and storage sync |
| `ml/` | Data collection, recovery, research, training and evaluation scripts; `ml/artifacts/` holds frozen models |
| `data/` | Curated datasets (IPO master, subscription, market and post-listing prices, audit snapshots) |
| `output/` | Locked evaluation reports and evidence read by the app at runtime |
| `db/` | Supabase migrations, ledger adapter and sync scripts |
| `frontend/` | Next.js dashboard and API proxy route |
| `tests/` | Python test suite (`frontend/tests/` holds the Node tests) |
| `docs/` | Deployment, data lifecycle and release guides, project plan |

## Run locally

**Requirements:** Python 3.11+, Node.js 18.17+ and npm.

Locally, the backend uses SQLite under `data/telemetry/` (created on first run). Supabase is optional.

**1. Backend**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-runtime.txt
cp .env.example .env          # optional: add Supabase keys to mirror local ledgers
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Keep `--workers 1`. The API process also runs the data collectors, and extra workers would poll sources twice.

**2. Frontend** (in a second terminal)

```bash
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000. Health check: http://127.0.0.1:8000/api/v1/health.

## Configuration

| Variable | Where | Purpose |
| --- | --- | --- |
| `SUPABASE_URL` | backend | Supabase project URL (required in production) |
| `SUPABASE_SERVICE_ROLE_KEY` | backend | Server-side key for ledgers and storage. Never expose it to the browser. |
| `OPENBELL_STORAGE_BACKEND` | backend | `sqlite` (local default) or `supabase` (set by the Dockerfiles) |
| `OPENBELL_COLLECTORS_ENABLED` | backend | `1` to run scheduled collectors in that process |
| `LIVE_DATA_TTL_MINUTES` | backend | Market Today cache lifetime (default `30`) |
| `OPENBELL_APP_URL` | frontend | Backend URL used by the `/api/v1` proxy (injected automatically on Vercel Services) |

## Deployment

Production runs three pieces: the **Next.js frontend**, the **FastAPI API** (`Dockerfile`) and one **collector worker** (`Dockerfile.worker`). All three use Supabase for storage.

### Step 1 — Prepare Supabase

1. Create a Supabase project and copy its URL and **service role** key.
2. In the SQL editor, run `db/migrations/002` → `005` in order.
3. From your machine, with `.env` filled in:
   ```bash
   python -m db.install_native_ledgers --apply   # create the typed ledger views
   python -m services.cloud_assets --publish     # upload runtime evidence to the private bucket
   python scripts/verify_native_supabase.py      # optional end-to-end storage check
   ```

A new Supabase project starts with an empty saved-call history. The locked benchmark reports in `output/` are uploaded in step 3, and new calls accumulate once the collector runs.

### Step 2, option A — Vercel Services (configured in `vercel.json`)

1. In Vercel, import this repository with the **root directory set to `.`**. Don't create a frontend-only project.
2. Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as environment variables.
3. Deploy. `vercel.json` builds the `frontend` service, the internal `app` container and the `collector` daemon (region `bom1`), and binds `OPENBELL_APP_URL` automatically. Don't set it yourself, and leave `NEXT_PUBLIC_API_BASE_URL` unset.

Vercel Services and daemons are a beta feature and count against your account's compute quota.

### Step 2, option B — Vercel frontend + Docker backend

Use this if Vercel Services isn't available on your plan.

1. Deploy the backend from `Dockerfile` to any container host (Render, Railway, Fly.io or a VM), exposing port `8000` and setting the Supabase variables.
2. Deploy `Dockerfile.worker` as a separate always-on worker with the same variables. Run exactly one instance.
3. Import the repo into Vercel with **root directory `frontend`** and set `OPENBELL_APP_URL` to the backend's public URL, e.g. `https://your-api.onrender.com`.

### Step 3 — Check the deployment

1. `GET /api/v1/health` should report `storage.backend = supabase_api` and `storage.sqlite_used = false`.
2. Open Market Today, Research, Compare, Performance and Module 2 from the deployed URL.
3. During market hours, check the worker logs for a completed collection slot. A homepage that loads does not prove collection is working.

See [`docs/VERCEL_DEPLOYMENT.md`](docs/VERCEL_DEPLOYMENT.md) and [`docs/AUTOMATIC_DATA_LIFECYCLE.md`](docs/AUTOMATIC_DATA_LIFECYCLE.md) for details.

## Testing

```bash
pytest -q                          # Python tests
cd frontend && node --test tests/*.test.mjs && npm run build
```

Some research tests need large local archives that are not in Git (`data/archive/`, `data/telemetry/`) and are skipped or fail without them.

## Limitations

- GMP and some subscription data come from unofficial aggregators, and those sources are not independent confirmations.
- Collectors follow weekday market hours. Exchange holidays and special sessions need separate calendar checks.
- Missing evidence is reported as missing; it is never filled with invented values.
- The cloud deployment path has passed configuration, build and Supabase storage checks, but has not been run as a long-lived production service.

## Further reading

- [`docs/RELEASE_GUIDE_20261030.md`](docs/RELEASE_GUIDE_20261030.md): release notes and evidence map
- [`docs/research_papers_bibliography.md`](docs/research_papers_bibliography.md): research references
- [`docs/README_HISTORY_BEFORE_RELEASE_20261030.md`](docs/README_HISTORY_BEFORE_RELEASE_20261030.md): earlier designs (historical)
- `output/conference_revision/`: conference manuscript source
