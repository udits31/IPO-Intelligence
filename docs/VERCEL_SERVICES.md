# Historical deployment preparation — superseded

Use [the current deployment guide](VERCEL_DEPLOYMENT.md). The storage blockers
described below were subsequently implemented and verified against Supabase.

Import the **repository root**, not `frontend/`, as a single Vercel project.
The root `vercel.json` is the only service/routing configuration.

## Proposed names and routing

- `frontend`: Next.js, root `frontend`. Public on `/` and `/api/v1/*`.
- `app`: FastAPI, root `.`, explicit `api.main:app` entrypoint. Internal only.
- Caller `frontend` binds to `app` as `OPENBELL_APP_URL`.

Please confirm these names, public paths and internal binding before treating
the layout as approved. Routing every path to FastAPI, as in the initial
suggestion, would prevent the dashboard from loading.

## Request path

Browser `/api/v1/module2/holdings?target=...` → top-level frontend rewrite →
Next.js Node route handler → runtime `OPENBELL_APP_URL` binding → FastAPI route
`/api/v1/module2/holdings`.

Bindings do not exist during builds or in middleware. The handler reads the
binding only at runtime, preserves method/body/query/status, disables caching
and does not expose the internal URL. Production fails with a clear 503 when
the binding is absent, rather than trying localhost. Local `npm start` uses
the same handler with `127.0.0.1:8000` as its development fallback.

Do not set `OPENBELL_APP_URL` yourself. Do not set
`NEXT_PUBLIC_API_BASE_URL` to an old external backend: leave it unset so the
browser uses same-origin `/api/v1`. `OPENBELL_BACKEND_ORIGIN` is only a local
fallback, not a Vercel binding override.

Run `vercel dev` from the root for the complete service topology. It requires
the Vercel CLI/account and has not been run in this unauthenticated session.
The isolated Next production build and the proxy unit tests are local checks,
not proof of a successful Vercel deployment.

## Supabase-only production is still a blocker

The existing system uses direct SQLite queries in its collectors, saved-call
readers, outcome reconciliation, intraday storage and maintenance. The current
Supabase row mirror is durable backup/sync; it is **not** a native PostgreSQL
replacement for those queries. No environment variable or `vercel.json` setting
changes that. This configuration does not claim otherwise.

Before a functioning production deployment:

1. Replace the working ledger repositories with native Supabase/PostgreSQL
   readers/writers, with transactions and stable immutable event identities.
2. Migrate/validate existing saved predictions and price/source evidence without
   deleting or rewriting original history. Verify row counts and SHA-256 values.
3. Store/retrieve required archived source bytes remotely, rather than assuming
   Git contains ignored local archives. Preserve checksum validation.
4. Test reads/writes against the real Supabase schema, including duplicate
   collection attempts and concurrent requests. Never use anonymous browser
   credentials for privileged ledger writes.
5. Move continuous collectors out of request-bound lifespan threads into an
   independently scheduled worker/daemon. Vercel Services alone does not keep
   them running after scale-down.

Backend secret settings are `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` and
optional `GEMINI_API_KEY`. A future native PostgreSQL adapter may additionally
require the Supabase pooler connection URL; do not paste passwords in chat.
These settings alone do not complete the migration above.

Keep the existing local backend/frontend and collection running until the
deployed environment passes real end-to-end checks. A deployed static homepage
with unavailable database routes is not a completed application deployment.

## Saturday presentation

Historical price trends, saved calls and their completed outcomes do not need
an open market. Live collectors do observe market/weekday windows. Future
trading-session outcomes must remain pending until that session completes.
The deployed backend must have access to the saved calls **and** validated
price archives for these charts to work; pushing source alone does not transfer
all the ignored data.
