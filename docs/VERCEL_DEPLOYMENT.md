# OpenBell: manual Vercel deployment

Import `udits31/IPO-Intelligence` with repository root **`.`**.
Keep the root `vercel.json`; do not make a separate frontend-only project.

## Services

- `frontend`: public Next.js service, including same-origin `/api/v1` routes.
- `app`: internal Python/FastAPI container. Frontend Node handlers reach it
  through the runtime `OPENBELL_APP_URL` binding. Do not set that variable.
- `collector`: one continuous daemon in `bom1`. A Supabase lease prevents two
  cloud workers from collecting simultaneously during overlapping deployments.

Leave `NEXT_PUBLIC_API_BASE_URL` unset. No localhost backend URL is used on Vercel.

## Environment settings

Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to your Supabase project's
values. Never prefix server secrets with `NEXT_PUBLIC_`. The Dockerfiles configure Supabase storage and
separate API/collector roles automatically. No PostgreSQL password, local
Supabase instance or local Docker is required; Vercel builds the containers.

## Database and source files

Once the Supabase project is prepared (see below), thirty-six typed ledger
views expose its stored records without duplicating hundreds of MB. Production
reads/writes use service-role API transactions and do not open SQLite files.
Previews roll back, commits are atomic, unique identities are retained and
stale writes are rejected. Saved history cannot be changed through the ledger
views. Anonymous users cannot call the database functions.

Required evidence, caches and report inputs are uploaded to the private
`openbell-runtime` bucket by `services.cloud_assets --publish`. Files are checksum-addressed and verified on restore.
Calendar, monitoring and research assets load separately. New daily decision
sources are archived before their calls are committed. Original SQLite files,
cloud rows, prediction payloads and model weights were not deleted or rewritten.

Local development still uses SQLite and its mirror. Local uploads pause while
the cloud collector holds its lease, protecting cloud identities. Once cloud
collection is verified, stop the laptop's collector to avoid duplicate source
polling; retain its data for recovery.

For a new Supabase project, apply migrations `002` through `005`, run
`python -m db.install_native_ledgers --apply`, then
`python -m services.cloud_assets --publish`. If you have the original local
SQLite ledgers, populate the mirror from them first; otherwise the saved-call
history starts empty and builds up as the collector runs.

## Checks after Deploy

1. Confirm both services build and the collector daemon starts.
2. Open `/api/v1/health`. `storage.backend` must be `supabase_api` and
   `storage.sqlite_used` must be `false`. The collector's
   `external_lease_active` must become `true`; request-service threads are
   intentionally disabled.
3. Test Market Today, Research, Compare, Performance and Module 2 through the
   deployed URL. Check a saved call against its actual later close.
4. Inspect the daemon logs and an actual collection slot during an open market.
   A successful homepage is not proof of working collection.

The config schema, frontend production build, targeted tests and real Supabase
transaction checks have passed. Vercel's final image build, runtime bindings
and deployed URL still need the manual deployment smoke test. No deployed
uptime is claimed. Daemons are public beta and use the account's compute quotas.

## Saturday presentation

Past assessments and outcomes remain visible while markets are closed. Module
2 shows the assessment/reason and one saved-call comparison chart. Intraday
data, the full call table, evidence and personal holdings are expandable.
The minute chart retains the latest saved session on Saturday; it does not
require Saturday candles. Unobserved future outcomes remain pending.

## Evidence

`scripts/verify_native_supabase.py` tests the real Supabase API with SQLite
connections blocked: 93 frozen benchmark replays, saved monitor outcomes,
durable writes, rollback, stale-write protection, immutable history, duplicate
handling and exclusive worker ownership. Verification writes use a separate
storage-check table, never invented IPO prediction records.

Results: `output/native_supabase_storage_verification_20261030.json`.
