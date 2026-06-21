-- Typed ledger views order by the original immutable JSON primary key.
-- Index that small key instead of decompressing raw source payloads to sort.
CREATE INDEX IF NOT EXISTS capstone_ledger_numeric_id_lookup_last
ON public.capstone_sqlite_mirror
  (db_name,table_name,((row_key::jsonb->>'id')::bigint) DESC NULLS LAST);
