-- Private row mirror for the local SQLite ledgers. Only the service-role key
-- can read or write these records; no anon/authenticated policy is provided.
CREATE SEQUENCE IF NOT EXISTS public.capstone_sqlite_sync_version_seq;

CREATE TABLE IF NOT EXISTS public.capstone_sqlite_mirror (
    db_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    row_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    row_sha256 TEXT NOT NULL,
    sync_version BIGINT NOT NULL DEFAULT nextval('public.capstone_sqlite_sync_version_seq'),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (db_name, table_name, row_key)
);

CREATE OR REPLACE FUNCTION public.touch_capstone_sqlite_mirror()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.sync_version := nextval('public.capstone_sqlite_sync_version_seq');
    NEW.updated_at := clock_timestamp();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_capstone_sqlite_mirror
    ON public.capstone_sqlite_mirror;
CREATE TRIGGER trg_touch_capstone_sqlite_mirror
BEFORE UPDATE ON public.capstone_sqlite_mirror
FOR EACH ROW EXECUTE FUNCTION public.touch_capstone_sqlite_mirror();

CREATE INDEX IF NOT EXISTS idx_capstone_sqlite_mirror_version
    ON public.capstone_sqlite_mirror (sync_version);

ALTER TABLE public.capstone_sqlite_mirror ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.capstone_sqlite_mirror FROM anon, authenticated;
GRANT ALL ON public.capstone_sqlite_mirror TO service_role;
GRANT USAGE, SELECT ON SEQUENCE public.capstone_sqlite_sync_version_seq
    TO service_role;

COMMENT ON TABLE public.capstone_sqlite_mirror IS
    'Private bidirectional row mirror for Capstone local SQLite ledgers; managed by services.sqlite_supabase_sync.';
