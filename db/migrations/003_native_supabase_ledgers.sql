-- Native production ledgers, accessed through the existing Supabase API key.
-- No public/anonymous access and no database password is required by the app.
-- Original SQLite files and the old row mirror are not deleted or rewritten.
CREATE SCHEMA IF NOT EXISTS openbell_snapshots;
CREATE SCHEMA IF NOT EXISTS openbell_module2_decisions;
CREATE SCHEMA IF NOT EXISTS openbell_module2_intraday;
CREATE SCHEMA IF NOT EXISTS openbell_module2_historical_news;
GRANT USAGE, CREATE ON SCHEMA openbell_snapshots, openbell_module2_decisions,
  openbell_module2_intraday, openbell_module2_historical_news TO service_role;
REVOKE ALL ON SCHEMA openbell_snapshots, openbell_module2_decisions,
  openbell_module2_intraday, openbell_module2_historical_news FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.openbell_ledger_transaction_v1(
  p_database text, p_statements jsonb, p_preview boolean DEFAULT false
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog
SET standard_conforming_strings = on
AS $function$
DECLARE
  namespace text;
  statement text;
  results jsonb := '[]'::jsonb;
  result_rows json;
  column_names jsonb;
  binary_names jsonb;
  affected bigint;
  mutation boolean;
BEGIN
  IF current_user <> 'service_role' THEN
    RAISE EXCEPTION 'Service role required' USING ERRCODE = '42501';
  END IF;
  IF p_database NOT IN ('snapshots.db','module2_decisions.db','module2_intraday.db','module2_historical_news.db')
      OR jsonb_typeof(p_statements) <> 'array' OR jsonb_array_length(p_statements) > 2000 THEN
    RAISE EXCEPTION 'Invalid ledger transaction' USING ERRCODE = '22023';
  END IF;
  namespace := 'openbell_' || replace(p_database,'.db','');
  PERFORM set_config('search_path',format('%I,pg_catalog',namespace),true);
  -- Serialises competing collectors even across multiple Vercel instances.
  PERFORM pg_advisory_xact_lock(hashtextextended('openbell:'||p_database,0));
  BEGIN
    FOR statement IN SELECT jsonb_array_elements_text(p_statements)
    LOOP
      column_names := '[]'::jsonb;
      binary_names := '[]'::jsonb;
      result_rows := '[]'::json;
      mutation := false;
      affected := 0;
      IF statement ~* '^SELECT\M' OR statement ~* '^WITH\M' THEN
        -- CTAS preserves column order/type, including bytea, for the old reader
        -- interface. A temporary result is transaction-local and never durable.
        EXECUTE 'CREATE TEMP TABLE __openbell_rpc_result ON COMMIT DROP AS ' || statement;
        SELECT jsonb_agg(attname ORDER BY attnum),
          COALESCE(jsonb_agg(attname ORDER BY attnum) FILTER(WHERE atttypid=17),'[]'::jsonb)
          INTO column_names,binary_names
          FROM pg_attribute WHERE attrelid='pg_temp.__openbell_rpc_result'::regclass
          AND attnum>0 AND NOT attisdropped;
        EXECUTE 'SELECT COALESCE(json_agg(row_to_json(r)),''[]''::json),count(*) FROM pg_temp.__openbell_rpc_result r'
          INTO result_rows,affected;
        DROP TABLE pg_temp.__openbell_rpc_result;
      ELSIF statement ~* '^(INSERT|UPDATE)\M' THEN
        EXECUTE statement;
        GET DIAGNOSTICS affected = ROW_COUNT;
        mutation := true;
      ELSIF statement ~* '^CREATE (TABLE|INDEX|UNIQUE INDEX|OR REPLACE VIEW|OR REPLACE TRIGGER)\M'
          OR statement ~* '^ALTER TABLE\M' THEN
        EXECUTE statement;
      ELSE
        RAISE EXCEPTION 'Unsupported ledger statement' USING ERRCODE = '22023';
      END IF;
      results := results || jsonb_build_array(jsonb_build_object(
        'columns',COALESCE(column_names,'[]'::jsonb),'binary_columns',binary_names,
        'rows',result_rows,'rowcount',affected,'mutation',mutation));
    END LOOP;
    IF p_preview THEN
      -- The result variable survives this subtransaction rollback. No preview
      -- insert/update/DDL is durable; commit submits the complete journal once.
      RAISE EXCEPTION 'openbell preview rollback' USING ERRCODE='OB001';
    END IF;
  EXCEPTION WHEN SQLSTATE 'OB001' THEN
    IF NOT p_preview THEN RAISE; END IF;
  END;
  RETURN jsonb_build_object('results',results,'preview',p_preview);
END;
$function$;

REVOKE ALL ON FUNCTION public.openbell_ledger_transaction_v1(text,jsonb,boolean) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.openbell_ledger_transaction_v1(text,jsonb,boolean) TO service_role;

-- This helper reproduces the two existing ISO-timestamp ordering queries.
CREATE OR REPLACE FUNCTION openbell_module2_intraday.julianday(value text)
RETURNS double precision LANGUAGE sql IMMUTABLE STRICT
SET search_path=pg_catalog
AS $$ SELECT extract(epoch FROM value::timestamptz)/86400.0+2440587.5 $$;
GRANT EXECUTE ON FUNCTION openbell_module2_intraday.julianday(text) TO service_role;
REVOKE ALL ON FUNCTION openbell_module2_intraday.julianday(text) FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.openbell_prevent_history_mutation_v1()
RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER
SET search_path=pg_catalog
AS $$ BEGIN RAISE EXCEPTION 'Saved prediction/source history is immutable' USING ERRCODE='23000'; END $$;
REVOKE ALL ON FUNCTION public.openbell_prevent_history_mutation_v1() FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.openbell_prevent_history_mutation_v1() TO service_role;
