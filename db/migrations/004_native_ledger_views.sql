-- Production queries use typed PostgreSQL views over the existing Supabase
-- records. This avoids duplicating hundreds of MB on the free project.
CREATE OR REPLACE FUNCTION public.openbell_write_ledger_view_v1()
RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER SET search_path=pg_catalog
AS $function$
DECLARE
  spec jsonb := TG_ARGV[0]::jsonb;
  row_data jsonb := to_jsonb(NEW);
  old_data jsonb;
  key_data jsonb := '{}'::jsonb;
  key_text text;
  field text;
  unique_group jsonb;
  duplicate boolean;
  ignored boolean := COALESCE(current_setting('openbell.ignore_conflict',true),'false')='true';
  content_hash text;
BEGIN
  IF current_user <> 'service_role' THEN RAISE EXCEPTION 'Service role required' USING ERRCODE='42501'; END IF;
  IF TG_OP='DELETE' OR (TG_OP='UPDATE' AND (spec->>'immutable')::boolean) THEN
    RAISE EXCEPTION 'Saved history is immutable' USING ERRCODE='23000';
  END IF;
  FOR field IN SELECT jsonb_array_elements_text(spec->'notnull') LOOP
    IF row_data->field IS NULL OR row_data->field='null'::jsonb THEN
      RAISE EXCEPTION 'Required ledger field missing' USING ERRCODE='23502';
    END IF;
  END LOOP;
  FOR field IN SELECT jsonb_array_elements_text(spec->'binary') LOOP
    IF row_data->field IS NOT NULL AND row_data->field<>'null'::jsonb THEN
      row_data := jsonb_set(row_data,ARRAY[field],jsonb_build_object('__capstone_type__','bytes',
        'base64',replace(encode(decode(substr(row_data->>field,3),'hex'),'base64'),E'\n','')));
    END IF;
  END LOOP;
  FOR field IN SELECT jsonb_array_elements_text(spec->'pk') LOOP
    IF row_data->field IS NULL OR row_data->field='null'::jsonb THEN
      RAISE EXCEPTION 'Ledger primary key missing' USING ERRCODE='23502';
    END IF;
    key_data := key_data || jsonb_build_object(field,row_data->field);
  END LOOP;
  SELECT '{'||string_agg(to_json(key)::text||':'||value::text,',' ORDER BY key COLLATE "C")||'}'
    INTO key_text FROM jsonb_each(key_data);
  IF TG_OP='UPDATE' THEN
    old_data := to_jsonb(OLD);
    FOR field IN SELECT jsonb_array_elements_text(spec->'pk') LOOP
      IF old_data->field IS DISTINCT FROM row_data->field THEN
        RAISE EXCEPTION 'Ledger identity cannot change' USING ERRCODE='23000';
      END IF;
    END LOOP;
  ELSE
    -- All native transactions take this same ledger-scoped lock. Check each
    -- original SQLite UNIQUE constraint before inserting the JSON record.
    FOR unique_group IN SELECT value FROM jsonb_array_elements(spec->'unique') LOOP
      IF NOT EXISTS(SELECT 1 FROM jsonb_array_elements_text(unique_group) k WHERE row_data->k IS NULL OR row_data->k='null'::jsonb) THEN
        SELECT EXISTS(SELECT 1 FROM public.capstone_sqlite_mirror m
          WHERE m.db_name=spec->>'database' AND m.table_name=spec->>'table'
          AND NOT EXISTS(SELECT 1 FROM jsonb_array_elements_text(unique_group) k WHERE m.payload->k IS DISTINCT FROM row_data->k))
          INTO duplicate;
        IF duplicate THEN
          IF ignored THEN RETURN NULL; END IF;
          RAISE EXCEPTION 'Duplicate ledger identity' USING ERRCODE='23505';
        END IF;
      END IF;
    END LOOP;
  END IF;
  content_hash := 'pgjson:'||encode(sha256(convert_to(row_data::text,'UTF8')),'hex');
  IF TG_OP='INSERT' THEN
    INSERT INTO public.capstone_sqlite_mirror(db_name,table_name,row_key,payload,row_sha256)
      VALUES(spec->>'database',spec->>'table',key_text,row_data,content_hash);
  ELSE
    UPDATE public.capstone_sqlite_mirror SET payload=row_data,row_sha256=content_hash
      WHERE db_name=spec->>'database' AND table_name=spec->>'table' AND row_key=key_text;
  END IF;
  RETURN NEW;
END;
$function$;
REVOKE ALL ON FUNCTION public.openbell_write_ledger_view_v1() FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.openbell_write_ledger_view_v1() TO service_role;

-- Updated transaction gateway supports only this project's private ledgers.
CREATE OR REPLACE FUNCTION public.openbell_ledger_transaction_v1(
  p_database text,p_statements jsonb,p_preview boolean DEFAULT false
) RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER
SET search_path=pg_catalog SET standard_conforming_strings=on
AS $function$
DECLARE
  namespace text;
  item jsonb;
  statement text;
  results jsonb := '[]'::jsonb;
  result_rows json;
  column_names jsonb;
  binary_names jsonb;
  affected bigint;
  mutation boolean;
  result jsonb;
BEGIN
  IF current_user<>'service_role' THEN RAISE EXCEPTION 'Service role required' USING ERRCODE='42501'; END IF;
  IF p_database NOT IN ('snapshots.db','module2_decisions.db','module2_intraday.db','module2_historical_news.db')
    OR jsonb_typeof(p_statements)<>'array' OR jsonb_array_length(p_statements)>2000 THEN
    RAISE EXCEPTION 'Invalid transaction' USING ERRCODE='22023';
  END IF;
  namespace := 'openbell_'||replace(p_database,'.db','');
  PERFORM set_config('search_path',format('%I,pg_catalog',namespace),true);
  PERFORM pg_advisory_xact_lock(hashtextextended('openbell:'||p_database,0));
  BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(p_statements) LOOP
      statement := CASE WHEN jsonb_typeof(item)='string' THEN item#>>'{}' ELSE item->>'sql' END;
      PERFORM set_config('openbell.ignore_conflict',
        CASE WHEN statement ~* ' ON CONFLICT DO NOTHING$' THEN 'true' ELSE 'false' END,true);
      statement := regexp_replace(statement,' ON CONFLICT DO NOTHING$','','i');
      column_names := '[]'::jsonb; binary_names := '[]'::jsonb;
      result_rows := '[]'::json; mutation := false; affected := 0;
      IF statement ~* '^(SELECT|WITH)\M' THEN
        EXECUTE 'CREATE TEMP TABLE __openbell_rpc_result ON COMMIT DROP AS '||statement;
        SELECT jsonb_agg(attname ORDER BY attnum),COALESCE(jsonb_agg(attname ORDER BY attnum) FILTER(WHERE atttypid=17),'[]'::jsonb)
          INTO column_names,binary_names FROM pg_attribute
          WHERE attrelid='pg_temp.__openbell_rpc_result'::regclass AND attnum>0 AND NOT attisdropped;
        EXECUTE 'SELECT COALESCE(json_agg(row_to_json(r)),''[]''::json),count(*) FROM pg_temp.__openbell_rpc_result r'
          INTO result_rows,affected;
        DROP TABLE pg_temp.__openbell_rpc_result;
      ELSIF statement ~* '^(INSERT|UPDATE)\M' THEN
        EXECUTE statement; GET DIAGNOSTICS affected=ROW_COUNT; mutation:=true;
      ELSIF statement ~* '^CREATE (TABLE|INDEX|UNIQUE INDEX|SEQUENCE|OR REPLACE VIEW|OR REPLACE TRIGGER)\M'
        OR statement ~* '^ALTER (TABLE|VIEW|SEQUENCE)\M' THEN
        EXECUTE statement;
      ELSE RAISE EXCEPTION 'Unsupported ledger statement' USING ERRCODE='22023';
      END IF;
      result:=jsonb_build_object('columns',COALESCE(column_names,'[]'::jsonb),'binary_columns',binary_names,
        'rows',result_rows,'rowcount',affected,'mutation',mutation);
      IF NOT p_preview AND item ? 'expected_rows' AND item->'expected_rows' IS DISTINCT FROM result->'rows' THEN
        RAISE EXCEPTION 'Ledger changed since read; retry transaction' USING ERRCODE='OB002';
      END IF;
      results:=results||jsonb_build_array(result);
    END LOOP;
    IF p_preview THEN RAISE EXCEPTION 'preview rollback' USING ERRCODE='OB001'; END IF;
  EXCEPTION WHEN SQLSTATE 'OB001' THEN IF NOT p_preview THEN RAISE; END IF;
  END;
  RETURN jsonb_build_object('results',results,'preview',p_preview);
END;
$function$;
REVOKE ALL ON FUNCTION public.openbell_ledger_transaction_v1(text,jsonb,boolean) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.openbell_ledger_transaction_v1(text,jsonb,boolean) TO service_role;
