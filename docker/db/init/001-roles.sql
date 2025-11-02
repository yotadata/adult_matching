-- Create minimal roles expected by Supabase services
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    CREATE ROLE anon NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
    CREATE ROLE authenticated NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    CREATE ROLE service_role NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_admin') THEN
    CREATE ROLE supabase_admin NOLOGIN;
  END IF;
END$$;

-- Allow postgres (and admin) to read server-side files for Supabase extension hooks
DO $$
BEGIN
  -- Predefined roles exist in Postgres; grant membership if available
  BEGIN
    EXECUTE 'GRANT pg_read_server_files TO postgres';
  EXCEPTION WHEN undefined_object THEN
    -- Older versions might not have the predefined role; ignore
    NULL;
  END;
  BEGIN
    EXECUTE 'GRANT pg_read_server_files TO supabase_admin';
  EXCEPTION WHEN undefined_object THEN
    NULL;
  END;
END$$;

-- Allow Supabase's extension bootstrap scripts to read custom SQL hooks (required for pgvector)
DO $$
DECLARE
  fn RECORD;
  supabase_admin_exists BOOLEAN := EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'supabase_admin'
  );
BEGIN
  FOR fn IN
    SELECT format('%I.%I(%s)', n.nspname, p.proname, pg_get_function_identity_arguments(p.oid)) AS signature
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'pg_catalog'
      AND p.proname = 'pg_read_file'
  LOOP
    EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO postgres', fn.signature);
    IF supabase_admin_exists THEN
      EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO supabase_admin', fn.signature);
    END IF;
  END LOOP;
END$$;

-- For local development, make supabase_admin superuser so extensions can be created
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_admin') THEN
    EXECUTE 'ALTER ROLE supabase_admin SUPERUSER';
  END IF;
END$$;
