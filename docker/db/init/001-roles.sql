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
