-- DEV-ONLY: Open RLS and privileges in public schema so anon/authenticated can read/write everything
-- This is intended for local development. Do NOT use in production.

DO $$
DECLARE rec record;
BEGIN
  -- Broad privileges on existing objects
  EXECUTE 'GRANT USAGE ON SCHEMA public TO anon, authenticated';
  EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO anon, authenticated';
  EXECUTE 'GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated';

  -- Also allow reading auth schema (dev only): needed for functions using auth.* tables or auth.uid()
  EXECUTE 'GRANT USAGE ON SCHEMA auth TO anon, authenticated';
  EXECUTE 'GRANT SELECT ON ALL TABLES IN SCHEMA auth TO anon, authenticated';

  -- Default privileges for future objects in public
  EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO anon, authenticated';
  EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO anon, authenticated';

  -- Default privileges for future objects in auth (read-only)
  EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA auth GRANT SELECT ON TABLES TO anon, authenticated';

  -- Enable RLS and create permissive policies for all public tables
  FOR rec IN
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
  LOOP
    EXECUTE format('ALTER TABLE %I.%I ENABLE ROW LEVEL SECURITY', rec.table_schema, rec.table_name);

    IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname=rec.table_schema AND tablename=rec.table_name AND policyname='all_anon_select'
    ) THEN
      EXECUTE format('CREATE POLICY all_anon_select ON %I.%I FOR SELECT TO anon, authenticated USING (true)', rec.table_schema, rec.table_name);
    END IF;

    IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname=rec.table_schema AND tablename=rec.table_name AND policyname='all_anon_insert'
    ) THEN
      EXECUTE format('CREATE POLICY all_anon_insert ON %I.%I FOR INSERT TO anon, authenticated WITH CHECK (true)', rec.table_schema, rec.table_name);
    END IF;

    IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname=rec.table_schema AND tablename=rec.table_name AND policyname='all_anon_update'
    ) THEN
      EXECUTE format('CREATE POLICY all_anon_update ON %I.%I FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true)', rec.table_schema, rec.table_name);
    END IF;

    IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname=rec.table_schema AND tablename=rec.table_name AND policyname='all_anon_delete'
    ) THEN
      EXECUTE format('CREATE POLICY all_anon_delete ON %I.%I FOR DELETE TO anon, authenticated USING (true)', rec.table_schema, rec.table_name);
    END IF;
  END LOOP;
END $$;
