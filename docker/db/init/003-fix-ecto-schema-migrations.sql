-- Ensure Ecto-compatible schema_migrations table for Supabase Realtime
-- Context: Realtime (Ecto) expects public.schema_migrations(version, inserted_at)
-- If an existing table (e.g., created by other tools) lacks inserted_at, add it.

DO $$
DECLARE rec RECORD;
BEGIN
  -- Add inserted_at to any existing <schema>.schema_migrations that lacks it
  FOR rec IN
    SELECT table_schema
    FROM information_schema.tables
    WHERE table_name = 'schema_migrations'
  LOOP
    IF NOT EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = rec.table_schema
        AND table_name = 'schema_migrations'
        AND column_name = 'inserted_at'
    ) THEN
      EXECUTE format(
        'ALTER TABLE %I.schema_migrations ADD COLUMN inserted_at timestamp without time zone NOT NULL DEFAULT now()',
        rec.table_schema
      );
    END IF;
  END LOOP;
END$$;

-- Ensure public.schema_migrations exists with the expected columns for Ecto >= 3.13
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'schema_migrations'
  ) THEN
    CREATE TABLE public.schema_migrations (
      version      bigint PRIMARY KEY,
      inserted_at  timestamp without time zone NOT NULL DEFAULT now()
    );
  END IF;
END$$;
