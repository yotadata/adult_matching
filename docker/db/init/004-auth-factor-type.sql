-- Ensure enum types used by GoTrue migrations exist to avoid partial-DO early exit
DO $$
BEGIN
  -- factor_type
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'auth' AND t.typname = 'factor_type'
  ) THEN
    CREATE TYPE auth.factor_type AS ENUM ('totp', 'webauthn');
  END IF;

  -- factor_status
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'auth' AND t.typname = 'factor_status'
  ) THEN
    CREATE TYPE auth.factor_status AS ENUM ('unverified', 'verified');
  END IF;

  -- aal_level
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'auth' AND t.typname = 'aal_level'
  ) THEN
    CREATE TYPE auth.aal_level AS ENUM ('aal1', 'aal2', 'aal3');
  END IF;
END$$;
