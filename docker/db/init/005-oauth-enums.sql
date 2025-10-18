-- Ensure enum used by GoTrue OAuth migrations exists
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE n.nspname = 'auth' AND t.typname = 'code_challenge_method'
  ) THEN
    -- PKCE methods. GoTrue expects 'plain' and 's256'
    CREATE TYPE auth.code_challenge_method AS ENUM ('plain','s256');
  END IF;
END$$;

