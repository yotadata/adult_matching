-- Remove legacy profiles table and replace with view sourcing from auth.users metadata
DROP TABLE IF EXISTS public.profiles;
DROP VIEW IF EXISTS public.profiles;

DO $$
DECLARE
    metadata_col text;
    view_sql text;
BEGIN
    SELECT column_name
    INTO metadata_col
    FROM information_schema.columns
    WHERE table_schema = 'auth'
      AND table_name = 'users'
      AND column_name IN ('raw_user_meta_data', 'user_metadata', 'raw_user_metadata')
    ORDER BY CASE column_name
        WHEN 'raw_user_meta_data' THEN 1
        WHEN 'user_metadata' THEN 2
        ELSE 3
    END
    LIMIT 1;

    IF metadata_col IS NOT NULL THEN
        view_sql := format($f$
            CREATE VIEW public.profiles AS
            SELECT
              u.id AS user_id,
              COALESCE((u.%I->>'display_name')::text, '') AS display_name,
              u.created_at
            FROM auth.users u;
        $f$, metadata_col);
    ELSE
        view_sql := $f$
            CREATE VIEW public.profiles AS
            SELECT
              u.id AS user_id,
              ''::text AS display_name,
              u.created_at
            FROM auth.users u;
        $f$;
    END IF;

    EXECUTE view_sql;
END
$$;

GRANT SELECT ON public.profiles TO anon, authenticated;
