-- Remove legacy profiles table and replace with view sourcing from auth.users metadata
DROP TABLE IF EXISTS public.profiles;
DROP VIEW IF EXISTS public.profiles;

CREATE VIEW public.profiles AS
SELECT
  u.id AS user_id,
  COALESCE((u.raw_user_meta_data->>'display_name')::text, '') AS display_name,
  u.created_at
FROM auth.users u;

GRANT SELECT ON public.profiles TO anon, authenticated;
