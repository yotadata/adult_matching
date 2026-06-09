-- profilesビューを再作成してusernameを追加
DROP VIEW IF EXISTS public.profiles;
CREATE VIEW public.profiles AS
SELECT
  id AS user_id,
  COALESCE(raw_user_meta_data->>'display_name', '') AS display_name,
  raw_user_meta_data->>'username' AS username,
  created_at
FROM auth.users;
