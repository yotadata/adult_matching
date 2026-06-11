-- キュレーター機能: ユーザープロフィール拡張テーブル
CREATE TABLE public.user_profiles (
  user_id              uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  avatar_url           text,
  bio                  text,
  x_url                text,
  affiliate_fanza_id   text,
  affiliate_fc2_id     text,
  affiliate_mgs_id     text,
  updated_at           timestamptz DEFAULT now()
);

ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "user_profiles_public_read"
  ON public.user_profiles FOR SELECT USING (true);

CREATE POLICY "user_profiles_owner_write"
  ON public.user_profiles FOR ALL USING (auth.uid() = user_id);

-- profiles ビューを再作成して user_profiles を JOIN
DROP VIEW IF EXISTS public.profiles;
CREATE VIEW public.profiles AS
SELECT
  u.id AS user_id,
  COALESCE(u.raw_user_meta_data->>'display_name', '') AS display_name,
  u.raw_user_meta_data->>'username' AS username,
  u.created_at,
  up.avatar_url,
  up.bio,
  up.x_url,
  up.affiliate_fanza_id,
  up.affiliate_fc2_id,
  up.affiliate_mgs_id
FROM auth.users u
LEFT JOIN public.user_profiles up ON up.user_id = u.id;

-- プロフィールアップサート用 RPC
CREATE OR REPLACE FUNCTION public.upsert_user_profile(
  p_avatar_url         text DEFAULT NULL,
  p_bio                text DEFAULT NULL,
  p_x_url              text DEFAULT NULL,
  p_affiliate_fanza_id text DEFAULT NULL,
  p_affiliate_fc2_id   text DEFAULT NULL,
  p_affiliate_mgs_id   text DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  INSERT INTO public.user_profiles (
    user_id, avatar_url, bio, x_url,
    affiliate_fanza_id, affiliate_fc2_id, affiliate_mgs_id, updated_at
  )
  VALUES (
    auth.uid(), p_avatar_url, p_bio, p_x_url,
    p_affiliate_fanza_id, p_affiliate_fc2_id, p_affiliate_mgs_id, now()
  )
  ON CONFLICT (user_id) DO UPDATE SET
    avatar_url           = COALESCE(EXCLUDED.avatar_url, user_profiles.avatar_url),
    bio                  = COALESCE(EXCLUDED.bio, user_profiles.bio),
    x_url                = COALESCE(EXCLUDED.x_url, user_profiles.x_url),
    affiliate_fanza_id   = COALESCE(EXCLUDED.affiliate_fanza_id, user_profiles.affiliate_fanza_id),
    affiliate_fc2_id     = COALESCE(EXCLUDED.affiliate_fc2_id, user_profiles.affiliate_fc2_id),
    affiliate_mgs_id     = COALESCE(EXCLUDED.affiliate_mgs_id, user_profiles.affiliate_mgs_id),
    updated_at           = now();
END;
$$;

GRANT EXECUTE ON FUNCTION public.upsert_user_profile TO authenticated;
