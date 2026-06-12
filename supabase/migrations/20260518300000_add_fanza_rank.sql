-- FANZA公式ランキング順位を保持するカラム
ALTER TABLE public.videos
  ADD COLUMN IF NOT EXISTS fanza_rank int,
  ADD COLUMN IF NOT EXISTS fanza_rank_updated_at timestamptz;

CREATE INDEX IF NOT EXISTS idx_videos_fanza_rank ON public.videos (fanza_rank ASC NULLS LAST)
  WHERE fanza_rank IS NOT NULL;

-- get_popular_videos: fanza_rank をフォールバックとして使用
DROP FUNCTION IF EXISTS public.get_popular_videos(uuid, int, int);
CREATE OR REPLACE FUNCTION public.get_popular_videos(
  user_uuid uuid DEFAULT NULL,
  limit_count int DEFAULT 20,
  lookback_days int DEFAULT 7
)
RETURNS TABLE (
  id uuid, title text, description text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, product_url text,
  product_released_at timestamptz,
  performers jsonb, tags jsonb,
  score double precision
)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  WITH user_likes AS (
    SELECT vpd.video_id, SUM(vpd.likes)::double precision AS total_likes
    FROM public.video_popularity_daily vpd
    WHERE vpd.d >= (NOW() - make_interval(days => lookback_days))
    GROUP BY vpd.video_id
  ),
  ranked AS (
    SELECT
      v.id,
      COALESCE(ul.total_likes, 0) AS like_score,
      -- fanza_rank は小さいほど良い → スコアに変換（上位500位以内を対象）
      CASE WHEN v.fanza_rank IS NOT NULL AND v.fanza_rank <= 500
        THEN (501 - v.fanza_rank)::double precision / 500.0
        ELSE 0
      END AS rank_score
    FROM public.videos v
    LEFT JOIN user_likes ul ON ul.video_id = v.id
    WHERE v.sample_video_url IS NOT NULL
      AND (user_uuid IS NULL OR NOT EXISTS (
        SELECT 1 FROM public.user_video_decisions uvd
        WHERE uvd.user_id = user_uuid AND uvd.video_id = v.id
      ))
  )
  SELECT
    v.id, v.title, v.description, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url, v.sample_video_url,
    COALESCE(v.affiliate_url, v.product_url) AS product_url,
    v.product_released_at,
    COALESCE((
      SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name)
      FROM public.video_performers vp JOIN public.performers p ON p.id = vp.performer_id
      WHERE vp.video_id = v.id
    ), '[]'::jsonb) AS performers,
    COALESCE((
      SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name)
      FROM public.video_tags vt JOIN public.tags t ON t.id = vt.tag_id
      LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
      WHERE vt.video_id = v.id AND COALESCE(tg.show_in_ui, TRUE)
    ), '[]'::jsonb) AS tags,
    -- like_score がなければ rank_score で代替
    CASE WHEN r.like_score > 0 THEN r.like_score ELSE r.rank_score END AS score
  FROM ranked r
  JOIN public.videos v ON v.id = r.id
  ORDER BY score DESC NULLS LAST, v.product_released_at DESC
  LIMIT limit_count;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_popular_videos(uuid, int, int) TO anon, authenticated;
