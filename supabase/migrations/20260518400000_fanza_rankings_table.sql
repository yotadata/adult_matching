-- fanza_rank を videos テーブルから切り出し、fanza_rankings テーブルへ移行

-- 1. 新テーブル作成
CREATE TABLE IF NOT EXISTS public.fanza_rankings (
  source      text        NOT NULL,
  rank        int         NOT NULL,
  video_id    uuid        REFERENCES public.videos(id) ON DELETE SET NULL,
  external_id text        NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (source, rank)
);

CREATE INDEX IF NOT EXISTS idx_fanza_rankings_video_id ON public.fanza_rankings (video_id)
  WHERE video_id IS NOT NULL;

-- anon/authenticated から読み取り可能にする（get_popular_videos は security definer なので不要だが念のため）
ALTER TABLE public.fanza_rankings ENABLE ROW LEVEL SECURITY;
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE tablename = 'fanza_rankings' AND policyname = 'fanza_rankings_readable'
  ) THEN
    CREATE POLICY "fanza_rankings_readable" ON public.fanza_rankings FOR SELECT USING (true);
  END IF;
END $$;
GRANT SELECT ON public.fanza_rankings TO anon, authenticated;
GRANT INSERT, UPDATE, DELETE ON public.fanza_rankings TO service_role;

-- 2. videos テーブルから fanza_rank カラムを削除
DROP INDEX IF EXISTS idx_videos_fanza_rank;
ALTER TABLE public.videos DROP COLUMN IF EXISTS fanza_rank;
ALTER TABLE public.videos DROP COLUMN IF EXISTS fanza_rank_updated_at;

-- 3. get_popular_videos: fanza_rankings テーブルを JOIN
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
  best_rank AS (
    SELECT fr.video_id, MIN(fr.rank) AS rank
    FROM public.fanza_rankings fr
    WHERE fr.video_id IS NOT NULL
    GROUP BY fr.video_id
  ),
  ranked AS (
    SELECT
      v.id,
      COALESCE(ul.total_likes, 0) AS like_score,
      CASE WHEN br.rank IS NOT NULL
        THEN (501 - LEAST(br.rank, 500))::double precision / 500.0
        ELSE 0
      END AS rank_score
    FROM public.videos v
    LEFT JOIN user_likes ul ON ul.video_id = v.id
    LEFT JOIN best_rank br ON br.video_id = v.id
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
    CASE WHEN r.like_score > 0 THEN r.like_score ELSE r.rank_score END AS score
  FROM ranked r
  JOIN public.videos v ON v.id = r.id
  ORDER BY score DESC NULLS LAST, v.product_released_at DESC
  LIMIT limit_count;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_popular_videos(uuid, int, int) TO anon, authenticated;
