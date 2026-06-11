-- videos テーブルに has_performer フラグを追加
-- get_videos_feed: has_performer + TABLESAMPLE で高速化
ALTER TABLE public.videos ADD COLUMN IF NOT EXISTS has_performer boolean NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS videos_feed_eligible_idx
  ON public.videos (id)
  WHERE sample_video_url IS NOT NULL
    AND source NOT IN ('FANZA_ANIME')
    AND (has_performer = true OR source = 'FANZA_AMATEUR');

DROP FUNCTION IF EXISTS public.get_videos_feed(int);
CREATE OR REPLACE FUNCTION public.get_videos_feed(page_limit int DEFAULT 20)
RETURNS TABLE(
  id uuid, title text, description text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, preview_video_url text,
  product_url text, product_released_at timestamptz,
  performers jsonb, tags jsonb,
  source text, image_urls text[]
)
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  RETURN QUERY
  SELECT
    v.id, v.title, v.description, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url,
    v.sample_video_url, v.preview_video_url,
    coalesce(v.affiliate_url, v.product_url),
    v.product_released_at,
    coalesce((
      SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name)
      FROM public.video_performers vp JOIN public.performers p ON p.id = vp.performer_id
      WHERE vp.video_id = v.id
    ), '[]'::jsonb),
    coalesce((
      SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name)
      FROM public.video_tags vt JOIN public.tags t ON t.id = vt.tag_id
      LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
      WHERE vt.video_id = v.id AND coalesce(tg.show_in_ui, true)
    ), '[]'::jsonb),
    v.source,
    v.image_urls
  FROM public.videos v TABLESAMPLE SYSTEM(5)
  WHERE v.sample_video_url IS NOT NULL
    AND v.source NOT IN ('FANZA_ANIME')
    AND (v.has_performer = true OR v.source = 'FANZA_AMATEUR')
    AND NOT EXISTS (
      SELECT 1 FROM public.user_video_decisions uvd
      WHERE uvd.video_id = v.id
        AND auth.uid() IS NOT NULL AND uvd.user_id = auth.uid()
    )
  ORDER BY random()
  LIMIT page_limit;
END;
$$;
