-- get_videos_feed / get_videos_grid: mgs ソースを performer 条件から免除
-- MGS動画は出演者情報がない作品が多いため FANZA_AMATEUR と同様に扱う

DROP FUNCTION IF EXISTS public.get_videos_feed(int);
CREATE OR REPLACE FUNCTION public.get_videos_feed(page_limit int DEFAULT 20)
RETURNS TABLE(
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  thumbnail_vertical_url text,
  sample_video_url text,
  preview_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  total_count int;
  start_offset int := 0;
BEGIN
  SELECT count(*) INTO total_count
  FROM public.videos v
  WHERE v.sample_video_url IS NOT NULL
    AND v.source NOT IN ('FANZA_ANIME')
    AND (
      v.source IN ('FANZA_AMATEUR', 'mgs')
      OR EXISTS (SELECT 1 FROM public.video_performers vp WHERE vp.video_id = v.id)
    )
    AND NOT EXISTS (
      SELECT 1 FROM public.user_video_decisions uvd
      WHERE uvd.video_id = v.id
        AND auth.uid() IS NOT NULL AND uvd.user_id = auth.uid()
    );

  IF total_count > page_limit THEN
    start_offset := floor(random() * (total_count - page_limit))::int;
  END IF;

  RETURN QUERY
  SELECT
    v.id, v.title, v.description, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url,
    v.sample_video_url, v.preview_video_url,
    coalesce(v.affiliate_url, v.product_url) AS product_url,
    v.product_released_at,
    coalesce((
      SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name)
      FROM public.video_performers vp JOIN public.performers p ON p.id = vp.performer_id
      WHERE vp.video_id = v.id
    ), '[]'::jsonb) AS performers,
    coalesce((
      SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name)
      FROM public.video_tags vt JOIN public.tags t ON t.id = vt.tag_id
      LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
      WHERE vt.video_id = v.id AND coalesce(tg.show_in_ui, true)
    ), '[]'::jsonb) AS tags
  FROM public.videos v
  WHERE v.id IN (
    SELECT v2.id FROM public.videos v2
    WHERE v2.sample_video_url IS NOT NULL
      AND v2.source NOT IN ('FANZA_ANIME')
      AND (
        v2.source IN ('FANZA_AMATEUR', 'mgs')
        OR EXISTS (SELECT 1 FROM public.video_performers vp2 WHERE vp2.video_id = v2.id)
      )
      AND NOT EXISTS (
        SELECT 1 FROM public.user_video_decisions uvd2
        WHERE uvd2.video_id = v2.id
          AND auth.uid() IS NOT NULL AND uvd2.user_id = auth.uid()
      )
    ORDER BY v2.id
    OFFSET start_offset LIMIT page_limit
  )
  ORDER BY (v.sample_video_url IS NULL), v.id;
END;
$$;
