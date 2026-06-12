-- get_videos_feed のパフォーマンス改善
-- 問題: auth.uid() が行ごとに評価される + excluded_tags の相関サブクエリが遅い
-- 修正: uid と除外動画IDを事前計算してキャッシュ

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
  tags jsonb,
  source text,
  image_urls text[]
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  current_uid uuid;
  excluded_tag_ids uuid[];
  total_count int;
  start_offset int := 0;
BEGIN
  current_uid := auth.uid();

  -- 除外タグIDを一度だけ取得
  IF current_uid IS NOT NULL THEN
    SELECT array_agg(uet.tag_id) INTO excluded_tag_ids
    FROM public.user_excluded_tags uet
    WHERE uet.user_id = current_uid;
  END IF;

  SELECT count(*) INTO total_count
  FROM public.videos v
  WHERE v.sample_video_url IS NOT NULL
    AND v.source NOT IN ('FANZA_ANIME')
    AND (
      v.source IN ('FANZA_AMATEUR', 'mgs')
      OR EXISTS (SELECT 1 FROM public.video_performers vp WHERE vp.video_id = v.id)
    )
    AND (current_uid IS NULL OR NOT EXISTS (
      SELECT 1 FROM public.user_video_decisions uvd
      WHERE uvd.video_id = v.id AND uvd.user_id = current_uid
    ))
    AND (
      excluded_tag_ids IS NULL
      OR NOT EXISTS (
        SELECT 1 FROM public.video_tags vt
        WHERE vt.video_id = v.id AND vt.tag_id = ANY(excluded_tag_ids)
      )
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
    ), '[]'::jsonb) AS tags,
    v.source,
    v.image_urls
  FROM public.videos v
  WHERE v.id IN (
    SELECT v2.id FROM public.videos v2
    WHERE v2.sample_video_url IS NOT NULL
      AND v2.source NOT IN ('FANZA_ANIME')
      AND (
        v2.source IN ('FANZA_AMATEUR', 'mgs')
        OR EXISTS (SELECT 1 FROM public.video_performers vp2 WHERE vp2.video_id = v2.id)
      )
      AND (current_uid IS NULL OR NOT EXISTS (
        SELECT 1 FROM public.user_video_decisions uvd2
        WHERE uvd2.video_id = v2.id AND uvd2.user_id = current_uid
      ))
      AND (
        excluded_tag_ids IS NULL
        OR NOT EXISTS (
          SELECT 1 FROM public.video_tags vt2
          WHERE vt2.video_id = v2.id AND vt2.tag_id = ANY(excluded_tag_ids)
        )
      )
    ORDER BY v2.id
    OFFSET start_offset LIMIT page_limit
  )
  ORDER BY (v.sample_video_url IS NULL), v.id;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_videos_feed(int) TO anon, authenticated;
