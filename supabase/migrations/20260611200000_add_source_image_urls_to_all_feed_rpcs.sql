-- get_videos_feed / get_popular_videos / get_videos_by_tags / explore_videos に
-- source, image_urls を追加（FANZA_AMATEUR サムネイル高画質化）

-- get_videos_feed
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

-- get_popular_videos
DROP FUNCTION IF EXISTS public.get_popular_videos(int, int, uuid);
CREATE OR REPLACE FUNCTION public.get_popular_videos(
  limit_count int DEFAULT 20,
  lookback_days int DEFAULT 7,
  user_uuid uuid DEFAULT NULL
)
RETURNS TABLE(
  id uuid, title text, description text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, product_url text, product_released_at timestamptz,
  performers jsonb, tags jsonb, score double precision,
  source text, image_urls text[]
)
LANGUAGE plpgsql SECURITY DEFINER AS $$
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
    SELECT v.id,
      COALESCE(ul.total_likes, 0) AS like_score,
      CASE WHEN br.rank IS NOT NULL
        THEN (501 - LEAST(br.rank, 500))::double precision / 500.0
        ELSE 0
      END AS rank_score
    FROM public.videos v
    LEFT JOIN user_likes ul ON ul.video_id = v.id
    LEFT JOIN best_rank br ON br.video_id = v.id
    WHERE v.sample_video_url IS NOT NULL
      AND v.source NOT IN ('FANZA_ANIME')
      AND (v.has_performer = true OR v.source = 'FANZA_AMATEUR')
      AND (user_uuid IS NULL OR NOT EXISTS (
        SELECT 1 FROM public.user_video_decisions uvd
        WHERE uvd.user_id = user_uuid AND uvd.video_id = v.id
      ))
  )
  SELECT
    v.id, v.title, v.description, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url, v.sample_video_url,
    coalesce(v.affiliate_url, v.product_url), v.product_released_at,
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
    (r.like_score + r.rank_score) AS score,
    v.source,
    v.image_urls
  FROM ranked r
  JOIN public.videos v ON v.id = r.id
  ORDER BY score DESC
  LIMIT limit_count;
END;
$$;

-- get_videos_by_tags
DROP FUNCTION IF EXISTS public.get_videos_by_tags(uuid[], uuid[], int);
CREATE OR REPLACE FUNCTION public.get_videos_by_tags(
  tag_ids uuid[],
  exclude_ids uuid[] DEFAULT '{}',
  p_limit int DEFAULT 20
)
RETURNS TABLE(
  id uuid, title text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, product_url text, product_released_at timestamptz,
  performers jsonb, tags jsonb,
  source text, image_urls text[]
)
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  RETURN QUERY
  SELECT v.id, v.title, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url, v.sample_video_url,
    coalesce(v.affiliate_url, v.product_url), v.product_released_at,
    coalesce((
      SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name))
      FROM public.video_performers vp JOIN public.performers p ON p.id = vp.performer_id
      WHERE vp.video_id = v.id
    ), '[]'::jsonb),
    coalesce((
      SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name))
      FROM public.video_tags vt JOIN public.tags t ON t.id = vt.tag_id
      LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
      WHERE vt.video_id = v.id AND coalesce(tg.show_in_ui, true)
    ), '[]'::jsonb),
    v.source,
    v.image_urls
  FROM public.videos v
  WHERE v.sample_video_url IS NOT NULL
    AND v.source NOT IN ('FANZA_ANIME')
    AND (v.has_performer = true OR v.source = 'FANZA_AMATEUR')
    AND (array_length(exclude_ids, 1) IS NULL OR v.id != ALL(exclude_ids))
    AND EXISTS (SELECT 1 FROM public.video_tags vt WHERE vt.video_id = v.id AND vt.tag_id = ANY(tag_ids))
  ORDER BY random()
  LIMIT p_limit;
END;
$$;

-- explore_videos
CREATE OR REPLACE FUNCTION public.explore_videos(
  p_query text DEFAULT NULL,
  p_tag_id uuid DEFAULT NULL,
  p_performer_id uuid DEFAULT NULL,
  p_limit int DEFAULT 20,
  p_offset int DEFAULT 0
)
RETURNS json
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  IF p_query IS NULL AND p_tag_id IS NULL AND p_performer_id IS NULL THEN
    RETURN (
      SELECT json_agg(row_to_json(v))
      FROM (
        SELECT vi.id, vi.title, vi.external_id,
          vi.thumbnail_url, vi.thumbnail_vertical_url,
          vi.product_url, vi.distribution_code, vi.product_released_at,
          vi.source, vi.image_urls
        FROM public.videos vi
        WHERE vi.external_id IS NOT NULL
          AND vi.sample_video_url IS NOT NULL
          AND (vi.has_performer = true OR vi.source = 'FANZA_AMATEUR')
          AND vi.source NOT IN ('FANZA_ANIME')
        ORDER BY vi.product_released_at DESC NULLS LAST
        LIMIT p_limit OFFSET p_offset
      ) v
    );
  END IF;

  IF p_tag_id IS NOT NULL AND p_query IS NULL AND p_performer_id IS NULL THEN
    RETURN (
      SELECT json_agg(row_to_json(v))
      FROM (
        SELECT vi.id, vi.title, vi.external_id,
          vi.thumbnail_url, vi.thumbnail_vertical_url,
          vi.product_url, vi.distribution_code, vi.product_released_at,
          vi.source, vi.image_urls
        FROM public.video_tags vt
        JOIN public.videos vi ON vi.id = vt.video_id
        LEFT JOIN public.fanza_rankings fr ON fr.video_id = vi.id
        WHERE vt.tag_id = p_tag_id
          AND vi.external_id IS NOT NULL
          AND vi.sample_video_url IS NOT NULL
          AND (vi.has_performer = true OR vi.source = 'FANZA_AMATEUR')
          AND vi.source NOT IN ('FANZA_ANIME')
        ORDER BY fr.rank ASC NULLS LAST, vi.product_released_at DESC NULLS LAST
        LIMIT p_limit OFFSET p_offset
      ) v
    );
  END IF;

  RETURN (
    SELECT json_agg(row_to_json(v))
    FROM (
      SELECT vi.id, vi.title, vi.external_id,
        vi.thumbnail_url, vi.thumbnail_vertical_url,
        vi.product_url, vi.distribution_code, vi.product_released_at,
        vi.source, vi.image_urls
      FROM public.videos vi
      LEFT JOIN public.fanza_rankings fr ON fr.video_id = vi.id
      WHERE vi.external_id IS NOT NULL
        AND (p_query IS NULL
          OR vi.title ILIKE '%' || p_query || '%'
          OR vi.distribution_code ILIKE '%' || p_query || '%')
        AND (p_tag_id IS NULL
          OR EXISTS (SELECT 1 FROM public.video_tags vt WHERE vt.video_id = vi.id AND vt.tag_id = p_tag_id))
        AND (p_performer_id IS NULL
          OR EXISTS (SELECT 1 FROM public.video_performers vp WHERE vp.video_id = vi.id AND vp.performer_id = p_performer_id))
      ORDER BY
        CASE WHEN p_tag_id IS NOT NULL OR p_query IS NOT NULL THEN fr.rank END ASC NULLS LAST,
        vi.product_released_at DESC NULLS LAST
      LIMIT p_limit OFFSET p_offset
    ) v
  );
END;
$$;
