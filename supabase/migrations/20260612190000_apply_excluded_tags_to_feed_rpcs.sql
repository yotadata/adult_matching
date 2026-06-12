-- 各フィードRPCにユーザーの除外タグフィルタを追加

-- get_videos_feed
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
    )
    AND NOT EXISTS (
      SELECT 1 FROM public.user_excluded_tags uet
      JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
      WHERE vt.video_id = v.id AND uet.user_id = auth.uid()
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
      AND NOT EXISTS (
        SELECT 1 FROM public.user_excluded_tags uet
        JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
        WHERE vt.video_id = v2.id AND uet.user_id = auth.uid()
      )
    ORDER BY v2.id
    OFFSET start_offset LIMIT page_limit
  )
  ORDER BY (v.sample_video_url IS NULL), v.id;
END;
$$;

-- get_videos_recommendations
CREATE OR REPLACE FUNCTION public.get_videos_recommendations(user_uuid uuid, page_limit int DEFAULT 20)
RETURNS TABLE (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  thumbnail_vertical_url text,
  sample_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  score double precision,
  model_version text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  user_vec halfvec(128);
  approx_vec halfvec(128);
  latest_version text;
BEGIN
  SELECT avg(ve.embedding)::halfvec(128)
    INTO approx_vec
  FROM (
    SELECT uvd.video_id
    FROM public.user_video_decisions uvd
    WHERE uvd.user_id = user_uuid
      AND uvd.decision_type = 'like'
    ORDER BY uvd.created_at DESC
    LIMIT 200
  ) recent_likes
  JOIN public.video_embeddings ve ON ve.video_id = recent_likes.video_id;

  IF approx_vec IS NULL THEN
    SELECT ue.embedding INTO user_vec
    FROM public.user_embeddings ue
    WHERE ue.user_id = user_uuid;
  END IF;

  user_vec := coalesce(approx_vec, user_vec);

  IF user_vec IS NULL THEN
    RETURN;
  END IF;

  SELECT ve.model_version
    INTO latest_version
  FROM public.video_embeddings ve
  WHERE ve.model_version IS NOT NULL
  ORDER BY ve.updated_at DESC
  LIMIT 1;

  RETURN QUERY
  SELECT
    v.id,
    v.title,
    v.description,
    v.external_id,
    v.thumbnail_url,
    v.thumbnail_vertical_url,
    v.sample_video_url,
    coalesce(v.affiliate_url, v.product_url) AS product_url,
    v.product_released_at,
    coalesce(
      (
        SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name)
        FROM public.video_performers vp
        JOIN public.performers p ON p.id = vp.performer_id
        WHERE vp.video_id = v.id
      ), '[]'::jsonb
    ) AS performers,
    coalesce(
      (
        SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name)
        FROM public.video_tags vt
        JOIN public.tags t ON t.id = vt.tag_id
        LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
        WHERE vt.video_id = v.id
          AND coalesce(tg.show_in_ui, true)
      ), '[]'::jsonb
    ) AS tags,
    1 - (ve.embedding <-> user_vec) AS score,
    ve.model_version
  FROM public.video_embeddings ve
  JOIN public.videos v ON v.id = ve.video_id
  WHERE v.sample_video_url IS NOT NULL
    AND (latest_version IS NULL OR ve.model_version = latest_version)
    AND NOT EXISTS (
      SELECT 1 FROM public.user_video_decisions uvd
      WHERE uvd.video_id = v.id
        AND uvd.user_id = user_uuid
    )
    AND NOT EXISTS (
      SELECT 1 FROM public.user_excluded_tags uet
      JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
      WHERE vt.video_id = v.id AND uet.user_id = user_uuid
    )
  ORDER BY ve.embedding <-> user_vec
  LIMIT page_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_videos_recommendations(uuid, int) TO anon, authenticated;

-- get_popular_videos
DROP FUNCTION IF EXISTS public.get_popular_videos(uuid, int, int);
CREATE OR REPLACE FUNCTION public.get_popular_videos(user_uuid uuid DEFAULT NULL, limit_count int DEFAULT 20, lookback_days int DEFAULT 7)
RETURNS TABLE (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  score double precision
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  pop_view_exists boolean;
BEGIN
  SELECT exists (
    SELECT 1
    FROM pg_matviews
    WHERE schemaname = 'public'
      AND matviewname = 'video_popularity_daily'
  ) INTO pop_view_exists;

  IF NOT pop_view_exists THEN
    RETURN;
  END IF;

  BEGIN
    RETURN QUERY
    WITH popularity AS (
      SELECT vpd.video_id,
             sum(vpd.likes)::double precision AS total_likes
      FROM public.video_popularity_daily vpd
      WHERE vpd.d >= (now() - make_interval(days => lookback_days))
      GROUP BY vpd.video_id
    )
    SELECT
      v.id,
      v.title,
      v.description,
      v.external_id,
      v.thumbnail_url,
      v.sample_video_url,
      v.product_released_at,
      coalesce(
        (
          SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name)
          FROM public.video_performers vp
          JOIN public.performers p ON p.id = vp.performer_id
          WHERE vp.video_id = v.id
        ), '[]'::jsonb
      ) AS performers,
      coalesce(
        (
          SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name)
          FROM public.video_tags vt
          JOIN public.tags t ON t.id = vt.tag_id
          WHERE vt.video_id = v.id
        ), '[]'::jsonb
      ) AS tags,
      pop.total_likes AS score
    FROM popularity pop
    JOIN public.videos v ON v.id = pop.video_id
    WHERE v.sample_video_url IS NOT NULL
      AND (
        user_uuid IS NULL
        OR NOT EXISTS (
          SELECT 1 FROM public.user_video_decisions uvd
          WHERE uvd.user_id = user_uuid AND uvd.video_id = v.id
        )
      )
      AND NOT EXISTS (
        SELECT 1 FROM public.user_excluded_tags uet
        JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
        WHERE vt.video_id = v.id
          AND uet.user_id = coalesce(user_uuid, auth.uid())
      )
    ORDER BY pop.total_likes DESC NULLS LAST, v.product_released_at DESC
    LIMIT limit_count;
  EXCEPTION
    WHEN object_not_in_prerequisite_state THEN
      RETURN;
  END;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_popular_videos(uuid, int, int) TO anon, authenticated;

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
          AND NOT EXISTS (
            SELECT 1 FROM public.user_excluded_tags uet
            JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
            WHERE vt.video_id = vi.id AND uet.user_id = auth.uid()
          )
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
          AND NOT EXISTS (
            SELECT 1 FROM public.user_excluded_tags uet
            JOIN public.video_tags vt2 ON vt2.tag_id = uet.tag_id
            WHERE vt2.video_id = vi.id AND uet.user_id = auth.uid()
          )
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
        vi.product_url, vi.distribution_code, vi.product_released_at
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
        AND NOT EXISTS (
          SELECT 1 FROM public.user_excluded_tags uet
          JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
          WHERE vt.video_id = vi.id AND uet.user_id = auth.uid()
        )
      ORDER BY
        CASE WHEN p_tag_id IS NOT NULL OR p_query IS NOT NULL THEN fr.rank END ASC NULLS LAST,
        vi.product_released_at DESC NULLS LAST
      LIMIT p_limit OFFSET p_offset
    ) v
  );
END;
$$;
