-- インデックス追加とフィードRPCのバッチJOIN化
--
-- 問題: performers/tags の集計が動画ごとに個別サブクエリで実行される
--   90動画 × 2 = 180 回のサブクエリ → 全動画まとめて2回のJOINに削減
--
-- 追加インデックス:
--   video_tags(video_id)        - 動画ごとのタグ集計サブクエリ用
--   video_performers(video_id)  - 動画ごとの出演者集計サブクエリ用
--   user_excluded_tags(user_id) - 除外タグフィルタ用

-- インデックス追加
CREATE INDEX IF NOT EXISTS video_tags_video_id_idx
  ON public.video_tags (video_id);

CREATE INDEX IF NOT EXISTS video_performers_video_id_idx
  ON public.video_performers (video_id);

CREATE INDEX IF NOT EXISTS user_excluded_tags_user_id_idx
  ON public.user_excluded_tags (user_id);

-- videos_feed_eligible_idx を has_performer 条件を含むように更新
DROP INDEX IF EXISTS public.videos_feed_eligible_idx;
CREATE INDEX IF NOT EXISTS videos_feed_eligible_idx
  ON public.videos (id)
  WHERE sample_video_url IS NOT NULL
    AND source NOT IN ('FANZA_ANIME')
    AND (has_performer = true OR source IN ('FANZA_AMATEUR', 'mgs'));


-- get_videos_feed: コリレートサブクエリ → バッチJOINに変更
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
  cutpoint uuid;
BEGIN
  current_uid := auth.uid();
  cutpoint := gen_random_uuid();

  IF current_uid IS NOT NULL THEN
    SELECT array_agg(uet.tag_id) INTO excluded_tag_ids
    FROM public.user_excluded_tags uet
    WHERE uet.user_id = current_uid;
  END IF;

  RETURN QUERY
  WITH video_ids AS (
    -- UUID スプリットで候補ID取得（インデックスのみでフィルタ）
    (
      SELECT v.id
      FROM public.videos v
      WHERE v.id >= cutpoint
        AND v.sample_video_url IS NOT NULL
        AND v.source NOT IN ('FANZA_ANIME')
        AND (v.has_performer = true OR v.source IN ('FANZA_AMATEUR', 'mgs'))
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
        )
      ORDER BY v.id
      LIMIT page_limit
    )
    UNION ALL
    (
      SELECT v.id
      FROM public.videos v
      WHERE v.id < cutpoint
        AND v.sample_video_url IS NOT NULL
        AND v.source NOT IN ('FANZA_ANIME')
        AND (v.has_performer = true OR v.source IN ('FANZA_AMATEUR', 'mgs'))
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
        )
      ORDER BY v.id
      LIMIT page_limit
    )
    LIMIT page_limit
  ),
  -- performers を候補全動画まとめて1回のJOINで集計
  perf_agg AS (
    SELECT vp.video_id,
           jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name) AS agg
    FROM public.video_performers vp
    JOIN public.performers p ON p.id = vp.performer_id
    WHERE vp.video_id IN (SELECT id FROM video_ids)
    GROUP BY vp.video_id
  ),
  -- tags を候補全動画まとめて1回のJOINで集計
  tags_agg AS (
    SELECT vt.video_id,
           jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name) AS agg
    FROM public.video_tags vt
    JOIN public.tags t ON t.id = vt.tag_id
    LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
    WHERE vt.video_id IN (SELECT id FROM video_ids)
      AND coalesce(tg.show_in_ui, true)
    GROUP BY vt.video_id
  )
  SELECT
    v.id, v.title, v.description, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url,
    v.sample_video_url, v.preview_video_url,
    coalesce(v.affiliate_url, v.product_url) AS product_url,
    v.product_released_at,
    coalesce(pa.agg, '[]'::jsonb) AS performers,
    coalesce(ta.agg, '[]'::jsonb) AS tags,
    v.source,
    v.image_urls
  FROM video_ids vi
  JOIN public.videos v ON v.id = vi.id
  LEFT JOIN perf_agg pa ON pa.video_id = v.id
  LEFT JOIN tags_agg ta ON ta.video_id = v.id;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_videos_feed(int) TO anon, authenticated;


-- get_videos_recommendations: コリレートサブクエリ → バッチJOINに変更
DROP FUNCTION IF EXISTS public.get_videos_recommendations(uuid, int);
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
  model_version text,
  source text,
  image_urls text[]
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
      AND uvd.decision_type IN ('swipe_like', 'grid_like')
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
  IF user_vec IS NULL THEN RETURN; END IF;

  SELECT ve.model_version INTO latest_version
  FROM public.video_embeddings ve
  WHERE ve.model_version IS NOT NULL
  ORDER BY ve.updated_at DESC
  LIMIT 1;

  RETURN QUERY
  WITH candidates AS (
    -- ベクトル類似度検索でID + スコア取得
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
      1 - (ve.embedding <-> user_vec) AS score,
      ve.model_version,
      v.source,
      v.image_urls
    FROM public.video_embeddings ve
    JOIN public.videos v ON v.id = ve.video_id
    WHERE v.sample_video_url IS NOT NULL
      AND (latest_version IS NULL OR ve.model_version = latest_version)
      AND NOT EXISTS (
        SELECT 1 FROM public.user_video_decisions uvd
        WHERE uvd.video_id = v.id AND uvd.user_id = user_uuid
      )
      AND NOT EXISTS (
        SELECT 1 FROM public.user_excluded_tags uet
        JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
        WHERE vt.video_id = v.id AND uet.user_id = user_uuid
      )
    ORDER BY ve.embedding <-> user_vec
    LIMIT page_limit
  ),
  perf_agg AS (
    SELECT vp.video_id,
           jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name) AS agg
    FROM public.video_performers vp
    JOIN public.performers p ON p.id = vp.performer_id
    WHERE vp.video_id IN (SELECT id FROM candidates)
    GROUP BY vp.video_id
  ),
  tags_agg AS (
    SELECT vt.video_id,
           jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name) AS agg
    FROM public.video_tags vt
    JOIN public.tags t ON t.id = vt.tag_id
    LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
    WHERE vt.video_id IN (SELECT id FROM candidates)
      AND coalesce(tg.show_in_ui, true)
    GROUP BY vt.video_id
  )
  SELECT
    c.id, c.title, c.description, c.external_id,
    c.thumbnail_url, c.thumbnail_vertical_url,
    c.sample_video_url, c.product_url,
    c.product_released_at,
    coalesce(pa.agg, '[]'::jsonb) AS performers,
    coalesce(ta.agg, '[]'::jsonb) AS tags,
    c.score,
    c.model_version,
    c.source,
    c.image_urls
  FROM candidates c
  LEFT JOIN perf_agg pa ON pa.video_id = c.id
  LEFT JOIN tags_agg ta ON ta.video_id = c.id
  ORDER BY c.score DESC;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_videos_recommendations(uuid, int) TO anon, authenticated;
