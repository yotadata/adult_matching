-- フィードRPCのパフォーマンス改善 v2
--
-- 問題1: get_videos_feed が COUNT + SELECT の2回フルスキャンをしていた
-- 問題2: has_performer フラグを使わず EXISTS(video_performers) を毎行実行していた
-- 問題3: get_videos_recommendations の decision_type = 'like' バグ（実際は swipe_like/grid_like）
--
-- 修正1: COUNT不要のランダムUUIDスプリット方式（1回スキャン）に変更
-- 修正2: has_performer カラムを使うように変更（インデックス利用）
-- 修正3: decision_type を IN ('swipe_like', 'grid_like') に修正

-- get_videos_feed: COUNT廃止 + has_performer 利用
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

  -- 除外タグIDを一度だけ取得
  IF current_uid IS NOT NULL THEN
    SELECT array_agg(uet.tag_id) INTO excluded_tag_ids
    FROM public.user_excluded_tags uet
    WHERE uet.user_id = current_uid;
  END IF;

  RETURN QUERY
  WITH base AS (
    -- id >= cutpoint の分（括弧でORDER BY + LIMITをUNION ALLで使えるようにする）
    (SELECT
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
          SELECT 1 FROM public.video_tags vt2
          WHERE vt2.video_id = v.id AND vt2.tag_id = ANY(excluded_tag_ids)
        )
      )
    ORDER BY v.id
    LIMIT page_limit)

    UNION ALL

    -- 足りない場合は id < cutpoint で補完
    (SELECT
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
          SELECT 1 FROM public.video_tags vt2
          WHERE vt2.video_id = v.id AND vt2.tag_id = ANY(excluded_tag_ids)
        )
      )
    ORDER BY v.id
    LIMIT page_limit)
  )
  SELECT * FROM base LIMIT page_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_videos_feed(int) TO anon, authenticated;


-- get_videos_recommendations: decision_type = 'like' バグ修正
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
  -- 修正: 'like' → IN ('swipe_like', 'grid_like')
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
    ve.model_version,
    v.source,
    v.image_urls
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
