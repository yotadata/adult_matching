-- videos-grid RPC 高速化インデックス
--
-- 追加するインデックス:
--   1. user_video_decisions (user_id, video_id) — NOT EXISTS サブクエリ高速化
--      (get_popular_videos / get_videos_recommendations 共通)
--   2. user_video_decisions (user_id, decision_type, created_at DESC)
--      — get_videos_recommendations の recent_likes CTE 高速化
--   3. video_popularity_daily (d, video_id)
--      — get_popular_videos の d >= threshold レンジスキャン高速化
--   4. video_embeddings HNSW を halfvec_cosine_ops に置き換え
--      — get_videos_recommendations の <=> 近傍探索に合致させる
--      — 既存 halfvec_ip_ops (内積) では <-> (L2) クエリがインデックスを使えない

-- 1. user_video_decisions: NOT EXISTS 用複合インデックス
CREATE INDEX IF NOT EXISTS uvd_user_video_idx
  ON public.user_video_decisions (user_id, video_id);

-- 2. user_video_decisions: recent_likes CTE 用インデックス
CREATE INDEX IF NOT EXISTS uvd_user_type_created_idx
  ON public.user_video_decisions (user_id, decision_type, created_at DESC);

-- 3. video_popularity_daily: d レンジスキャン用インデックス
CREATE INDEX IF NOT EXISTS vpd_d_video_idx
  ON public.video_popularity_daily (d, video_id);

-- 4. video_embeddings: HNSW を cosine_ops に差し替え
--    halfvec_ip_ops (内積) では <=> (cosine) / <-> (L2) クエリで使われない
DROP INDEX IF EXISTS public.video_embeddings_embedding_hnsw_idx;
CREATE INDEX IF NOT EXISTS video_embeddings_hnsw_cosine_idx
  ON public.video_embeddings
  USING hnsw (embedding halfvec_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 5. get_videos_recommendations: <-> (L2) → <=> (cosine) に変更して cosine HNSW を活用
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
  -- 近似ユーザー埋め込み: いいね済み動画の埋め込み平均（直近200件）
  -- uvd_user_type_created_idx を使用
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

  -- いいねがゼロの場合はバッチ生成済み埋め込みにフォールバック
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

  -- <=> (cosine distance) を使用して video_embeddings_hnsw_cosine_idx を活用
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
    (1 - (ve.embedding <=> user_vec))::double precision AS score,
    ve.model_version,
    v.source,
    v.image_urls
  FROM public.video_embeddings ve
  JOIN public.videos v ON v.id = ve.video_id
  WHERE v.sample_video_url IS NOT NULL
    AND (latest_version IS NULL OR ve.model_version = latest_version)
    AND NOT EXISTS (
      -- uvd_user_video_idx を使用
      SELECT 1 FROM public.user_video_decisions uvd
      WHERE uvd.user_id = user_uuid
        AND uvd.video_id = v.id
    )
  ORDER BY ve.embedding <=> user_vec
  LIMIT page_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_videos_recommendations(uuid, int) TO anon, authenticated;
