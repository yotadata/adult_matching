-- Reduce embedding dimensions to 128 and store as half precision (halfvec)
-- This migration truncates existing embedding tables; regenerate embeddings after applying.

DROP INDEX IF EXISTS idx_video_embeddings_cosine;
DROP INDEX IF EXISTS idx_user_embeddings_cosine;

TRUNCATE TABLE public.video_embeddings;
TRUNCATE TABLE public.user_embeddings;

ALTER TABLE public.video_embeddings
  ALTER COLUMN embedding TYPE halfvec(128);

ALTER TABLE public.user_embeddings
  ALTER COLUMN embedding TYPE halfvec(128);

CREATE INDEX IF NOT EXISTS idx_video_embeddings_cosine
  ON public.video_embeddings USING ivfflat (embedding halfvec_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_user_embeddings_cosine
  ON public.user_embeddings USING ivfflat (embedding halfvec_cosine_ops);

-- Align recommendation function signature with new embedding type
CREATE OR REPLACE FUNCTION public.get_videos_recommendations(user_uuid uuid, page_limit int DEFAULT 20)
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
  score double precision,
  model_version text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  user_vec halfvec(128);
  latest_version text;
BEGIN
  SELECT ue.embedding INTO user_vec
  FROM public.user_embeddings AS ue
  WHERE ue.user_id = user_uuid;

  IF user_vec IS NULL THEN
    RETURN;
  END IF;

  SELECT ve.model_version
    INTO latest_version
  FROM public.video_embeddings AS ve
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
    v.sample_video_url,
    v.product_released_at,
    COALESCE(
      (
        SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name)
        FROM public.video_performers vp
        JOIN public.performers p ON p.id = vp.performer_id
        WHERE vp.video_id = v.id
      ), '[]'::jsonb
    ) AS performers,
    COALESCE(
      (
        SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name)
        FROM public.video_tags vt
        JOIN public.tags t ON t.id = vt.tag_id
        WHERE vt.video_id = v.id
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
  ORDER BY ve.embedding <-> user_vec
  LIMIT page_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_videos_recommendations(uuid, int) TO anon, authenticated;
