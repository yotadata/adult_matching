-- 指定した動画と embedding が近い類似作品を返す関数
-- video_embeddings が存在しない動画の場合は空を返す
CREATE OR REPLACE FUNCTION get_similar_videos(
  p_video_id uuid,
  p_limit     int DEFAULT 8
)
RETURNS TABLE (
  id                     uuid,
  title                  text,
  thumbnail_url          text,
  thumbnail_vertical_url text,
  affiliate_url          text,
  product_url            text,
  external_id            text,
  maker                  text,
  product_released_at    timestamptz,
  similarity             float8
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    v.id,
    v.title,
    v.thumbnail_url,
    v.thumbnail_vertical_url,
    v.affiliate_url,
    v.product_url,
    v.external_id,
    v.maker,
    v.product_released_at,
    1.0 - (ve_other.embedding <=> ve_base.embedding) AS similarity
  FROM
    video_embeddings ve_base
    JOIN video_embeddings ve_other
      ON ve_other.video_id  != ve_base.video_id
     AND ve_other.model_version = ve_base.model_version
    JOIN videos v ON v.id = ve_other.video_id
  WHERE
    ve_base.video_id = p_video_id
  ORDER BY
    ve_other.embedding <=> ve_base.embedding
  LIMIT p_limit;
$$;

-- 匿名ユーザーも参照可能（videos テーブルと同様に公開データ）
GRANT EXECUTE ON FUNCTION get_similar_videos(uuid, int) TO anon, authenticated;
