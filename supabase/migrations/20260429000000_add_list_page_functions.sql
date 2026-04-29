-- ジャンル一覧ページ用: 動画数付きタグ一覧
CREATE OR REPLACE FUNCTION get_tags_with_count()
RETURNS TABLE (id uuid, name text, video_count bigint)
LANGUAGE sql STABLE AS $$
  SELECT t.id, t.name, COUNT(vt.video_id) AS video_count
  FROM tags t
  LEFT JOIN video_tags vt ON vt.tag_id = t.id
  GROUP BY t.id, t.name
  ORDER BY video_count DESC, t.name;
$$;

GRANT EXECUTE ON FUNCTION get_tags_with_count() TO anon, authenticated;

-- 出演者一覧ページ用: 動画数付き出演者一覧（ページネーション対応）
CREATE OR REPLACE FUNCTION get_performers_with_count(
  p_limit  int DEFAULT 60,
  p_offset int DEFAULT 0
)
RETURNS TABLE (id uuid, name text, video_count bigint)
LANGUAGE sql STABLE AS $$
  SELECT p.id, p.name, COUNT(vp.video_id) AS video_count
  FROM performers p
  LEFT JOIN video_performers vp ON vp.performer_id = p.id
  GROUP BY p.id, p.name
  ORDER BY video_count DESC, p.name
  LIMIT p_limit OFFSET p_offset;
$$;

GRANT EXECUTE ON FUNCTION get_performers_with_count(int, int) TO anon, authenticated;

-- 出演者総数
CREATE OR REPLACE FUNCTION get_performers_count()
RETURNS bigint
LANGUAGE sql STABLE AS $$
  SELECT COUNT(*) FROM performers;
$$;

GRANT EXECUTE ON FUNCTION get_performers_count() TO anon, authenticated;
