CREATE OR REPLACE FUNCTION get_popular_tags(p_limit int DEFAULT 20)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN (
    SELECT json_agg(row_to_json(t))
    FROM (
      SELECT tg.id, tg.name, COUNT(*) AS cnt
      FROM public.video_tags vt
      JOIN public.tags tg ON tg.id = vt.tag_id
      GROUP BY tg.id, tg.name
      ORDER BY cnt DESC
      LIMIT p_limit
    ) t
  );
END;
$$;
