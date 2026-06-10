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
      WHERE tg.name NOT IN (
        '独占配信', 'ハイビジョン', '単体作品', '4K', '8K', 'VR', 'Ultra HD',
        '無修正', '高画質', 'フルHD', 'DVD', 'Blu-ray', '収録時間4時間以上',
        'デジモ', 'サンプル動画あり', 'デモ・体験版あり', 'PPV', '独占'
      )
      GROUP BY tg.id, tg.name
      ORDER BY cnt DESC
      LIMIT p_limit
    ) t
  );
END;
$$;
