CREATE OR REPLACE FUNCTION explore_videos(
  p_query        text    DEFAULT NULL,
  p_tag_id       uuid    DEFAULT NULL,
  p_performer_id uuid    DEFAULT NULL,
  p_limit        int     DEFAULT 30,
  p_offset       int     DEFAULT 0
)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN (
    SELECT json_agg(row_to_json(v))
    FROM (
      SELECT
        vi.id,
        vi.title,
        vi.external_id,
        vi.thumbnail_url,
        vi.thumbnail_vertical_url,
        vi.product_url,
        vi.distribution_code,
        vi.product_released_at
      FROM public.videos vi
      LEFT JOIN public.fanza_rankings fr ON fr.video_id = vi.id
      WHERE vi.external_id IS NOT NULL
        AND (
          p_query IS NULL
          OR vi.title ILIKE '%' || p_query || '%'
          OR vi.distribution_code ILIKE '%' || p_query || '%'
        )
        AND (
          p_tag_id IS NULL
          OR EXISTS (
            SELECT 1 FROM public.video_tags vt
            WHERE vt.video_id = vi.id AND vt.tag_id = p_tag_id
          )
        )
        AND (
          p_performer_id IS NULL
          OR EXISTS (
            SELECT 1 FROM public.video_performers vp
            WHERE vp.video_id = vi.id AND vp.performer_id = p_performer_id
          )
        )
      ORDER BY
        CASE WHEN p_tag_id IS NOT NULL OR p_query IS NOT NULL
          THEN fr.rank END ASC NULLS LAST,
        vi.product_released_at DESC NULLS LAST
      LIMIT p_limit
      OFFSET p_offset
    ) v
  );
END;
$$;
