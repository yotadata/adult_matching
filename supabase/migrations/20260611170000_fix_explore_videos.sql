-- explore_videos: 部分インデックス・trgm・タグJOIN で全件スキャン排除
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS videos_title_trgm_idx
  ON public.videos USING gin (title gin_trgm_ops);

CREATE INDEX IF NOT EXISTS videos_distribution_code_trgm_idx
  ON public.videos USING gin (distribution_code gin_trgm_ops);

CREATE INDEX IF NOT EXISTS videos_explore_eligible_released_idx
  ON public.videos (product_released_at DESC NULLS LAST)
  WHERE external_id IS NOT NULL
    AND sample_video_url IS NOT NULL
    AND source NOT IN ('FANZA_ANIME')
    AND (has_performer = true OR source = 'FANZA_AMATEUR');

CREATE INDEX IF NOT EXISTS video_tags_tag_id_idx
  ON public.video_tags (tag_id, video_id);

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
          vi.product_url, vi.distribution_code, vi.product_released_at
        FROM public.videos vi
        WHERE vi.external_id IS NOT NULL
          AND vi.sample_video_url IS NOT NULL
          AND (vi.has_performer = true OR vi.source = 'FANZA_AMATEUR')
          AND vi.source NOT IN ('FANZA_ANIME')
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
          vi.product_url, vi.distribution_code, vi.product_released_at
        FROM public.video_tags vt
        JOIN public.videos vi ON vi.id = vt.video_id
        LEFT JOIN public.fanza_rankings fr ON fr.video_id = vi.id
        WHERE vt.tag_id = p_tag_id
          AND vi.external_id IS NOT NULL
          AND vi.sample_video_url IS NOT NULL
          AND (vi.has_performer = true OR vi.source = 'FANZA_AMATEUR')
          AND vi.source NOT IN ('FANZA_ANIME')
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
      ORDER BY
        CASE WHEN p_tag_id IS NOT NULL OR p_query IS NOT NULL THEN fr.rank END ASC NULLS LAST,
        vi.product_released_at DESC NULLS LAST
      LIMIT p_limit OFFSET p_offset
    ) v
  );
END;
$$;
