-- ユーザーの公開リスト一覧を取得するRPC
CREATE OR REPLACE FUNCTION get_user_public_lists(p_username text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_user_id      uuid;
  v_display_name text;
  v_lists        json;
BEGIN
  SELECT id, COALESCE(raw_user_meta_data->>'display_name', '')
  INTO v_user_id, v_display_name
  FROM auth.users
  WHERE raw_user_meta_data->>'username' = p_username
  LIMIT 1;

  IF v_user_id IS NULL THEN
    RETURN json_build_object('error', 'not_found');
  END IF;

  SELECT json_agg(row_to_json(l)) INTO v_lists
  FROM (
    SELECT
      pl.id,
      pl.token,
      pl.title,
      pl.description,
      pl.list_type,
      pl.created_at,
      CASE
        WHEN pl.list_type = 'custom' THEN
          (SELECT COUNT(*) FROM public.public_list_videos WHERE list_id = pl.id)
        ELSE
          (SELECT COUNT(*) FROM public.user_video_decisions
           WHERE user_id = v_user_id
             AND decision_type IN ('swipe_like', 'grid_like'))
      END AS video_count,
      CASE
        WHEN pl.list_type = 'custom' THEN
          (SELECT json_agg(thumbnail_url)
           FROM (
             SELECT vi.thumbnail_url
             FROM public.public_list_videos plv
             JOIN public.videos vi ON vi.id = plv.video_id
             WHERE plv.list_id = pl.id AND vi.thumbnail_url IS NOT NULL
             ORDER BY plv.sort_order, plv.added_at LIMIT 3
           ) t)
        ELSE
          (SELECT json_agg(thumbnail_url)
           FROM (
             SELECT vi.thumbnail_url
             FROM public.user_video_decisions uvd
             JOIN public.videos vi ON vi.id = uvd.video_id
             WHERE uvd.user_id = v_user_id
               AND uvd.decision_type IN ('swipe_like', 'grid_like')
               AND vi.thumbnail_url IS NOT NULL
             ORDER BY uvd.created_at DESC LIMIT 3
           ) t)
      END AS thumbnails
    FROM public.public_lists pl
    WHERE pl.user_id = v_user_id AND pl.is_active = true
    ORDER BY pl.created_at DESC
  ) l;

  RETURN json_build_object(
    'display_name', v_display_name,
    'username', p_username,
    'lists', COALESCE(v_lists, '[]'::json)
  );
END;
$$;

-- 動画検索RPC
CREATE OR REPLACE FUNCTION search_videos(p_query text, p_limit int DEFAULT 20, p_offset int DEFAULT 0)
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
        vi.product_url,
        vi.distribution_code
      FROM public.videos vi
      WHERE
        vi.external_id IS NOT NULL
        AND (
          vi.title ILIKE '%' || p_query || '%'
          OR vi.distribution_code ILIKE '%' || p_query || '%'
          OR vi.external_id ILIKE '%' || p_query || '%'
        )
      ORDER BY vi.product_released_at DESC NULLS LAST
      LIMIT p_limit
      OFFSET p_offset
    ) v
  );
END;
$$;
