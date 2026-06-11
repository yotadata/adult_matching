-- キュレーター機能: get_user_public_lists に avatar_url/bio/x_url/view_count/like_count を追加
CREATE OR REPLACE FUNCTION get_user_public_lists(p_username text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_user_id      uuid;
  v_display_name text;
  v_avatar_url   text;
  v_bio          text;
  v_x_url        text;
  v_lists        json;
BEGIN
  SELECT
    u.id,
    COALESCE(u.raw_user_meta_data->>'display_name', ''),
    up.avatar_url,
    up.bio,
    up.x_url
  INTO v_user_id, v_display_name, v_avatar_url, v_bio, v_x_url
  FROM auth.users u
  LEFT JOIN public.user_profiles up ON up.user_id = u.id
  WHERE u.raw_user_meta_data->>'username' = p_username
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
      pl.view_count,
      (SELECT COUNT(*) FROM public.list_likes WHERE list_id = pl.id) AS like_count,
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
    'user_id',      v_user_id,
    'display_name', v_display_name,
    'username',     p_username,
    'avatar_url',   v_avatar_url,
    'bio',          v_bio,
    'x_url',        v_x_url,
    'lists',        COALESCE(v_lists, '[]'::json)
  );
END;
$$;
