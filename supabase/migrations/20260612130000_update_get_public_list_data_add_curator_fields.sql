-- get_public_list_data に view_count/like_count/source/image_urls/affiliate_fanza_id を追加
CREATE OR REPLACE FUNCTION public.get_public_list_data(p_token text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_list_id            uuid;
  v_user_id            uuid;
  v_title              text;
  v_desc               text;
  v_list_type          text;
  v_view_count         bigint;
  v_display_name       text;
  v_username           text;
  v_affiliate_fanza_id text;
  v_affiliate_fc2_id   text;
  v_affiliate_mgs_id   text;
  v_like_count         bigint;
  v_videos             json;
  v_tags               json;
  v_performers         json;
BEGIN
  SELECT id, user_id, title, description, list_type, view_count
  INTO v_list_id, v_user_id, v_title, v_desc, v_list_type, v_view_count
  FROM public.public_lists
  WHERE token = p_token AND is_active = true;

  IF v_user_id IS NULL THEN
    RETURN json_build_object('error', 'not_found');
  END IF;

  SELECT
    COALESCE(u.raw_user_meta_data->>'display_name', ''),
    u.raw_user_meta_data->>'username',
    up.affiliate_fanza_id,
    up.affiliate_fc2_id,
    up.affiliate_mgs_id
  INTO v_display_name, v_username, v_affiliate_fanza_id, v_affiliate_fc2_id, v_affiliate_mgs_id
  FROM auth.users u
  LEFT JOIN public.user_profiles up ON up.user_id = u.id
  WHERE u.id = v_user_id;

  SELECT COUNT(*) INTO v_like_count FROM public.list_likes WHERE list_id = v_list_id;

  IF v_list_type = 'custom' THEN
    SELECT json_agg(row_to_json(v)) INTO v_videos
    FROM (
      SELECT vi.id, vi.title, vi.external_id, vi.thumbnail_url,
             vi.thumbnail_vertical_url, vi.product_url, vi.source, vi.image_urls,
             plv.added_at AS liked_at
      FROM public.public_list_videos plv
      JOIN public.videos vi ON vi.id = plv.video_id
      WHERE plv.list_id = v_list_id AND vi.external_id IS NOT NULL
      ORDER BY plv.sort_order, plv.added_at DESC LIMIT 200
    ) v;
  ELSE
    SELECT json_agg(row_to_json(v)) INTO v_videos
    FROM (
      SELECT vi.id, vi.title, vi.external_id, vi.thumbnail_url,
             vi.thumbnail_vertical_url, vi.product_url, vi.source, vi.image_urls,
             uvd.created_at AS liked_at
      FROM public.user_video_decisions uvd
      JOIN public.videos vi ON vi.id = uvd.video_id
      WHERE uvd.user_id = v_user_id
        AND uvd.decision_type IN ('swipe_like', 'grid_like')
        AND vi.external_id IS NOT NULL
      ORDER BY uvd.created_at DESC
    ) v;
  END IF;

  IF v_list_type = 'custom' THEN
    SELECT json_agg(row_to_json(t)) INTO v_tags
    FROM (
      SELECT tg.name AS tag_name, COUNT(*) AS cnt
      FROM public.public_list_videos plv
      JOIN public.video_tags vt ON vt.video_id = plv.video_id
      JOIN public.tags tg ON tg.id = vt.tag_id
      WHERE plv.list_id = v_list_id
      GROUP BY tg.name ORDER BY cnt DESC LIMIT 12
    ) t;
  ELSE
    SELECT json_agg(row_to_json(t)) INTO v_tags
    FROM (
      SELECT tg.name AS tag_name, COUNT(*) AS cnt
      FROM public.user_video_decisions uvd
      JOIN public.video_tags vt ON vt.video_id = uvd.video_id
      JOIN public.tags tg ON tg.id = vt.tag_id
      WHERE uvd.user_id = v_user_id AND uvd.decision_type IN ('swipe_like', 'grid_like')
      GROUP BY tg.name ORDER BY cnt DESC LIMIT 12
    ) t;
  END IF;

  IF v_list_type = 'custom' THEN
    SELECT json_agg(row_to_json(p)) INTO v_performers
    FROM (
      SELECT pf.name AS performer_name, COUNT(*) AS cnt
      FROM public.public_list_videos plv
      JOIN public.video_performers vp ON vp.video_id = plv.video_id
      JOIN public.performers pf ON pf.id = vp.performer_id
      WHERE plv.list_id = v_list_id
      GROUP BY pf.name ORDER BY cnt DESC LIMIT 10
    ) p;
  ELSE
    SELECT json_agg(row_to_json(p)) INTO v_performers
    FROM (
      SELECT pf.name AS performer_name, COUNT(*) AS cnt
      FROM public.user_video_decisions uvd
      JOIN public.video_performers vp ON vp.video_id = uvd.video_id
      JOIN public.performers pf ON pf.id = vp.performer_id
      WHERE uvd.user_id = v_user_id AND uvd.decision_type IN ('swipe_like', 'grid_like')
      GROUP BY pf.name ORDER BY cnt DESC LIMIT 10
    ) p;
  END IF;

  RETURN json_build_object(
    'list_id',             v_list_id,
    'user_id',             v_user_id,
    'display_name',        v_display_name,
    'username',            v_username,
    'affiliate_fanza_id',  v_affiliate_fanza_id,
    'affiliate_fc2_id',    v_affiliate_fc2_id,
    'affiliate_mgs_id',    v_affiliate_mgs_id,
    'title',               v_title,
    'description',         v_desc,
    'list_type',           v_list_type,
    'view_count',          COALESCE(v_view_count, 0),
    'like_count',          v_like_count,
    'videos',              COALESCE(v_videos,     '[]'::json),
    'tags',                COALESCE(v_tags,       '[]'::json),
    'performers',          COALESCE(v_performers, '[]'::json)
  );
END;
$$;
