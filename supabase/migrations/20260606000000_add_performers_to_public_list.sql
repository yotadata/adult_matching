-- get_public_list_data に女優集計（performers）を追加
CREATE OR REPLACE FUNCTION public.get_public_list_data(p_token text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_list_id      uuid;
  v_user_id      uuid;
  v_title        text;
  v_desc         text;
  v_list_type    text;
  v_display_name text;
  v_videos       json;
  v_tags         json;
  v_performers   json;
BEGIN
  SELECT id, user_id, title, description, list_type
  INTO v_list_id, v_user_id, v_title, v_desc, v_list_type
  FROM public.public_lists
  WHERE token = p_token AND is_active = true;

  IF v_user_id IS NULL THEN
    RETURN json_build_object('error', 'not_found');
  END IF;

  -- 動画一覧取得
  IF v_list_type = 'custom' THEN
    SELECT json_agg(row_to_json(v)) INTO v_videos
    FROM (
      SELECT
        vi.id,
        vi.title,
        vi.external_id,
        vi.thumbnail_url,
        vi.thumbnail_vertical_url,
        vi.product_url,
        plv.added_at AS liked_at
      FROM public.public_list_videos plv
      JOIN public.videos vi ON vi.id = plv.video_id
      WHERE plv.list_id = v_list_id
        AND vi.external_id IS NOT NULL
      ORDER BY plv.sort_order, plv.added_at DESC
      LIMIT 200
    ) v;
  ELSE
    SELECT json_agg(row_to_json(v)) INTO v_videos
    FROM (
      SELECT
        vi.id,
        vi.title,
        vi.external_id,
        vi.thumbnail_url,
        vi.thumbnail_vertical_url,
        vi.product_url,
        uvd.created_at AS liked_at
      FROM public.user_video_decisions uvd
      JOIN public.videos vi ON vi.id = uvd.video_id
      WHERE uvd.user_id = v_user_id
        AND uvd.decision_type IN ('swipe_like', 'grid_like')
        AND vi.external_id IS NOT NULL
      ORDER BY uvd.created_at DESC
    ) v;
  END IF;

  -- タグ集計
  IF v_list_type = 'custom' THEN
    SELECT json_agg(row_to_json(t)) INTO v_tags
    FROM (
      SELECT tg.name AS tag_name, COUNT(*) AS cnt
      FROM public.public_list_videos plv
      JOIN public.video_tags vt ON vt.video_id = plv.video_id
      JOIN public.tags tg ON tg.id = vt.tag_id
      WHERE plv.list_id = v_list_id
      GROUP BY tg.name
      ORDER BY cnt DESC
      LIMIT 12
    ) t;
  ELSE
    SELECT json_agg(row_to_json(t)) INTO v_tags
    FROM (
      SELECT tg.name AS tag_name, COUNT(*) AS cnt
      FROM public.user_video_decisions uvd
      JOIN public.video_tags vt ON vt.video_id = uvd.video_id
      JOIN public.tags tg ON tg.id = vt.tag_id
      WHERE uvd.user_id = v_user_id
        AND uvd.decision_type IN ('swipe_like', 'grid_like')
      GROUP BY tg.name
      ORDER BY cnt DESC
      LIMIT 12
    ) t;
  END IF;

  -- 女優集計
  IF v_list_type = 'custom' THEN
    SELECT json_agg(row_to_json(p)) INTO v_performers
    FROM (
      SELECT pf.name AS performer_name, COUNT(*) AS cnt
      FROM public.public_list_videos plv
      JOIN public.video_performers vp ON vp.video_id = plv.video_id
      JOIN public.performers pf ON pf.id = vp.performer_id
      WHERE plv.list_id = v_list_id
      GROUP BY pf.name
      ORDER BY cnt DESC
      LIMIT 10
    ) p;
  ELSE
    SELECT json_agg(row_to_json(p)) INTO v_performers
    FROM (
      SELECT pf.name AS performer_name, COUNT(*) AS cnt
      FROM public.user_video_decisions uvd
      JOIN public.video_performers vp ON vp.video_id = uvd.video_id
      JOIN public.performers pf ON pf.id = vp.performer_id
      WHERE uvd.user_id = v_user_id
        AND uvd.decision_type IN ('swipe_like', 'grid_like')
      GROUP BY pf.name
      ORDER BY cnt DESC
      LIMIT 10
    ) p;
  END IF;

  SELECT display_name INTO v_display_name
  FROM public.profiles WHERE user_id = v_user_id;

  RETURN json_build_object(
    'display_name', v_display_name,
    'title',        v_title,
    'description',  v_desc,
    'list_type',    v_list_type,
    'videos',       COALESCE(v_videos,     '[]'::json),
    'tags',         COALESCE(v_tags,       '[]'::json),
    'performers',   COALESCE(v_performers, '[]'::json)
  );
END;
$$;
