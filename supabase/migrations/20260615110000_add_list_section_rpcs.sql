-- =============================================
-- セクション管理 & 並べ替え RPC
-- =============================================

-- セクション作成・更新
CREATE OR REPLACE FUNCTION public.upsert_list_section(
  p_list_id    uuid,
  p_section_id uuid,    -- NULL で新規作成
  p_title      text,
  p_sort_order integer
) RETURNS uuid
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_owner_id uuid;
  v_id       uuid;
BEGIN
  SELECT user_id INTO v_owner_id FROM public.public_lists WHERE id = p_list_id;
  IF v_owner_id IS DISTINCT FROM auth.uid() THEN
    RAISE EXCEPTION 'permission_denied';
  END IF;

  IF p_section_id IS NULL THEN
    INSERT INTO public.public_list_sections(list_id, title, sort_order)
    VALUES (p_list_id, p_title, p_sort_order)
    RETURNING id INTO v_id;
  ELSE
    UPDATE public.public_list_sections
    SET title = p_title, sort_order = p_sort_order
    WHERE id = p_section_id AND list_id = p_list_id
    RETURNING id INTO v_id;
  END IF;

  RETURN v_id;
END;
$$;

-- セクション削除（動画の section_id は ON DELETE SET NULL で自動 NULL 化）
CREATE OR REPLACE FUNCTION public.delete_list_section(
  p_section_id uuid
) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_owner_id uuid;
BEGIN
  SELECT pl.user_id INTO v_owner_id
  FROM public.public_list_sections s
  JOIN public.public_lists pl ON pl.id = s.list_id
  WHERE s.id = p_section_id;

  IF v_owner_id IS DISTINCT FROM auth.uid() THEN
    RAISE EXCEPTION 'permission_denied';
  END IF;

  DELETE FROM public.public_list_sections WHERE id = p_section_id;
END;
$$;

-- セクション内の動画を一括並べ替え
-- p_video_ids: 新しい順序の video_id 配列（先頭が sort_order=0）
CREATE OR REPLACE FUNCTION public.reorder_list_videos(
  p_list_id   uuid,
  p_video_ids uuid[]
) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_owner_id uuid;
  i          integer;
BEGIN
  SELECT user_id INTO v_owner_id FROM public.public_lists WHERE id = p_list_id;
  IF v_owner_id IS DISTINCT FROM auth.uid() THEN
    RAISE EXCEPTION 'permission_denied';
  END IF;

  FOR i IN 1..array_length(p_video_ids, 1) LOOP
    UPDATE public.public_list_videos
    SET sort_order = i - 1
    WHERE list_id = p_list_id AND video_id = p_video_ids[i];
  END LOOP;
END;
$$;

-- 動画をセクションに割り当て
CREATE OR REPLACE FUNCTION public.assign_video_section(
  p_list_id    uuid,
  p_video_id   uuid,
  p_section_id uuid   -- NULL で未分類に戻す
) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_owner_id uuid;
BEGIN
  SELECT user_id INTO v_owner_id FROM public.public_lists WHERE id = p_list_id;
  IF v_owner_id IS DISTINCT FROM auth.uid() THEN
    RAISE EXCEPTION 'permission_denied';
  END IF;

  UPDATE public.public_list_videos
  SET section_id = p_section_id
  WHERE list_id = p_list_id AND video_id = p_video_id;
END;
$$;
