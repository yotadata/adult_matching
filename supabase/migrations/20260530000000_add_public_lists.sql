-- いいねリストの公開共有機能
CREATE TABLE IF NOT EXISTS public.public_lists (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  token       text        NOT NULL UNIQUE DEFAULT substring(encode(gen_random_bytes(9), 'base64'), 1, 12),
  title       text,
  is_active   boolean     NOT NULL DEFAULT true,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.public_lists ENABLE ROW LEVEL SECURITY;

-- token を知っていれば誰でも読める
CREATE POLICY "public_lists_select"
  ON public.public_lists FOR SELECT
  USING (is_active = true);

-- 本人のみ作成・更新・削除
CREATE POLICY "public_lists_insert"
  ON public.public_lists FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "public_lists_update"
  ON public.public_lists FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "public_lists_delete"
  ON public.public_lists FOR DELETE
  USING (auth.uid() = user_id);

-- 公開リストのデータを取得する関数
-- token で検索し、対応ユーザーのいいね動画＋タグ集計を返す
CREATE OR REPLACE FUNCTION public.get_public_list_data(p_token text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_user_id uuid;
  v_title   text;
  v_videos  json;
  v_tags    json;
BEGIN
  -- token から user_id を取得
  SELECT user_id, title INTO v_user_id, v_title
  FROM public.public_lists
  WHERE token = p_token AND is_active = true;

  IF v_user_id IS NULL THEN
    RETURN json_build_object('error', 'not_found');
  END IF;

  -- いいね動画を最新100件取得（外部IDが必要なものだけ）
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
    LIMIT 100
  ) v;

  -- タグ集計（上位12件）
  SELECT json_agg(row_to_json(t)) INTO v_tags
  FROM (
    SELECT
      tg.name AS tag_name,
      COUNT(*) AS cnt
    FROM public.user_video_decisions uvd
    JOIN public.video_tags vt ON vt.video_id = uvd.video_id
    JOIN public.tags tg ON tg.id = vt.tag_id
    WHERE uvd.user_id = v_user_id
      AND uvd.decision_type IN ('swipe_like', 'grid_like')
    GROUP BY tg.name
    ORDER BY cnt DESC
    LIMIT 12
  ) t;

  RETURN json_build_object(
    'title',  v_title,
    'videos', COALESCE(v_videos, '[]'::json),
    'tags',   COALESCE(v_tags, '[]'::json)
  );
END;
$$;
