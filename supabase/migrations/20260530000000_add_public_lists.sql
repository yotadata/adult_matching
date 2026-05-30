-- =============================================
-- いいねリスト公開 & カスタムリスト共有機能
-- =============================================

-- リスト本体
CREATE TABLE IF NOT EXISTS public.public_lists (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  token       text        NOT NULL UNIQUE DEFAULT translate(substring(encode(gen_random_bytes(9), 'base64'), 1, 12), '+/=', '-_'),
  title       text,
  description text,
  -- 'liked'  : いいねリストを動的に表示
  -- 'custom' : public_list_videos で明示的に管理
  list_type   text        NOT NULL DEFAULT 'liked' CHECK (list_type IN ('liked', 'custom')),
  is_active   boolean     NOT NULL DEFAULT true,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

-- リストに含める動画（custom リスト用・liked でも将来使用可）
CREATE TABLE IF NOT EXISTS public.public_list_videos (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  list_id     uuid        NOT NULL REFERENCES public.public_lists(id) ON DELETE CASCADE,
  video_id    uuid        NOT NULL REFERENCES public.videos(id) ON DELETE CASCADE,
  sort_order  integer     NOT NULL DEFAULT 0,
  added_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (list_id, video_id)
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_public_lists_user_id  ON public.public_lists(user_id);
CREATE INDEX IF NOT EXISTS idx_public_lists_token     ON public.public_lists(token);
CREATE INDEX IF NOT EXISTS idx_public_list_videos_list ON public.public_list_videos(list_id, sort_order);

-- =============================================
-- RLS
-- =============================================

ALTER TABLE public.public_lists        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.public_list_videos  ENABLE ROW LEVEL SECURITY;

-- public_lists ポリシー（冪等）
DROP POLICY IF EXISTS "public_lists_select" ON public.public_lists;
CREATE POLICY "public_lists_select"
  ON public.public_lists FOR SELECT
  USING (is_active = true);

DROP POLICY IF EXISTS "public_lists_update" ON public.public_lists;
CREATE POLICY "public_lists_update"
  ON public.public_lists FOR UPDATE
  USING (auth.uid() = user_id);

-- public_list_videos ポリシー（冪等）
DROP POLICY IF EXISTS "public_list_videos_select" ON public.public_list_videos;
CREATE POLICY "public_list_videos_select"
  ON public.public_list_videos FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.public_lists pl
      WHERE pl.id = list_id AND pl.is_active = true
    )
  );

DROP POLICY IF EXISTS "public_list_videos_update" ON public.public_list_videos;
CREATE POLICY "public_list_videos_update"
  ON public.public_list_videos FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.public_lists pl
      WHERE pl.id = list_id AND pl.user_id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "public_list_videos_delete" ON public.public_list_videos;
CREATE POLICY "public_list_videos_delete"
  ON public.public_list_videos FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM public.public_lists pl
      WHERE pl.id = list_id AND pl.user_id = auth.uid()
    )
  );

-- =============================================
-- 公開リストデータ取得関数
-- list_type='liked'  → user_video_decisions から動的取得（常に最新）
-- list_type='custom' → public_list_videos から取得
-- =============================================
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
    -- カスタムリスト: public_list_videos の順番通りに返す
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
    -- いいねリスト: user_video_decisions から動的取得
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

  -- タグ集計（liked: いいね全体, custom: リスト内動画）
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

  SELECT display_name INTO v_display_name
  FROM public.profiles WHERE user_id = v_user_id;

  RETURN json_build_object(
    'display_name', v_display_name,
    'title',        v_title,
    'description',  v_desc,
    'list_type',    v_list_type,
    'videos',       COALESCE(v_videos, '[]'::json),
    'tags',         COALESCE(v_tags,   '[]'::json)
  );
END;
$$;

-- タグ指定動画取得 RPC（コールドスタート用）
CREATE OR REPLACE FUNCTION public.get_videos_by_tags(
  tag_ids uuid[],
  exclude_ids uuid[],
  p_limit int DEFAULT 60
)
RETURNS TABLE (
  id uuid, title text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, product_url text, product_released_at timestamptz,
  performers jsonb, tags jsonb
)
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  RETURN QUERY
  SELECT v.id, v.title, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url, v.sample_video_url,
    coalesce(v.affiliate_url, v.product_url), v.product_released_at,
    coalesce((SELECT jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name)) FROM public.video_performers vp JOIN public.performers p ON p.id = vp.performer_id WHERE vp.video_id = v.id), '[]'::jsonb),
    coalesce((SELECT jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name)) FROM public.video_tags vt JOIN public.tags t ON t.id = vt.tag_id LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id WHERE vt.video_id = v.id AND coalesce(tg.show_in_ui, true)), '[]'::jsonb)
  FROM public.videos v
  WHERE v.sample_video_url IS NOT NULL
    AND (array_length(exclude_ids, 1) IS NULL OR v.id != ALL(exclude_ids))
    AND EXISTS (SELECT 1 FROM public.video_tags vt WHERE vt.video_id = v.id AND vt.tag_id = ANY(tag_ids))
  ORDER BY random()
  LIMIT p_limit;
END;
$$;
