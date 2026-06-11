-- キュレーター機能: リスト閲覧数・いいね

-- 1. lists テーブルに view_count を追加
ALTER TABLE public.public_lists ADD COLUMN IF NOT EXISTS view_count bigint NOT NULL DEFAULT 0;

-- 2. 閲覧数インクリメント RPC
CREATE OR REPLACE FUNCTION public.increment_list_view(p_token text)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  UPDATE public.public_lists SET view_count = view_count + 1 WHERE token = p_token;
END;
$$;

GRANT EXECUTE ON FUNCTION public.increment_list_view TO anon, authenticated;

-- 3. list_likes テーブル（Cookie/fingerprint ベース）
CREATE TABLE IF NOT EXISTS public.list_likes (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  list_id     uuid NOT NULL REFERENCES public.public_lists(id) ON DELETE CASCADE,
  fingerprint text NOT NULL,
  created_at  timestamptz DEFAULT now(),
  UNIQUE(list_id, fingerprint)
);

ALTER TABLE public.list_likes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "list_likes_public_read"
  ON public.list_likes FOR SELECT USING (true);

CREATE POLICY "list_likes_public_insert"
  ON public.list_likes FOR INSERT WITH CHECK (true);

CREATE POLICY "list_likes_owner_delete"
  ON public.list_likes FOR DELETE USING (true);

-- 4. いいねトグル RPC
CREATE OR REPLACE FUNCTION public.toggle_list_like(p_token text, p_fingerprint text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_list_id uuid;
  v_liked   boolean;
  v_count   bigint;
BEGIN
  SELECT id INTO v_list_id FROM public.public_lists WHERE token = p_token LIMIT 1;
  IF v_list_id IS NULL THEN
    RETURN json_build_object('error', 'not_found');
  END IF;

  IF EXISTS (
    SELECT 1 FROM public.list_likes
    WHERE list_id = v_list_id AND fingerprint = p_fingerprint
  ) THEN
    DELETE FROM public.list_likes WHERE list_id = v_list_id AND fingerprint = p_fingerprint;
    v_liked := false;
  ELSE
    INSERT INTO public.list_likes(list_id, fingerprint) VALUES (v_list_id, p_fingerprint);
    v_liked := true;
  END IF;

  SELECT COUNT(*) INTO v_count FROM public.list_likes WHERE list_id = v_list_id;
  RETURN json_build_object('liked', v_liked, 'count', v_count);
END;
$$;

GRANT EXECUTE ON FUNCTION public.toggle_list_like TO anon, authenticated;

-- 5. いいね状態確認 RPC
CREATE OR REPLACE FUNCTION public.get_list_like_status(p_token text, p_fingerprint text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_list_id uuid;
  v_liked   boolean;
  v_count   bigint;
BEGIN
  SELECT id INTO v_list_id FROM public.public_lists WHERE token = p_token LIMIT 1;
  IF v_list_id IS NULL THEN
    RETURN json_build_object('liked', false, 'count', 0);
  END IF;

  v_liked := EXISTS (
    SELECT 1 FROM public.list_likes
    WHERE list_id = v_list_id AND fingerprint = p_fingerprint
  );
  SELECT COUNT(*) INTO v_count FROM public.list_likes WHERE list_id = v_list_id;
  RETURN json_build_object('liked', v_liked, 'count', v_count);
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_list_like_status TO anon, authenticated;
