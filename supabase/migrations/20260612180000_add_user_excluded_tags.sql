-- ユーザーの除外タグ設定テーブル
CREATE TABLE public.user_excluded_tags (
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  tag_id  uuid NOT NULL REFERENCES public.tags(id) ON DELETE CASCADE,
  created_at timestamptz DEFAULT now(),
  PRIMARY KEY (user_id, tag_id)
);

ALTER TABLE public.user_excluded_tags ENABLE ROW LEVEL SECURITY;

CREATE POLICY "user_excluded_tags_owner_all"
  ON public.user_excluded_tags FOR ALL USING (auth.uid() = user_id);

-- 新規ユーザーはデフォルトでスカトロを除外
CREATE OR REPLACE FUNCTION public.set_default_excluded_tags()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO public.user_excluded_tags (user_id, tag_id)
  SELECT NEW.user_id, t.id
  FROM public.tags t
  WHERE t.name IN ('スカトロ', '人格排泄')
  ON CONFLICT DO NOTHING;
  RETURN NEW;
END;
$$;

CREATE TRIGGER trg_set_default_excluded_tags
  AFTER INSERT ON public.user_profiles
  FOR EACH ROW EXECUTE FUNCTION public.set_default_excluded_tags();

-- 除外タグ一覧取得
CREATE OR REPLACE FUNCTION public.get_user_excluded_tags()
RETURNS json LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  RETURN (
    SELECT json_agg(json_build_object('tag_id', uet.tag_id, 'name', t.name))
    FROM public.user_excluded_tags uet
    JOIN public.tags t ON t.id = uet.tag_id
    WHERE uet.user_id = auth.uid()
  );
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_user_excluded_tags() TO authenticated;

-- 除外タグ追加
CREATE OR REPLACE FUNCTION public.add_user_excluded_tag(p_tag_id uuid)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO public.user_excluded_tags (user_id, tag_id)
  VALUES (auth.uid(), p_tag_id)
  ON CONFLICT DO NOTHING;
END;
$$;

GRANT EXECUTE ON FUNCTION public.add_user_excluded_tag(uuid) TO authenticated;

-- 除外タグ削除
CREATE OR REPLACE FUNCTION public.remove_user_excluded_tag(p_tag_id uuid)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  DELETE FROM public.user_excluded_tags
  WHERE user_id = auth.uid() AND tag_id = p_tag_id;
END;
$$;

GRANT EXECUTE ON FUNCTION public.remove_user_excluded_tag(uuid) TO authenticated;
