-- RLS有効化（読み取り専用ポリシー付き）
ALTER TABLE public.tag_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.performers ENABLE ROW LEVEL SECURITY;

-- 全員（未ログイン含む）が読み取り可能
CREATE POLICY "誰でも読み取り可能" ON public.tag_groups FOR SELECT USING (true);
CREATE POLICY "誰でも読み取り可能" ON public.tags FOR SELECT USING (true);
CREATE POLICY "誰でも読み取り可能" ON public.performers FOR SELECT USING (true);
