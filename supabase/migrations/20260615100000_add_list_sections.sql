-- =============================================
-- リストセクション機能
-- ランキングはセクションの1表示形態（display_mode='ranked'）
-- =============================================

CREATE TABLE IF NOT EXISTS public.public_list_sections (
  id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  list_id      uuid        NOT NULL REFERENCES public.public_lists(id) ON DELETE CASCADE,
  title        text,
  display_mode text        NOT NULL DEFAULT 'ranked'
                           CHECK (display_mode IN ('ranked', 'plain')),
  sort_order   integer     NOT NULL DEFAULT 0,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_public_list_sections_list
  ON public.public_list_sections(list_id, sort_order);

ALTER TABLE public.public_list_videos
  ADD COLUMN IF NOT EXISTS section_id uuid
  REFERENCES public.public_list_sections(id) ON DELETE SET NULL;

ALTER TABLE public.public_list_sections ENABLE ROW LEVEL SECURITY;

CREATE POLICY "sections_select" ON public.public_list_sections FOR SELECT USING (
  EXISTS (SELECT 1 FROM public.public_lists pl WHERE pl.id = list_id AND pl.is_active = true)
);
CREATE POLICY "sections_insert" ON public.public_list_sections FOR INSERT WITH CHECK (
  EXISTS (SELECT 1 FROM public.public_lists pl WHERE pl.id = list_id AND pl.user_id = auth.uid())
);
CREATE POLICY "sections_update" ON public.public_list_sections FOR UPDATE USING (
  EXISTS (SELECT 1 FROM public.public_lists pl WHERE pl.id = list_id AND pl.user_id = auth.uid())
);
CREATE POLICY "sections_delete" ON public.public_list_sections FOR DELETE USING (
  EXISTS (SELECT 1 FROM public.public_lists pl WHERE pl.id = list_id AND pl.user_id = auth.uid())
);
