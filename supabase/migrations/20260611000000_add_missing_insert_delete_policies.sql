-- public_lists / public_list_videos の INSERT・DELETE ポリシー
-- （リモートに直接適用済みのため、ローカル追跡用に記録）

DROP POLICY IF EXISTS "public_lists_insert" ON public.public_lists;
CREATE POLICY "public_lists_insert"
  ON public.public_lists FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "public_lists_delete" ON public.public_lists;
CREATE POLICY "public_lists_delete"
  ON public.public_lists FOR DELETE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "public_list_videos_insert" ON public.public_list_videos;
CREATE POLICY "public_list_videos_insert"
  ON public.public_list_videos FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.public_lists pl
      WHERE pl.id = list_id AND pl.user_id = auth.uid()
    )
  );
