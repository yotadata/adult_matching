-- 重複行を削除（最新のものを残す）
DELETE FROM public.user_video_decisions
WHERE id NOT IN (
  SELECT DISTINCT ON (user_id, video_id) id
  FROM public.user_video_decisions
  ORDER BY user_id, video_id, created_at DESC
);

-- ユニーク制約を追加（ON CONFLICT (user_id, video_id) に必要、冪等）
DO $$ BEGIN
  ALTER TABLE public.user_video_decisions
    ADD CONSTRAINT user_video_decisions_user_id_video_id_key
    UNIQUE (user_id, video_id);
EXCEPTION WHEN duplicate_table THEN NULL;
END $$;
