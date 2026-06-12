-- user_video_decisions.user_id に NOT NULL 制約を追加
-- user_id なしでいいねが保存されるバグを DB レベルで防ぐ
ALTER TABLE public.user_video_decisions
  ALTER COLUMN user_id SET NOT NULL;
