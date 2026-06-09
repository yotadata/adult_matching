ALTER TABLE user_video_decisions
  ADD COLUMN IF NOT EXISTS surface TEXT CHECK (surface IN ('swipe', 'grid'));
