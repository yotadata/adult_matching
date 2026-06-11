-- get_videos_feed の COUNT/WHERE 条件を高速化する部分インデックス
CREATE INDEX IF NOT EXISTS videos_sample_source_idx
  ON public.videos (source)
  WHERE sample_video_url IS NOT NULL;
