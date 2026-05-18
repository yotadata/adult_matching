-- FANZA_DOUJIN ソースの動画および関連データを削除する
-- ingest 対象から除外したため、既存データも整理する

-- 関連するいいね・決定履歴を先に削除
DELETE FROM user_video_decisions
WHERE video_id IN (
  SELECT id FROM videos WHERE source = 'FANZA_DOUJIN'
);

-- 関連タグを削除
DELETE FROM video_tags
WHERE video_id IN (
  SELECT id FROM videos WHERE source = 'FANZA_DOUJIN'
);

-- 関連出演者を削除
DELETE FROM video_performers
WHERE video_id IN (
  SELECT id FROM videos WHERE source = 'FANZA_DOUJIN'
);

-- 関連埋め込みを削除
DELETE FROM video_embeddings
WHERE video_id IN (
  SELECT id FROM videos WHERE source = 'FANZA_DOUJIN'
);

-- 動画本体を削除
DELETE FROM videos WHERE source = 'FANZA_DOUJIN';
