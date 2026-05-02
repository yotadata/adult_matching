-- video_popularity_daily の定期リフレッシュを pg_cron で設定
--
-- 問題: 20260501000000 でマテビューの参照テーブルを修正したが、
--       定期 REFRESH が未設定のため新しいいいねデータが反映されず
--       get_popular_videos が常に 0 件を返していた。
--
-- 対策: 毎日 JST 4:00 (UTC 19:00) に CONCURRENT REFRESH を実行。
--       CONCURRENT を使うことでリフレッシュ中もクエリをブロックしない。

-- pg_cron 拡張を有効化（既に有効な場合は何もしない）
create extension if not exists pg_cron with schema extensions;

-- 既存のジョブがあれば削除（idempotent）
select cron.unschedule('refresh_video_popularity_daily')
where exists (
  select 1 from cron.job where jobname = 'refresh_video_popularity_daily'
);

-- 毎日 JST 4:00 (UTC 19:00) に REFRESH
select cron.schedule(
  'refresh_video_popularity_daily',
  '0 19 * * *',
  'REFRESH MATERIALIZED VIEW CONCURRENTLY public.video_popularity_daily'
);

-- 即時リフレッシュ（マイグレーション適用時点のデータを反映）
refresh materialized view concurrently public.video_popularity_daily;
