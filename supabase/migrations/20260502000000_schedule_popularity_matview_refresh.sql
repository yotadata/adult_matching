-- video_popularity_daily の定期リフレッシュを pg_cron で設定
--
-- 問題: 20260501000000 でマテビューの参照テーブルを修正したが、
--       定期 REFRESH が未設定のため新しいいいねデータが反映されず
--       get_popular_videos が常に 0 件を返していた。
--
-- 対策: 毎日 JST 4:00 (UTC 19:00) に CONCURRENT REFRESH を実行。
--       CONCURRENT を使うことでリフレッシュ中もクエリをブロックしない。

do $$
begin
  -- CI や一部ローカル環境では pg_cron パッケージ自体が入っていないため、
  -- 利用可能なときだけ extension 作成とジョブ登録を行う。
  if exists (
    select 1
    from pg_available_extensions
    where name = 'pg_cron'
  ) then
    create extension if not exists pg_cron with schema extensions;

    -- 既存のジョブがあれば削除（idempotent）
    if exists (
      select 1
      from cron.job
      where jobname = 'refresh_video_popularity_daily'
    ) then
      perform cron.unschedule('refresh_video_popularity_daily');
    end if;

    -- 毎日 JST 4:00 (UTC 19:00) に REFRESH
    perform cron.schedule(
      'refresh_video_popularity_daily',
      '0 19 * * *',
      'REFRESH MATERIALIZED VIEW CONCURRENTLY public.video_popularity_daily'
    );
  else
    raise notice 'pg_cron is not available in this environment; skipping cron job setup.';
  end if;
end
$$;

-- 即時リフレッシュ（マイグレーション適用時点のデータを反映）
refresh materialized view concurrently public.video_popularity_daily;
