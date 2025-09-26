-- DMM APIから定期的にデータを同期するcron jobを設定
select cron.schedule(
  'dmm-sync-daily',
  '0 2 * * *', -- 毎日午前2時に実行
  $$
  select
    net.http_post(
      url:='https://mfleexehdteobgsyokex.supabase.co/functions/v1/dmm_sync',
      headers:='{"Content-Type": "application/json", "Authorization": "Bearer ' || current_setting('app.service_role_key') || '"}'::jsonb,
      body:='{"page": 1, "limit": 100}'::jsonb
    ) as request_id;
  $$
);

-- DMM APIから新着データを定期的に同期するcron job
select cron.schedule(
  'dmm-sync-new-items',
  '0 */6 * * *', -- 6時間ごとに実行
  $$
  select
    net.http_post(
      url:='https://mfleexehdterobgsyokex.supabase.co/functions/v1/dmm_sync',
      headers:='{"Content-Type": "application/json", "Authorization": "Bearer ' || current_setting('app.service_role_key') || '"}'::jsonb,
      body:='{"page": 1, "limit": 50}'::jsonb
    ) as request_id;
  $$
);