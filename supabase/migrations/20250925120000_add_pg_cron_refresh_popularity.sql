-- Enable pg_cron (if not already enabled)
create extension if not exists pg_cron;

-- Unique index required for CONCURRENT REFRESH on materialized view
create unique index if not exists mv_pop_daily_uidx
  on public.video_popularity_daily (video_id, d);

-- Unschedule existing job with the same name (idempotent)
do $$
declare
  jid int;
begin
  select jobid into jid from cron.job where jobname = 'refresh_video_popularity_daily_hourly' limit 1;
  if jid is not null then
    perform cron.unschedule(jid);
  end if;
end $$;

-- Schedule hourly refresh at minute 15 (UTC) to avoid top-of-hour spikes
select cron.schedule(
  'refresh_video_popularity_daily_hourly',
  '15 * * * *',
  $$refresh materialized view concurrently public.video_popularity_daily;$$
);

