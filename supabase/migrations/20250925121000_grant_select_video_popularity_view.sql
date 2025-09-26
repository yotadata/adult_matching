-- Ensure anon/authenticated can read the popularity materialized view
grant usage on schema public to anon, authenticated;
grant select on public.video_popularity_daily to anon, authenticated;

