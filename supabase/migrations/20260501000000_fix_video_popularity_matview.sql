-- video_popularity_daily を likes テーブルから user_video_decisions ベースに修正
--
-- 問題: matview が旧 likes テーブルを参照しており、user_video_decisions 移行後は
--       常に空だった。get_popular_videos が 0 件を返すのはこれが原因。

drop materialized view if exists public.video_popularity_daily cascade;

create materialized view public.video_popularity_daily as
select
  uvd.video_id,
  date_trunc('day', uvd.created_at at time zone 'UTC') as d,
  count(*) as likes
from public.user_video_decisions uvd
where uvd.decision_type = 'like'
group by uvd.video_id, date_trunc('day', uvd.created_at at time zone 'UTC');

-- CONCURRENT REFRESH に必要なユニークインデックス
create unique index mv_pop_daily_uidx
  on public.video_popularity_daily (video_id, d);

-- 初回データを即時投入
refresh materialized view public.video_popularity_daily;

-- アクセス権限を再付与
grant select on public.video_popularity_daily to anon, authenticated, service_role;
