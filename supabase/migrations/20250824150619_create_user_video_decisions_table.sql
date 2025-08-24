-- 新しい user_video_decisions テーブルを作成
create table public.user_video_decisions (
  user_id uuid references auth.users(id) on delete cascade,
  video_id uuid references public.videos(id) on delete cascade,
  decision_type text not null, -- 'like' or 'nope'
  created_at timestamptz default now(),
  primary key (user_id, video_id)
);

-- RLS (Row Level Security) を有効化
alter table public.user_video_decisions enable row level security;

-- RLS ポリシーの定義
create policy "select own decisions"
on public.user_video_decisions for select to authenticated
using (auth.uid() = user_id);

create policy "insert own decisions"
on public.user_video_decisions for insert to authenticated
with check (auth.uid() = user_id);

create policy "update own decisions"
on public.user_video_decisions for update to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "delete own decisions"
on public.user_video_decisions for delete to authenticated
using (auth.uid() = user_id);

-- likes テーブルを削除
drop table public.likes;

-- video_popularity_daily materialized view を修正
-- likes テーブルの代わりに user_video_decisions を参照し、decision_type が 'like' のもののみをカウント
drop materialized view public.video_popularity_daily;

create materialized view public.video_popularity_daily as
select
  v.id as video_id,
  date_trunc('day', uvd.created_at) as d,
  count(uvd.video_id) as likes
from videos v
left join user_video_decisions uvd on uvd.video_id = v.id and uvd.decision_type = 'like'
group by v.id, date_trunc('day', uvd.created_at);


-- down.sql の内容

-- video_popularity_daily materialized view を元に戻す (likes テーブルを参照)
drop materialized view public.video_popularity_daily;

create materialized view public.video_popularity_daily as
select
  v.id as video_id,
  date_trunc('day', l.created_at) as d,
  count(l.video_id) as likes
from videos v
left join likes l on l.video_id = v.id
group by v.id, date_trunc('day', l.created_at);

-- likes テーブルを再作成
create table public.likes (
  user_id uuid references auth.users(id) on delete cascade,
  video_id uuid references public.videos(id) on delete cascade,
  purchased boolean default false,
  created_at timestamptz default now(),
  primary key (user_id, video_id)
);

-- RLS を元に戻す
alter table public.likes enable row level security;

create policy "select own likes"
on public.likes for select to authenticated
using (auth.uid() = user_id);

create policy "insert own likes"
on public.likes for insert to authenticated
with check (auth.uid() = user_id);

create policy "delete own likes"
on public.likes for delete to authenticated
using (auth.uid() = user_id);

-- user_video_decisions テーブルを削除
drop table public.user_video_decisions;
