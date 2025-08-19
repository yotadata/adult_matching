-- Enable extensions
create extension if not exists "uuid-ossp";
create extension if not exists "pgcrypto";
create extension if not exists vector;

-- 動画テーブル
create table public.videos (
  id uuid primary key default gen_random_uuid(),
  external_id text unique,
  title text not null,
  description text,
  categories text[],
  performers text[],
  duration_seconds int,
  thumbnail_url text,
  preview_video_url text,
  source text not null,
  published_at timestamptz,
  created_at timestamptz default now()
);

-- 動画埋め込み
create table public.video_embeddings (
  video_id uuid primary key references public.videos(id) on delete cascade,
  embedding vector(768),
  updated_at timestamptz default now()
);

-- プロフィール
create table public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz default now()
);

-- 行動ログ
create type public.event_type as enum ('view', 'like', 'dislike', 'dismiss');
create table public.user_events (
  id bigserial primary key,
  user_id uuid references auth.users(id) on delete cascade,
  video_id uuid references public.videos(id) on delete cascade,
  type public.event_type not null,
  occurred_at timestamptz default now()
);

-- LIKE管理
create table public.likes (
  user_id uuid references auth.users(id) on delete cascade,
  video_id uuid references public.videos(id) on delete cascade,
  purchased boolean default false,
  created_at timestamptz default now(),
  primary key (user_id, video_id)
);

-- ユーザー埋め込み
create table public.user_embeddings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  embedding vector(768),
  updated_at timestamptz default now()
);

-- 人気集計ビュー
create materialized view public.video_popularity_daily as
select
  v.id as video_id,
  date_trunc('day', e.occurred_at) as d,
  count(*) filter (where e.type='view') as views,
  count(*) filter (where e.type='like') as likes,
  count(*) filter (where e.type='dismiss') as dismisses
from videos v
left join user_events e on e.video_id = v.id
group by v.id, date_trunc('day', e.occurred_at);

-- RLS
alter table public.user_events enable row level security;
create policy "insert own events"
on public.user_events for insert
to authenticated
with check (auth.uid() = user_id);

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

alter table public.videos enable row level security;
create policy "read videos"
on public.videos for select
to anon, authenticated
using (true);

alter table public.video_embeddings enable row level security;
create policy "read embeddings"
on public.video_embeddings for select
to anon, authenticated
using (true);

alter table public.user_embeddings enable row level security;
create policy "read own embedding"
on public.user_embeddings for select
to authenticated
using (auth.uid() = user_id);
create policy "update own embedding"
on public.user_embeddings for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
