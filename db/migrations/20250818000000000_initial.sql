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

-- LIKE管理
create table public.likes (
  user_id uuid references auth.users(id) on delete cascade,
  video_id uuid references public.videos(id) on delete cascade,
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
  date_trunc('day', l.created_at) as d,
  count(l.video_id) as likes
from videos v
left join likes l on l.video_id = v.id
group by v.id, date_trunc('day', l.created_at);

-- RLS
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
