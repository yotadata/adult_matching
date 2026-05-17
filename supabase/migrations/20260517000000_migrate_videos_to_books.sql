-- ================================================================
-- 動画→漫画への完全移行マイグレーション
-- 既存のvideosテーブル系を削除し、booksテーブル系を新設する
-- ================================================================

-- 1. 既存テーブル・ビュー・関数の削除（依存関係の逆順）
-- ----------------------------------------------------------------

-- 関数
drop function if exists public.get_videos_recommendations cascade;
drop function if exists public.get_popular_videos cascade;
drop function if exists public.get_videos_feed cascade;
drop function if exists public.search_videos cascade;
drop function if exists public.search_videos_by_embedding cascade;
drop function if exists public.get_user_likes cascade;
drop function if exists public.get_user_liked_tags cascade;
drop function if exists public.get_user_likes_total_count cascade;
drop function if exists public.get_similar_videos cascade;
drop function if exists public.get_list_page_videos cascade;
drop function if exists public.update_user_features cascade;

-- マテリアライズドビュー
drop materialized view if exists public.video_popularity_daily cascade;

-- テーブル（依存関係順）
drop table if exists public.video_tags cascade;
drop table if exists public.video_performers cascade;
drop table if exists public.video_embeddings cascade;
drop table if exists public.likes cascade;
drop table if exists public.user_video_decisions cascade;
drop table if exists public.videos cascade;

-- 2. 漫画テーブル新設
-- ----------------------------------------------------------------

create table public.books (
  id uuid primary key default gen_random_uuid(),
  external_id text unique not null,          -- FANZAのcontent_id
  title text not null,
  description text,
  page_count int,                             -- 総ページ数
  thumbnail_url text,                         -- 表紙サムネ（横）
  thumbnail_vertical_url text,               -- 表紙サムネ（縦）
  sample_image_urls text[],                  -- サンプル画像URL配列
  author text,                               -- 著者名
  author_id text,                            -- FANZAの著者ID
  series text,                               -- シリーズ名
  publisher text,                            -- 出版社
  label text,                                -- レーベル
  price numeric,                             -- 価格
  product_url text,                          -- FANZA商品URL
  affiliate_url text,                        -- アフィリエイトURL
  product_released_at timestamptz,           -- 発売日
  source text not null default 'FANZA',
  created_at timestamptz default now()
);

-- 3. 漫画タグ（既存のtags/tag_groupsは流用）
create table public.book_tags (
  book_id uuid references public.books(id) on delete cascade,
  tag_id uuid references public.tags(id) on delete cascade,
  primary key (book_id, tag_id)
);

-- 4. 漫画埋め込み（推薦用）
create table public.book_embeddings (
  book_id uuid primary key references public.books(id) on delete cascade,
  embedding halfvec(128),
  model_version text,
  updated_at timestamptz default now()
);

-- 5. スワイプ決定（漫画版）
create table public.user_book_decisions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  anonymous_session_id text,
  book_id uuid references public.books(id) on delete cascade,
  decision_type text not null check (decision_type in ('like', 'nope')),
  recommendation_source text,
  recommendation_score float,
  recommendation_model_version text,
  recommendation_params jsonb,
  created_at timestamptz default now()
);

-- 6. お気に入り（漫画版）
create table public.book_likes (
  user_id uuid references auth.users(id) on delete cascade,
  book_id uuid references public.books(id) on delete cascade,
  purchased boolean default false,
  created_at timestamptz default now(),
  primary key (user_id, book_id)
);

-- 7. 人気集計ビュー（漫画版）
create materialized view public.book_popularity_daily as
select
  b.id as book_id,
  date_trunc('day', d.created_at) as day,
  count(*) filter (where d.decision_type = 'like') as likes
from books b
left join user_book_decisions d on d.book_id = b.id
group by b.id, date_trunc('day', d.created_at);

create unique index on public.book_popularity_daily (book_id, day);

-- 8. インデックス
create index on public.books (source);
create index on public.books (product_released_at desc);
create index on public.books (author_id);
create index on public.user_book_decisions (user_id, created_at desc);
create index on public.user_book_decisions (book_id);

-- 9. RLS
alter table public.books enable row level security;
create policy "read books" on public.books for select to anon, authenticated using (true);
create policy "insert books" on public.books for insert to anon, authenticated, service_role with check (true);
create policy "update books" on public.books for update to service_role using (true);

alter table public.book_tags enable row level security;
create policy "read book_tags" on public.book_tags for select to anon, authenticated using (true);
create policy "insert book_tags" on public.book_tags for insert to anon, authenticated, service_role with check (true);

alter table public.book_embeddings enable row level security;
create policy "read book_embeddings" on public.book_embeddings for select to anon, authenticated using (true);
create policy "upsert book_embeddings" on public.book_embeddings for all to service_role using (true);

alter table public.user_book_decisions enable row level security;
create policy "insert own decisions" on public.user_book_decisions for insert to authenticated, anon with check (true);
create policy "read own decisions" on public.user_book_decisions for select to authenticated using (auth.uid() = user_id);

alter table public.book_likes enable row level security;
create policy "select own book_likes" on public.book_likes for select to authenticated using (auth.uid() = user_id);
create policy "insert own book_likes" on public.book_likes for insert to authenticated with check (auth.uid() = user_id);
create policy "delete own book_likes" on public.book_likes for delete to authenticated using (auth.uid() = user_id);

-- 10. ユーザー埋め込みのリセット（次元・データ流用可）
truncate table public.user_embeddings;

-- 11. 推薦関数（books版）
create or replace function public.get_books_recommendations(
  p_user_id uuid,
  p_model_version text default null,
  p_limit int default 20,
  p_offset int default 0
)
returns table (
  book_id uuid,
  score float
)
language sql stable
as $$
  select
    be.book_id,
    (ue.embedding <#> be.embedding)::float * -1 as score
  from user_embeddings ue
  cross join book_embeddings be
  where ue.user_id = p_user_id
    and (p_model_version is null or be.model_version = p_model_version)
    and be.book_id not in (
      select book_id from user_book_decisions
      where user_id = p_user_id
    )
  order by ue.embedding <#> be.embedding
  limit p_limit
  offset p_offset;
$$;

create or replace function public.get_popular_books(
  p_user_id uuid default null,
  p_anonymous_session_id text default null,
  p_days int default 7,
  p_limit int default 20,
  p_offset int default 0
)
returns table (
  book_id uuid,
  score float
)
language sql stable
as $$
  select
    bpd.book_id,
    sum(bpd.likes)::float as score
  from book_popularity_daily bpd
  where bpd.day >= now() - (p_days || ' days')::interval
    and bpd.book_id not in (
      select book_id from user_book_decisions
      where (p_user_id is not null and user_id = p_user_id)
         or (p_anonymous_session_id is not null and anonymous_session_id = p_anonymous_session_id)
    )
  group by bpd.book_id
  order by score desc
  limit p_limit
  offset p_offset;
$$;

create or replace function public.get_books_feed(
  p_user_id uuid default null,
  p_anonymous_session_id text default null,
  p_limit int default 20,
  p_tag_ids uuid[] default null
)
returns table (
  book_id uuid
)
language sql stable
as $$
  select b.id as book_id
  from books b
  where b.id not in (
    select book_id from user_book_decisions
    where (p_user_id is not null and user_id = p_user_id)
       or (p_anonymous_session_id is not null and anonymous_session_id = p_anonymous_session_id)
  )
  and (
    p_tag_ids is null
    or exists (
      select 1 from book_tags bt
      where bt.book_id = b.id and bt.tag_id = any(p_tag_ids)
    )
  )
  order by random()
  limit p_limit;
$$;

-- service_roleへの権限付与
grant all on public.books to service_role;
grant all on public.book_tags to service_role;
grant all on public.book_embeddings to service_role;
grant all on public.user_book_decisions to service_role;
grant all on public.book_likes to service_role;
grant all on public.book_popularity_daily to service_role;
