-- ================================================================
-- 漫画→動画への復元マイグレーション
-- booksテーブル系を削除し、videosテーブル系を復活させる
-- ================================================================

-- 1. booksテーブル系の関数・ビュー・テーブルを削除
-- ----------------------------------------------------------------
drop function if exists public.get_books_recommendations cascade;
drop function if exists public.get_popular_books cascade;
drop function if exists public.get_books_feed cascade;
drop function if exists public.get_user_book_likes cascade;
drop function if exists public.get_user_liked_tags cascade;

drop materialized view if exists public.book_popularity_daily cascade;

drop table if exists public.book_tags cascade;
drop table if exists public.book_embeddings cascade;
drop table if exists public.user_book_decisions cascade;
drop table if exists public.book_likes cascade;
drop table if exists public.books cascade;

-- 2. videosテーブル系を復活
-- ----------------------------------------------------------------
create table public.videos (
  id uuid primary key default gen_random_uuid(),
  external_id text unique,
  title text not null,
  description text,
  duration_seconds int,
  thumbnail_url text,
  thumbnail_vertical_url text,
  preview_video_url text,
  distribution_code text,
  maker_code text,
  director text,
  series text,
  maker text,
  label text,
  price numeric,
  distribution_started_at timestamptz,
  product_released_at timestamptz,
  sample_video_url text,
  image_urls text[],
  source text not null,
  published_at timestamptz default now(),
  product_url text,
  affiliate_url text,
  created_at timestamptz default now(),
  unique (source, distribution_code, maker_code)
);

create table public.video_tags (
  video_id uuid references public.videos(id) on delete cascade,
  tag_id uuid references public.tags(id) on delete cascade,
  primary key (video_id, tag_id)
);

create table public.video_performers (
  video_id uuid references public.videos(id) on delete cascade,
  performer_id uuid references public.performers(id) on delete cascade,
  primary key (video_id, performer_id)
);

create table public.video_embeddings (
  video_id uuid primary key references public.videos(id) on delete cascade,
  embedding halfvec(128),
  model_version text,
  updated_at timestamptz default now()
);

create table public.user_video_decisions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  anonymous_session_id text,
  video_id uuid references public.videos(id) on delete cascade,
  decision_type text not null check (decision_type in ('like', 'nope')),
  recommendation_source text,
  recommendation_score float,
  recommendation_model_version text,
  recommendation_params jsonb,
  created_at timestamptz default now()
);

-- 3. マテリアライズドビュー
-- ----------------------------------------------------------------
create materialized view public.video_popularity_daily as
select
  uvd.video_id,
  date_trunc('day', uvd.created_at at time zone 'UTC') as d,
  count(*) as likes
from public.user_video_decisions uvd
where uvd.decision_type = 'like'
group by uvd.video_id, date_trunc('day', uvd.created_at at time zone 'UTC');

create unique index mv_pop_daily_uidx on public.video_popularity_daily (video_id, d);

-- 4. インデックス
-- ----------------------------------------------------------------
create index on public.videos (source);
create index on public.videos (product_released_at desc);
create index on public.videos (maker_code);
create index on public.user_video_decisions (user_id, created_at desc);
create index on public.user_video_decisions (video_id);

-- 5. RLS
-- ----------------------------------------------------------------
alter table public.videos enable row level security;
create policy "read videos" on public.videos for select to anon, authenticated using (true);
create policy "insert videos" on public.videos for insert to anon, authenticated, service_role with check (true);
create policy "update videos" on public.videos for update to service_role using (true);

alter table public.video_tags enable row level security;
create policy "read video_tags" on public.video_tags for select to anon, authenticated using (true);
create policy "insert video_tags" on public.video_tags for insert to anon, authenticated, service_role with check (true);

alter table public.video_performers enable row level security;
create policy "read video_performers" on public.video_performers for select to anon, authenticated using (true);
create policy "insert video_performers" on public.video_performers for insert to anon, authenticated, service_role with check (true);

alter table public.video_embeddings enable row level security;
create policy "read video_embeddings" on public.video_embeddings for select to anon, authenticated using (true);
create policy "upsert video_embeddings" on public.video_embeddings for all to service_role using (true);

alter table public.user_video_decisions enable row level security;
create policy "insert own decisions" on public.user_video_decisions for insert to authenticated, anon with check (true);
create policy "read own decisions" on public.user_video_decisions for select to authenticated using (auth.uid() = user_id);

grant select on public.video_popularity_daily to anon, authenticated, service_role;
grant all on public.videos to service_role;
grant all on public.video_tags to service_role;
grant all on public.video_performers to service_role;
grant all on public.video_embeddings to service_role;
grant all on public.user_video_decisions to service_role;

-- user_embeddingsリセット
truncate table public.user_embeddings;

-- 6. 関数群を復元
-- ----------------------------------------------------------------

-- get_videos_feed
create or replace function public.get_videos_feed(page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  thumbnail_vertical_url text,
  sample_video_url text,
  preview_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb
)
language plpgsql security definer set search_path = public
as $$
declare
  total_count int;
  start_offset int := 0;
begin
  select count(*) into total_count
  from public.videos v
  where not exists (
    select 1 from public.user_video_decisions uvd
    where uvd.video_id = v.id
      and auth.uid() is not null and uvd.user_id = auth.uid()
  );

  if total_count > page_limit then
    start_offset := floor(random() * (total_count - page_limit))::int;
  end if;

  return query
  select
    v.id, v.title, v.description, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url,
    v.sample_video_url, v.preview_video_url,
    coalesce(v.affiliate_url, v.product_url) as product_url,
    v.product_released_at,
    coalesce((
      select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
      from public.video_performers vp join public.performers p on p.id = vp.performer_id
      where vp.video_id = v.id
    ), '[]'::jsonb) as performers,
    coalesce((
      select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
      from public.video_tags vt join public.tags t on t.id = vt.tag_id
      left join public.tag_groups tg on tg.id = t.tag_group_id
      where vt.video_id = v.id and coalesce(tg.show_in_ui, true)
    ), '[]'::jsonb) as tags
  from public.videos v
  where v.id in (
    select v2.id from public.videos v2
    where not exists (
      select 1 from public.user_video_decisions uvd2
      where uvd2.video_id = v2.id
        and auth.uid() is not null and uvd2.user_id = auth.uid()
    )
    order by v2.id
    offset start_offset limit page_limit
  )
  order by (v.sample_video_url is null), v.id;
end;
$$;

grant execute on function public.get_videos_feed(int) to anon, authenticated;

-- get_videos_recommendations（maker diversity reranking付き最新版）
create or replace function public.get_videos_recommendations(user_uuid uuid, page_limit int default 20)
returns table (
  id uuid, title text, description text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, product_url text,
  product_released_at timestamptz,
  performers jsonb, tags jsonb,
  score double precision, model_version text
)
language plpgsql security definer set search_path = public
as $$
declare
  user_vec halfvec(128);
  approx_vec halfvec(128);
  latest_version text;
  candidate_limit int;
begin
  candidate_limit := page_limit * 5;

  select avg(ve.embedding)::halfvec(128) into approx_vec
  from (
    select uvd.video_id from public.user_video_decisions uvd
    where uvd.user_id = user_uuid and uvd.decision_type = 'like'
    order by uvd.created_at desc limit 200
  ) recent_likes
  join public.video_embeddings ve on ve.video_id = recent_likes.video_id;

  if approx_vec is null then
    select ue.embedding into user_vec from public.user_embeddings ue where ue.user_id = user_uuid;
  end if;

  user_vec := coalesce(approx_vec, user_vec);
  if user_vec is null then return; end if;

  select ve.model_version into latest_version
  from public.video_embeddings ve where ve.model_version is not null
  order by ve.updated_at desc limit 1;

  return query
  with candidates as (
    select
      v.id, v.title, v.description, v.external_id,
      v.thumbnail_url, v.thumbnail_vertical_url, v.sample_video_url,
      coalesce(v.affiliate_url, v.product_url) as product_url,
      v.product_released_at,
      coalesce(v.maker, 'unknown') as maker,
      coalesce((
        select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
        from public.video_performers vp join public.performers p on p.id = vp.performer_id
        where vp.video_id = v.id
      ), '[]'::jsonb) as performers,
      coalesce((
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
        from public.video_tags vt join public.tags t on t.id = vt.tag_id
        left join public.tag_groups tg on tg.id = t.tag_group_id
        where vt.video_id = v.id and coalesce(tg.show_in_ui, true)
      ), '[]'::jsonb) as tags,
      1 - (ve.embedding <-> user_vec) as raw_score,
      ve.model_version
    from public.video_embeddings ve
    join public.videos v on v.id = ve.video_id
    where v.sample_video_url is not null
      and (latest_version is null or ve.model_version = latest_version)
      and not exists (
        select 1 from public.user_video_decisions uvd
        where uvd.video_id = v.id and uvd.user_id = user_uuid
      )
    order by ve.embedding <-> user_vec
    limit candidate_limit
  ),
  ranked as (
    select *,
      raw_score / (1.0 + (row_number() over (partition by maker order by raw_score desc) - 1) * 0.3) as diversified_score
    from candidates
  )
  select r.id, r.title, r.description, r.external_id,
    r.thumbnail_url, r.thumbnail_vertical_url, r.sample_video_url,
    r.product_url, r.product_released_at,
    r.performers, r.tags,
    r.diversified_score as score, r.model_version
  from ranked r
  order by r.diversified_score desc
  limit page_limit;
end;
$$;

grant execute on function public.get_videos_recommendations(uuid, int) to anon, authenticated;

-- get_popular_videos
create or replace function public.get_popular_videos(user_uuid uuid default null, limit_count int default 20, lookback_days int default 7)
returns table (
  id uuid, title text, description text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, product_url text,
  product_released_at timestamptz,
  performers jsonb, tags jsonb,
  score double precision
)
language plpgsql security definer set search_path = public
as $$
begin
  return query
  with popularity as (
    select vpd.video_id, sum(vpd.likes)::double precision as total_likes
    from public.video_popularity_daily vpd
    where vpd.d >= (now() - make_interval(days => lookback_days))
    group by vpd.video_id
  )
  select
    v.id, v.title, v.description, v.external_id,
    v.thumbnail_url, v.thumbnail_vertical_url, v.sample_video_url,
    coalesce(v.affiliate_url, v.product_url) as product_url,
    v.product_released_at,
    coalesce((
      select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
      from public.video_performers vp join public.performers p on p.id = vp.performer_id
      where vp.video_id = v.id
    ), '[]'::jsonb) as performers,
    coalesce((
      select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
      from public.video_tags vt join public.tags t on t.id = vt.tag_id
      left join public.tag_groups tg on tg.id = t.tag_group_id
      where vt.video_id = v.id and coalesce(tg.show_in_ui, true)
    ), '[]'::jsonb) as tags,
    pop.total_likes as score
  from popularity pop
  join public.videos v on v.id = pop.video_id
  where v.sample_video_url is not null
    and (user_uuid is null or not exists (
      select 1 from public.user_video_decisions uvd
      where uvd.user_id = user_uuid and uvd.video_id = v.id
    ))
  order by pop.total_likes desc nulls last, v.product_released_at desc
  limit limit_count;
end;
$$;

grant execute on function public.get_popular_videos(uuid, int, int) to anon, authenticated;

-- get_similar_videos
create or replace function public.get_similar_videos(p_video_id uuid, p_limit int default 8)
returns table (
  id uuid, title text, thumbnail_url text, thumbnail_vertical_url text,
  affiliate_url text, product_url text, external_id text,
  maker text, product_released_at timestamptz, similarity float8
)
language sql stable as $$
  select v.id, v.title, v.thumbnail_url, v.thumbnail_vertical_url,
    v.affiliate_url, v.product_url, v.external_id, v.maker,
    v.product_released_at,
    1.0 - (ve_other.embedding <=> ve_base.embedding) as similarity
  from video_embeddings ve_base
  join video_embeddings ve_other on ve_other.video_id != ve_base.video_id
    and ve_other.model_version = ve_base.model_version
  join videos v on v.id = ve_other.video_id
  where ve_base.video_id = p_video_id
  order by ve_other.embedding <=> ve_base.embedding
  limit p_limit;
$$;

grant execute on function public.get_similar_videos(uuid, int) to anon, authenticated;

-- get_user_likes
create or replace function public.get_user_likes(
  p_search text default null,
  p_sort text default 'liked_at',
  p_order text default 'desc',
  p_limit int default 20,
  p_offset int default 0,
  p_tag_ids uuid[] default null,
  p_performer_ids uuid[] default null,
  p_price_min numeric default null,
  p_price_max numeric default null,
  p_release_gte timestamptz default null,
  p_release_lte timestamptz default null
)
returns table (
  id uuid, external_id text, title text, description text,
  thumbnail_url text, thumbnail_vertical_url text,
  product_url text, price numeric,
  product_released_at timestamptz, liked_at timestamptz,
  performers jsonb, tags jsonb, total_count bigint
)
language sql security invoker as $$
  with base as (
    select uvd.video_id, uvd.created_at as liked_at
    from public.user_video_decisions uvd
    where uvd.user_id = auth.uid() and uvd.decision_type = 'like'
  ), v as (
    select v.id, v.external_id, v.title, v.description,
      v.thumbnail_url, v.thumbnail_vertical_url,
      coalesce(v.affiliate_url, v.product_url) as product_url,
      v.price, v.product_released_at, b.liked_at
    from public.videos v join base b on b.video_id = v.id
    where (p_search is null or v.title ilike ('%' || p_search || '%'))
      and (p_price_min is null or v.price >= p_price_min)
      and (p_price_max is null or v.price <= p_price_max)
      and (p_release_gte is null or v.product_released_at >= p_release_gte)
      and (p_release_lte is null or v.product_released_at <= p_release_lte)
  ), vpt as (
    select v.id, jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name) as performers
    from v left join public.video_performers vp on vp.video_id = v.id
    left join public.performers p on p.id = vp.performer_id group by v.id
  ), vtg as (
    select v.id, jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name) as tags
    from v left join public.video_tags vt on vt.video_id = v.id
    left join public.tags t on t.id = vt.tag_id group by v.id
  ), vjoined as (
    select v.*, coalesce(vpt.performers, '[]'::jsonb) as performers, coalesce(vtg.tags, '[]'::jsonb) as tags
    from v left join vpt on vpt.id = v.id left join vtg on vtg.id = v.id
  ), vfiltered as (
    select * from vjoined
    where (p_tag_ids is null or exists (
      select 1 from jsonb_array_elements(tags) t(obj) where (t.obj->>'id')::uuid = any(p_tag_ids)
    ))
    and (p_performer_ids is null or exists (
      select 1 from jsonb_array_elements(performers) p(obj) where (p.obj->>'id')::uuid = any(p_performer_ids)
    ))
  )
  select id, external_id, title, description, thumbnail_url, thumbnail_vertical_url,
    product_url, price, product_released_at, liked_at, performers, tags,
    count(*) over () as total_count
  from vfiltered
  order by
    case when lower(p_sort) = 'liked_at' and lower(p_order) = 'asc' then liked_at end asc,
    case when lower(p_sort) = 'liked_at' and lower(p_order) = 'desc' then liked_at end desc,
    case when lower(p_sort) = 'released' and lower(p_order) = 'asc' then product_released_at end asc,
    case when lower(p_sort) = 'released' and lower(p_order) = 'desc' then product_released_at end desc,
    case when lower(p_sort) = 'price' and lower(p_order) = 'asc' then price end asc,
    case when lower(p_sort) = 'price' and lower(p_order) = 'desc' then price end desc,
    case when lower(p_sort) = 'title' and lower(p_order) = 'asc' then title end asc,
    case when lower(p_sort) = 'title' and lower(p_order) = 'desc' then title end desc,
    id
  limit greatest(p_limit, 1) offset greatest(p_offset, 0);
$$;

grant execute on function public.get_user_likes(text, text, text, int, int, uuid[], uuid[], numeric, numeric, timestamptz, timestamptz) to authenticated;

-- get_user_liked_tags
create or replace function public.get_user_liked_tags()
returns table (id uuid, name text, cnt bigint, tag_group_name text)
language sql security invoker as $$
  select t.id, t.name, count(*)::bigint as cnt, tg.name as tag_group_name
  from public.user_video_decisions uvd
  join public.video_tags vt on vt.video_id = uvd.video_id
  join public.tags t on t.id = vt.tag_id
  left join public.tag_groups tg on tg.id = t.tag_group_id
  where uvd.user_id = auth.uid()
    and uvd.decision_type = 'like'
    and coalesce(tg.show_in_ui, true)
  group by t.id, t.name, tg.name
  order by cnt desc, t.name asc;
$$;

grant execute on function public.get_user_liked_tags() to authenticated;

-- get_tags_with_count
create or replace function public.get_tags_with_count()
returns table (id uuid, name text, video_count bigint)
language sql stable as $$
  select t.id, t.name, count(vt.video_id) as video_count
  from tags t left join video_tags vt on vt.tag_id = t.id
  group by t.id, t.name order by video_count desc, t.name;
$$;

grant execute on function public.get_tags_with_count() to anon, authenticated;

-- get_performers_with_count
create or replace function public.get_performers_with_count(p_limit int default 60, p_offset int default 0)
returns table (id uuid, name text, video_count bigint)
language sql stable as $$
  select p.id, p.name, count(vp.video_id) as video_count
  from performers p left join video_performers vp on vp.performer_id = p.id
  group by p.id, p.name order by video_count desc, p.name
  limit p_limit offset p_offset;
$$;

grant execute on function public.get_performers_with_count(int, int) to anon, authenticated;

create or replace function public.get_performers_count()
returns bigint language sql stable as $$
  select count(*) from performers;
$$;

grant execute on function public.get_performers_count() to anon, authenticated;
