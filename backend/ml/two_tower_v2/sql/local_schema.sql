-- Basic schema for local PostgreSQL instance used by Two-Tower pipeline.
-- The schema is intentionally minimal and only covers the columns required by
-- feature engineeringおよび学習処理。

create extension if not exists vector;

create table if not exists public.profiles (
    user_id uuid primary key,
    display_name text,
    created_at timestamptz
);

create table if not exists public.videos (
    id uuid primary key,
    external_id text,
    title text,
    description text,
    duration_seconds integer,
    thumbnail_url text,
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
    image_urls jsonb,
    source text,
    published_at timestamptz,
    product_url text,
    created_at timestamptz,
    safety_level smallint default 0,
    region_codes text[] default '{}',
    is_active boolean default true
);

create table if not exists public.video_tags (
    video_id uuid references public.videos(id) on delete cascade,
    tag_name text not null,
    primary key (video_id, tag_name)
);

create table if not exists public.video_performers (
    video_id uuid references public.videos(id) on delete cascade,
    performer_name text not null,
    primary key (video_id, performer_name)
);

create table if not exists public.user_video_decisions (
    user_id uuid not null,
    video_id uuid not null,
    decision_type text not null,
    created_at timestamptz,
    primary key (user_id, video_id, decision_type, created_at)
);

create table if not exists public.video_embeddings (
    video_id uuid primary key,
    embedding vector(256),
    updated_at timestamptz default now()
);

create index if not exists idx_user_video_decisions_user_id on public.user_video_decisions(user_id);
create index if not exists idx_user_video_decisions_video_id on public.user_video_decisions(video_id);
create index if not exists idx_video_tags_tag_name on public.video_tags(tag_name);
create index if not exists idx_video_performers_name on public.video_performers(performer_name);

create or replace function public.get_user_embedding_features(user_id uuid)
returns jsonb
language sql
as $$
  with profile as (
    select created_at from public.profiles where user_id = $1
  ),
  decisions as (
    select * from public.user_video_decisions where user_id = $1
  ),
  stats as (
    select
      count(*) filter (where decision_type = 'like') as like_count,
      count(*) filter (where decision_type = 'nope') as nope_count,
      count(*) as decision_count,
      max(created_at) filter (where decision_type = 'like') as recent_like_at,
      avg(extract(hour from created_at)::numeric) filter (where decision_type = 'like') as average_like_hour
    from decisions
  ),
  like_rows as (
    select d.video_id, d.created_at, v.price, v.product_released_at
    from decisions d
    join public.videos v on v.id = d.video_id
    where d.decision_type = 'like'
  ),
  price_stats as (
    select
      avg(price)::numeric as mean,
      percentile_cont(0.5) within group (order by price) as median
    from like_rows
    where price is not null
  ),
  tag_values as (
    select array_agg(vt.tag_name) as tags
    from like_rows lr
    join public.video_tags vt on vt.video_id = lr.video_id
  ),
  performer_values as (
    select array_agg(vp.performer_name) as performers
    from like_rows lr
    join public.video_performers vp on vp.video_id = lr.video_id
  )
  select jsonb_build_object(
    'user_id', $1,
    'profile', jsonb_build_object('created_at', (select created_at from profile)),
    'decision_stats', jsonb_build_object(
      'like_count', coalesce((select like_count from stats), 0),
      'nope_count', coalesce((select nope_count from stats), 0),
      'decision_count', coalesce((select decision_count from stats), 0),
      'recent_like_at', (select recent_like_at from stats),
      'average_like_hour', coalesce((select average_like_hour from stats), 12.0)
    ),
    'price_stats', jsonb_build_object(
      'mean', coalesce((select mean from price_stats), 0),
      'median', coalesce((select median from price_stats), 0)
    ),
    'tags', to_jsonb(coalesce((select tags from tag_values), array[]::text[])),
    'performers', to_jsonb(coalesce((select performers from performer_values), array[]::text[]))
  );
$$;

create or replace function public.get_item_embedding_features(video_id uuid)
returns jsonb
language sql
as $$
  with base as (
    select * from public.videos where id = $1
  ),
  tag_values as (
    select array_agg(tag_name order by tag_name) as tags
    from public.video_tags where video_id = $1
  ),
  performer_values as (
    select array_agg(performer_name order by performer_name) as performers
    from public.video_performers where video_id = $1
  )
  select jsonb_build_object(
    'video_id', $1,
    'price', (select price from base),
    'duration_seconds', (select duration_seconds from base),
    'product_released_at', (select product_released_at from base),
    'distribution_started_at', (select distribution_started_at from base),
    'safety_level', (select safety_level from base),
    'region_codes', to_jsonb(coalesce((select region_codes from base), array[]::text[])),
    'tags', to_jsonb(coalesce((select tags from tag_values), array[]::text[])),
    'performers', to_jsonb(coalesce((select performers from performer_values), array[]::text[]))
  );
$$;

create or replace function public.get_item_embedding_feature_batch(limit_count int default 100, offset_count int default 0)
returns table (video_id uuid, payload jsonb)
language sql
as $$
  with limited as (
    select id
    from public.videos
    where is_active
    order by id
    limit greatest(1, least(coalesce(limit_count, 100), 1000))
    offset greatest(0, coalesce(offset_count, 0))
  )
  select l.id, public.get_item_embedding_features(l.id)
  from limited l;
$$;

create or replace function public.get_user_embedding_feature_batch(limit_count int default 100, offset_count int default 0)
returns table (user_id uuid, payload jsonb)
language sql
as $$
  with limited as (
    select distinct user_id
    from public.user_video_decisions
    order by user_id
    limit greatest(1, least(coalesce(limit_count, 100), 1000))
    offset greatest(0, coalesce(offset_count, 0))
  )
  select l.user_id, public.get_user_embedding_features(l.user_id)
  from limited l;
$$;
