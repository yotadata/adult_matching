-- Add FANZA metadata columns
alter table public.videos
  add column if not exists affiliate_url text,
  add column if not exists thumbnail_vertical_url text;

alter table public.videos alter column published_at set default now();
update public.videos
set published_at = coalesce(published_at, created_at)
where source = 'FANZA';

-- get_videos_feed: expose vertical thumbnail and affiliate-aware product_url
drop function if exists public.get_videos_feed(integer);
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
language plpgsql
security definer
set search_path = public
as $$
declare
  total_count int;
  start_offset int := 0;
begin
  with filtered as (
    select v.*
    from public.videos v
    where not exists (
      select 1 from public.user_video_decisions uvd
      where uvd.video_id = v.id
        and (auth.uid() is not null and uvd.user_id = auth.uid())
    )
  )
  select count(*) into total_count from filtered;

  if total_count > page_limit then
    start_offset := floor(random() * (total_count - page_limit))::int;
  else
    start_offset := 0;
  end if;

  return query
  select
    v.id,
    v.title,
    v.description,
    v.external_id,
    v.thumbnail_url,
    v.thumbnail_vertical_url,
    v.sample_video_url,
    v.preview_video_url,
    coalesce(v.affiliate_url, v.product_url) as product_url,
    v.product_released_at,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
        from public.video_performers vp
        join public.performers p on p.id = vp.performer_id
        where vp.video_id = v.id
      ), '[]'::jsonb
    ) as performers,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
        from public.video_tags vt
        join public.tags t on t.id = vt.tag_id
        left join public.tag_groups tg on tg.id = t.tag_group_id
        where vt.video_id = v.id
          and coalesce(tg.show_in_ui, true)
      ), '[]'::jsonb
    ) as tags
  from public.videos v
  where v.id in (
    select s.id from (
      select v2.id as id from public.videos v2
      where not exists (
        select 1 from public.user_video_decisions uvd2
        where uvd2.video_id = v2.id
          and (auth.uid() is not null and uvd2.user_id = auth.uid())
      )
      order by v2.id
      offset start_offset limit page_limit
    ) s
  )
  order by (v.sample_video_url is null), v.id;
end;
$$;

grant execute on function public.get_videos_feed(int) to anon, authenticated;

-- get_user_likes: surface vertical thumbnail and affiliate-aware URL
drop function if exists public.get_user_likes(
  text, text, text, int, int, uuid[], uuid[], numeric, numeric, timestamptz, timestamptz
);

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
  id uuid,
  external_id text,
  title text,
  description text,
  thumbnail_url text,
  thumbnail_vertical_url text,
  product_url text,
  price numeric,
  product_released_at timestamptz,
  liked_at timestamptz,
  performers jsonb,
  tags jsonb
) language sql security invoker as $$
  with base as (
    select uvd.video_id, uvd.created_at as liked_at
    from public.user_video_decisions as uvd
    where uvd.user_id = auth.uid()
      and uvd.decision_type = 'like'
  ), v as (
    select
      v.id,
      v.external_id,
      v.title,
      v.description,
      v.thumbnail_url,
      v.thumbnail_vertical_url,
      coalesce(v.affiliate_url, v.product_url) as product_url,
      v.price,
      v.product_released_at,
      b.liked_at
    from public.videos v
    join base b on b.video_id = v.id
    where (
      p_search is null
      or v.title ilike ('%' || p_search || '%')
    )
      and (p_price_min is null or v.price >= p_price_min)
      and (p_price_max is null or v.price <= p_price_max)
      and (p_release_gte is null or v.product_released_at >= p_release_gte)
      and (p_release_lte is null or v.product_released_at <= p_release_lte)
  ), vpt as (
    select
      v.id,
      jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name) as performers
    from v
    left join public.video_performers vp on vp.video_id = v.id
    left join public.performers p on p.id = vp.performer_id
    group by v.id
  ), vtg as (
    select
      v.id,
      jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name) as tags
    from v
    left join public.video_tags vt on vt.video_id = v.id
    left join public.tags t on t.id = vt.tag_id
    group by v.id
  ), vjoined as (
    select v.*, coalesce(vpt.performers, '[]'::jsonb) as performers, coalesce(vtg.tags, '[]'::jsonb) as tags
    from v
    left join vpt on vpt.id = v.id
    left join vtg on vtg.id = v.id
  ), vfiltered as (
    select * from vjoined
    where (
      p_tag_ids is null
      or exists (
        select 1 from jsonb_array_elements(tags) as t(obj)
        where (t.obj->>'id')::uuid = any(p_tag_ids)
      )
    ) and (
      p_performer_ids is null
      or exists (
        select 1 from jsonb_array_elements(performers) as p(obj)
        where (p.obj->>'id')::uuid = any(p_performer_ids)
      )
    )
  )
  select
    id,
    external_id,
    title,
    description,
    thumbnail_url,
    thumbnail_vertical_url,
    product_url,
    price,
    product_released_at,
    liked_at,
    performers,
    tags
  from vfiltered
  order by
    case when lower(p_sort) = 'liked_at' then liked_at end,
    case when lower(p_sort) = 'released' then product_released_at end,
    case when lower(p_sort) = 'price' then price end,
    case when lower(p_sort) = 'title' then title end,
    id
  limit greatest(p_limit, 1)
  offset greatest(p_offset, 0);
$$;

grant execute on function public.get_user_likes(text, text, text, int, int, uuid[], uuid[], numeric, numeric, timestamptz, timestamptz) to authenticated;

-- get_videos_recommendations / get_popular_videos: pass through affiliate URLs
drop function if exists public.get_videos_recommendations(uuid, int);
create or replace function public.get_videos_recommendations(user_uuid uuid, page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  thumbnail_vertical_url text,
  sample_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  score double precision,
  model_version text
)
language plpgsql
security definer
set search_path = public
as $$
declare
  user_vec vector(256);
  latest_version text;
begin
  select ue.embedding into user_vec
  from public.user_embeddings as ue
  where ue.user_id = user_uuid;

  if user_vec is null then
    return;
  end if;

  select ve.model_version
    into latest_version
  from public.video_embeddings as ve
  where ve.model_version is not null
  order by ve.updated_at desc
  limit 1;

  return query
  select
    v.id,
    v.title,
    v.description,
    v.external_id,
    v.thumbnail_url,
    v.thumbnail_vertical_url,
    v.sample_video_url,
    coalesce(v.affiliate_url, v.product_url) as product_url,
    v.product_released_at,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
        from public.video_performers vp
        join public.performers p on p.id = vp.performer_id
        where vp.video_id = v.id
      ), '[]'::jsonb
    ) as performers,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
        from public.video_tags vt
        join public.tags t on t.id = vt.tag_id
        left join public.tag_groups tg on tg.id = t.tag_group_id
        where vt.video_id = v.id
          and coalesce(tg.show_in_ui, true)
      ), '[]'::jsonb
    ) as tags,
    1 - (ve.embedding <-> user_vec) as score,
    ve.model_version
  from public.video_embeddings ve
  join public.videos v on v.id = ve.video_id
  where v.sample_video_url is not null
    and (latest_version is null or ve.model_version = latest_version)
    and not exists (
      select 1 from public.user_video_decisions uvd
      where uvd.video_id = v.id
        and uvd.user_id = user_uuid
    )
  order by ve.embedding <-> user_vec
  limit page_limit;
end;
$$;

grant execute on function public.get_videos_recommendations(uuid, int) to anon, authenticated;

drop function if exists public.get_popular_videos(uuid, int, int);
create or replace function public.get_popular_videos(user_uuid uuid default null, limit_count int default 20, lookback_days int default 7)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  thumbnail_vertical_url text,
  sample_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  score double precision
)
language plpgsql
security definer
set search_path = public
as $$
declare
  pop_view_exists boolean;
begin
  select exists (
    select 1
    from pg_matviews
    where schemaname = 'public'
      and matviewname = 'video_popularity_daily'
  ) into pop_view_exists;

  if not pop_view_exists then
    return;
  end if;

  begin
    return query
    with popularity as (
      select vpd.video_id,
             sum(vpd.likes)::double precision as total_likes
      from public.video_popularity_daily vpd
      where vpd.d >= (now() - make_interval(days => lookback_days))
      group by vpd.video_id
    )
    select
      v.id,
      v.title,
      v.description,
      v.external_id,
      v.thumbnail_url,
      v.thumbnail_vertical_url,
      v.sample_video_url,
      coalesce(v.affiliate_url, v.product_url) as product_url,
      v.product_released_at,
      coalesce(
        (
          select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
          from public.video_performers vp
          join public.performers p on p.id = vp.performer_id
          where vp.video_id = v.id
        ), '[]'::jsonb
      ) as performers,
      coalesce(
        (
          select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
          from public.video_tags vt
          join public.tags t on t.id = vt.tag_id
          left join public.tag_groups tg on tg.id = t.tag_group_id
          where vt.video_id = v.id
            and coalesce(tg.show_in_ui, true)
        ), '[]'::jsonb
      ) as tags,
      pop.total_likes as score
    from popularity pop
    join public.videos v on v.id = pop.video_id
    where v.sample_video_url is not null
      and (
        user_uuid is null
        or not exists (
          select 1
          from public.user_video_decisions uvd
          where uvd.user_id = user_uuid
            and uvd.video_id = v.id
        )
      )
    order by pop.total_likes desc nulls last, v.product_released_at desc
    limit limit_count;
  exception
    when object_not_in_prerequisite_state then
      return;
  end;
end;
$$;

grant execute on function public.get_popular_videos(uuid, int, int) to anon, authenticated;
