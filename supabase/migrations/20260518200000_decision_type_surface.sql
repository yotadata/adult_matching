-- decision_type に surface を含める
-- like/nope → swipe_like/grid_like/swipe_nope

-- 1. 既存データを更新（全て swipe 由来とみなす）
update public.user_video_decisions set decision_type = 'swipe_like' where decision_type = 'like';
update public.user_video_decisions set decision_type = 'swipe_nope' where decision_type = 'nope';

-- 2. CHECK 制約を更新
alter table public.user_video_decisions
  drop constraint if exists user_video_decisions_decision_type_check;
alter table public.user_video_decisions
  add constraint user_video_decisions_decision_type_check
  check (decision_type in ('swipe_like', 'swipe_nope', 'grid_like'));

-- 3. matview 再作成（like → swipe_like OR grid_like）
drop materialized view if exists public.video_popularity_daily cascade;
create materialized view public.video_popularity_daily as
select
  uvd.video_id,
  date_trunc('day', uvd.created_at at time zone 'UTC') as d,
  count(*) as likes
from public.user_video_decisions uvd
where uvd.decision_type in ('swipe_like', 'grid_like')
group by uvd.video_id, date_trunc('day', uvd.created_at at time zone 'UTC');

create unique index mv_pop_daily_uidx on public.video_popularity_daily (video_id, d);

-- 4. get_videos_recommendations
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
    where uvd.user_id = user_uuid and uvd.decision_type in ('swipe_like', 'grid_like')
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

-- 5. get_user_likes
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
    where uvd.user_id = auth.uid() and uvd.decision_type in ('swipe_like', 'grid_like')
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

-- 6. get_user_liked_tags
create or replace function public.get_user_liked_tags()
returns table (id uuid, name text, cnt bigint, tag_group_name text)
language sql security invoker as $$
  select t.id, t.name, count(*)::bigint as cnt, tg.name as tag_group_name
  from public.user_video_decisions uvd
  join public.video_tags vt on vt.video_id = uvd.video_id
  join public.tags t on t.id = vt.tag_id
  left join public.tag_groups tg on tg.id = t.tag_group_id
  where uvd.user_id = auth.uid()
    and uvd.decision_type in ('swipe_like', 'grid_like')
    and coalesce(tg.show_in_ui, true)
  group by t.id, t.name, tg.name
  order by cnt desc, t.name asc;
$$;

grant execute on function public.get_user_liked_tags() to authenticated;

-- 7. get_user_liked_performers
create or replace function public.get_user_liked_performers()
returns table (id uuid, name text, cnt bigint)
language sql security invoker as $$
  select p.id, p.name, count(*)::bigint as cnt
  from public.user_video_decisions uvd
  join public.video_performers vp on vp.video_id = uvd.video_id
  join public.performers p on p.id = vp.performer_id
  where uvd.user_id = auth.uid()
    and uvd.decision_type in ('swipe_like', 'grid_like')
  group by p.id, p.name
  order by cnt desc, p.name asc;
$$;

grant execute on function public.get_user_liked_performers() to authenticated;

-- 8. update_user_features
create or replace function public.update_user_features(p_user_id uuid)
returns void
language plpgsql security definer set search_path = public
as $$
begin
  insert into public.user_features (user_id, top_tags, top_performers, updated_at)
  select
    p_user_id,
    coalesce((
      select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name, 'cnt', cnt) order by cnt desc)
      from (
        select vt.tag_id, count(*) as cnt
        from public.user_video_decisions uvd
        join public.video_tags vt on vt.video_id = uvd.video_id
        where uvd.user_id = p_user_id and uvd.decision_type in ('swipe_like', 'grid_like')
        group by vt.tag_id order by cnt desc limit 20
      ) top join public.tags t on t.id = top.tag_id
    ), '[]'::jsonb),
    coalesce((
      select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name, 'cnt', cnt) order by cnt desc)
      from (
        select vp.performer_id, count(*) as cnt
        from public.user_video_decisions uvd
        join public.video_performers vp on vp.video_id = uvd.video_id
        where uvd.user_id = p_user_id and uvd.decision_type in ('swipe_like', 'grid_like')
        group by vp.performer_id order by cnt desc limit 20
      ) top join public.performers p on p.id = top.performer_id
    ), '[]'::jsonb),
    now()
  on conflict (user_id) do update
    set top_tags = excluded.top_tags,
        top_performers = excluded.top_performers,
        updated_at = excluded.updated_at;
end;
$$;

grant execute on function public.update_user_features(uuid) to authenticated, service_role;

-- 9. get_user_likes_total_count（既存関数を上書き）
create or replace function public.get_user_likes_total_count()
returns bigint
language sql security invoker as $$
  select count(*)::bigint
  from public.user_video_decisions
  where user_id = auth.uid()
    and decision_type in ('swipe_like', 'grid_like');
$$;

grant execute on function public.get_user_likes_total_count() to authenticated;
