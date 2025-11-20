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
  tags jsonb,
  total_count bigint
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
    tags,
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
  limit greatest(p_limit, 1)
  offset greatest(p_offset, 0);
$$;

grant execute on function public.get_user_likes(text, text, text, int, int, uuid[], uuid[], numeric, numeric, timestamptz, timestamptz) to authenticated;
