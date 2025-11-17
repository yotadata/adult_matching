-- Recreate functions to respect tag_groups.show_in_ui
drop function if exists public.get_videos_recommendations(uuid, int);

create or replace function public.get_videos_recommendations(user_uuid uuid, page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
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
    v.sample_video_url,
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
  sample_video_url text,
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
      v.sample_video_url,
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
