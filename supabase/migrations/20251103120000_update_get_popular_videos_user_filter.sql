drop function if exists public.get_popular_videos(int, int);

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
          where vt.video_id = v.id
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
