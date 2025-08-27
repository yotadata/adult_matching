-- Redefine function with unambiguous column references
create or replace function public.get_videos_feed(page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
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
  with candidates as (
    select v.id as video_id
    from public.videos as v
    where v.sample_video_url is not null
      and not exists (
        select 1
        from public.user_video_decisions as uvd
        where uvd.video_id = v.id
          and (auth.uid() is not null and uvd.user_id = auth.uid())
      )
  )
  select count(*) into total_count from candidates;

  if total_count > page_limit then
    start_offset := floor(random() * (total_count - page_limit))::int;
  else
    start_offset := 0;
  end if;

  return query
  with candidates as (
    select v.id as video_id
    from public.videos as v
    where v.sample_video_url is not null
      and not exists (
        select 1
        from public.user_video_decisions as uvd
        where uvd.video_id = v.id
          and (auth.uid() is not null and uvd.user_id = auth.uid())
      )
  ), picked as (
    select c.video_id
    from candidates as c
    order by c.video_id
    offset start_offset limit page_limit
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
        from public.video_performers as vp
        join public.performers as p on p.id = vp.performer_id
        where vp.video_id = v.id
      ), '[]'::jsonb
    ) as performers,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
        from public.video_tags as vt
        join public.tags as t on t.id = vt.tag_id
        where vt.video_id = v.id
      ), '[]'::jsonb
    ) as tags
  from public.videos as v
  join picked as pk on pk.video_id = v.id
  order by v.id;
end;
$$;

grant execute on function public.get_videos_feed(int) to anon, authenticated;

