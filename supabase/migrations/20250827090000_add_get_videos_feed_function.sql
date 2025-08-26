-- Create function to fetch videos excluding already decided ones for current user
create or replace function public.get_videos_feed(page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
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
  -- Candidate set: has sample url and not judged by current user
  with filtered as (
    select v.*
    from public.videos v
    where v.sample_video_url is not null
      and not exists (
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
    ) as tags
  from public.videos v
  where v.id in (
    select id from (
      select id from public.videos v2
      where v2.sample_video_url is not null
        and not exists (
          select 1 from public.user_video_decisions uvd2
          where uvd2.video_id = v2.id
            and (auth.uid() is not null and uvd2.user_id = auth.uid())
        )
      order by v2.id
      offset start_offset limit page_limit
    ) s
  )
  order by v.id;
end;
$$;

grant execute on function public.get_videos_feed(int) to anon, authenticated;

