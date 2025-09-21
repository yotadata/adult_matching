-- Facet RPCs for user's liked videos (tags and performers)

drop function if exists public.get_user_liked_tags();
create or replace function public.get_user_liked_tags()
returns table (
  id uuid,
  name text,
  cnt bigint
) language sql security invoker as $$
  select t.id, t.name, count(*)::bigint as cnt
  from public.user_video_decisions uvd
  join public.video_tags vt on vt.video_id = uvd.video_id
  join public.tags t on t.id = vt.tag_id
  where uvd.user_id = auth.uid()
    and uvd.decision_type = 'like'
  group by t.id, t.name
  order by cnt desc, t.name asc
$$;

drop function if exists public.get_user_liked_performers();
create or replace function public.get_user_liked_performers()
returns table (
  id uuid,
  name text,
  cnt bigint
) language sql security invoker as $$
  select p.id, p.name, count(*)::bigint as cnt
  from public.user_video_decisions uvd
  join public.video_performers vp on vp.video_id = uvd.video_id
  join public.performers p on p.id = vp.performer_id
  where uvd.user_id = auth.uid()
    and uvd.decision_type = 'like'
  group by p.id, p.name
  order by cnt desc, p.name asc
$$;

grant execute on function public.get_user_liked_tags() to authenticated;
grant execute on function public.get_user_liked_performers() to authenticated;

