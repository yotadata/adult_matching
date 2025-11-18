drop function if exists public.get_user_liked_tags();
create or replace function public.get_user_liked_tags()
returns table (
  id uuid,
  name text,
  cnt bigint,
  tag_group_name text
) language sql security invoker as $$
  select t.id, t.name, count(*)::bigint as cnt, tg.name as tag_group_name
  from public.user_video_decisions uvd
  join public.video_tags vt on vt.video_id = uvd.video_id
  join public.tags t on t.id = vt.tag_id
  left join public.tag_groups tg on tg.id = t.tag_group_id
  where uvd.user_id = auth.uid()
    and uvd.decision_type = 'like'
    and coalesce(tg.show_in_ui, true)
  group by t.id, t.name, tg.name
  order by cnt desc, t.name asc
$$;

grant execute on function public.get_user_liked_tags() to authenticated;
