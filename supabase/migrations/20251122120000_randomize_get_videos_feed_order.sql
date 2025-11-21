-- Randomize get_videos_feed ordering to avoid deterministic exploration for guests
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
begin
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
  where not exists (
    select 1 from public.user_video_decisions uvd
    where uvd.video_id = v.id
      and (auth.uid() is not null and uvd.user_id = auth.uid())
  )
  order by (v.sample_video_url is null), random()
  limit page_limit;
end;
$$;

grant execute on function public.get_videos_feed(int) to anon, authenticated;
