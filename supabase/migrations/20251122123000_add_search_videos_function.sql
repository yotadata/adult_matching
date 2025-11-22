-- search_videos: prompt / tag_ids / performer_ids で動画を検索する RPC
create or replace function public.search_videos(
  p_prompt text default null,
  p_tag_ids uuid[] default null,
  p_performer_ids uuid[] default null,
  p_limit int default 30
)
returns table (
  id uuid,
  external_id text,
  title text,
  description text,
  thumbnail_url text,
  sample_video_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb
)
language sql
security invoker
as $$
  with base as (
    select
      v.id,
      v.external_id,
      v.title,
      v.description,
      v.thumbnail_url,
      v.sample_video_url,
      v.product_released_at,
      coalesce(
        (
          select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
          from public.video_performers vp
          join public.performers p on p.id = vp.performer_id
          where vp.video_id = v.id
        ),
        '[]'::jsonb
      ) as performers,
      coalesce(
        (
          select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
          from public.video_tags vt
          join public.tags t on t.id = vt.tag_id
          left join public.tag_groups tg on tg.id = t.tag_group_id
          where vt.video_id = v.id
            and coalesce(tg.show_in_ui, true)
        ),
        '[]'::jsonb
      ) as tags
    from public.videos v
  ),
  filtered as (
    select *
    from base b
    where
      (
        p_tag_ids is null
        or exists (
          select 1 from jsonb_array_elements(b.tags) t(obj)
          where (t.obj->>'id')::uuid = any(p_tag_ids)
        )
      ) and (
        p_performer_ids is null
        or exists (
          select 1 from jsonb_array_elements(b.performers) p(obj)
          where (p.obj->>'id')::uuid = any(p_performer_ids)
        )
      ) and (
        p_prompt is null
        or p_prompt = ''
        or (
          b.title ilike ('%' || p_prompt || '%')
          or exists (
            select 1 from jsonb_array_elements(b.tags) t(obj)
            where t.obj->>'name' ilike ('%' || p_prompt || '%')
          )
          or exists (
            select 1 from jsonb_array_elements(b.performers) p(obj)
            where p.obj->>'name' ilike ('%' || p_prompt || '%')
          )
        )
      )
  )
  select *
  from filtered
  order by coalesce(product_released_at, now()) desc nulls last, id
  limit greatest(p_limit, 1);
$$;

grant execute on function public.search_videos(text, uuid[], uuid[], int) to anon, authenticated;
