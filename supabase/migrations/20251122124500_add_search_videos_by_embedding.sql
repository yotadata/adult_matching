-- search_videos_by_embedding: タグ/出演者に紐づく動画埋め込みの平均ベクトルで近傍検索
create or replace function public.search_videos_by_embedding(
  p_tag_ids uuid[] default null,
  p_performer_ids uuid[] default null,
  p_limit int default 50
)
returns table (
  id uuid,
  score double precision,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb
)
language sql
security definer
set search_path = public
as $$
  with candidate_videos as (
    select distinct v.id
    from public.videos v
    left join public.video_tags vt on vt.video_id = v.id
    left join public.video_performers vp on vp.video_id = v.id
    where (
      p_tag_ids is null
      or vt.tag_id = any(p_tag_ids)
    )
    and (
      p_performer_ids is null
      or vp.performer_id = any(p_performer_ids)
    )
    limit 200
  ),
  avg_vec as (
    select
      avg(ve.embedding) as embedding
    from public.video_embeddings ve
    join candidate_videos cv on cv.id = ve.video_id
  ),
  nearest as (
    select
      ve.video_id as id,
      1 - (ve.embedding <=> av.embedding) as score
    from public.video_embeddings ve
    cross join avg_vec av
    where av.embedding is not null
    order by ve.embedding <=> av.embedding
    limit greatest(p_limit, 1)
  )
  select
    n.id,
    n.score,
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
    ) as tags
  from nearest n
  join public.videos v on v.id = n.id
  order by n.score desc nulls last;
$$;

grant execute on function public.search_videos_by_embedding(uuid[], uuid[], int) to anon, authenticated;
