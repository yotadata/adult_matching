-- Two-Tower runtime support: filtering columns, ANN RPC, and HNSW index

-- additional video metadata for filtering
alter table public.videos
  add column if not exists safety_level smallint not null default 0,
  add column if not exists region_codes text[] not null default '{}'::text[],
  add column if not exists is_active boolean not null default true;

create index if not exists idx_videos_safety_level on public.videos (safety_level);
create index if not exists idx_videos_region_gin on public.videos using gin (region_codes);

-- ensure pgvector extension exists (idempotent)
create extension if not exists vector;

-- HNSW index for ANN search
create index if not exists idx_video_embeddings_embedding_hnsw
  on public.video_embeddings
  using hnsw (embedding vector_cosine_ops);

-- ANN + filter RPC
create or replace function public.recommend_videos_ann(
  p_embedding vector,
  p_limit int default 50,
  p_banned_tag_ids uuid[] default null,
  p_max_safety_level int default 1,
  p_allowed_regions text[] default null,
  p_exclude_video_ids uuid[] default null
)
returns table (
  video_id uuid,
  distance double precision,
  similarity double precision
) language plpgsql security definer as $$
declare
  v_limit int := greatest(1, least(500, coalesce(p_limit, 50)));
begin
  set search_path = public;
  if p_embedding is null then
    raise exception 'p_embedding must not be null';
  end if;

  return query
  with ann as (
    select
      ve.video_id,
      (ve.embedding <=> p_embedding) as distance
    from public.video_embeddings ve
    order by ve.embedding <=> p_embedding
    limit v_limit * 4
  ), filtered as (
    select
      ann.video_id,
      ann.distance,
      1 - ann.distance as similarity
    from ann
    join public.videos v on v.id = ann.video_id
    where (p_allowed_regions is null or array_length(p_allowed_regions, 1) = 0 or array_length(v.region_codes, 1) = 0 or v.region_codes && p_allowed_regions)
      and v.safety_level <= coalesce(p_max_safety_level, v.safety_level)
      and v.is_active
      and (p_exclude_video_ids is null or ann.video_id <> all(p_exclude_video_ids))
      and (
        p_banned_tag_ids is null or not exists (
          select 1 from public.video_tags vt
          where vt.video_id = ann.video_id
            and vt.tag_id = any(p_banned_tag_ids)
        )
      )
  )
  select
    video_id,
    distance,
    similarity
  from filtered
  order by distance
  limit v_limit;
end;
$$;

grant execute on function public.recommend_videos_ann(vector, int, uuid[], int, text[], uuid[]) to authenticated;
