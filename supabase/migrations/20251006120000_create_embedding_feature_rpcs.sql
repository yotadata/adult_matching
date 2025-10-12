-- Two-Tower 推論用 RPC: ユーザー／アイテム特徴量を返却

create or replace function public.get_user_embedding_features(user_id uuid)
returns jsonb
language plpgsql
security definer
set search_path = public
stable
as $$
declare
  v_profile timestamptz;
  v_like_count integer := 0;
  v_nope_count integer := 0;
  v_decision_count integer := 0;
  v_recent_like timestamptz;
  v_avg_like_hour numeric;
  v_price_mean numeric;
  v_price_median numeric;
  v_tags text[] := array[]::text[];
  v_performers text[] := array[]::text[];
  v_response jsonb;
  v_requester uuid := auth.uid();
begin
  if v_requester is not null and v_requester <> user_id then
    raise exception 'access to user % is not permitted for requester %', user_id, v_requester;
  end if;

  select p.created_at into v_profile
  from public.profiles p
  where p.user_id = user_id;

  select
    count(*) filter (where d.decision_type = 'like'),
    count(*) filter (where d.decision_type = 'nope'),
    count(*) as total_count,
    max(d.created_at) filter (where d.decision_type = 'like'),
    avg(extract(hour from d.created_at)::numeric) filter (where d.decision_type = 'like')
  into v_like_count, v_nope_count, v_decision_count, v_recent_like, v_avg_like_hour
  from public.user_video_decisions d
  where d.user_id = user_id;

  select avg(v.price)::numeric
  into v_price_mean
  from public.user_video_decisions d
  join public.videos v on v.id = d.video_id
  where d.user_id = user_id
    and d.decision_type = 'like'
    and v.price is not null;

  select percentile_cont(0.5) within group (order by v.price)
  into v_price_median
  from public.user_video_decisions d
  join public.videos v on v.id = d.video_id
  where d.user_id = user_id
    and d.decision_type = 'like'
    and v.price is not null;

  select coalesce(array_agg(t.name), array[]::text[])
  into v_tags
  from public.user_video_decisions d
  join public.video_tags vt on vt.video_id = d.video_id
  join public.tags t on t.id = vt.tag_id
  where d.user_id = user_id
    and d.decision_type = 'like';

  select coalesce(array_agg(pf.name), array[]::text[])
  into v_performers
  from public.user_video_decisions d
  join public.video_performers vp on vp.video_id = d.video_id
  join public.performers pf on pf.id = vp.performer_id
  where d.user_id = user_id
    and d.decision_type = 'like';

  v_response := jsonb_build_object(
    'user_id', user_id,
    'profile', jsonb_build_object('created_at', v_profile),
    'decision_stats', jsonb_build_object(
      'like_count', coalesce(v_like_count, 0),
      'nope_count', coalesce(v_nope_count, 0),
      'decision_count', coalesce(v_decision_count, 0),
      'recent_like_at', v_recent_like,
      'average_like_hour', coalesce(v_avg_like_hour, 12.0)
    ),
    'price_stats', jsonb_build_object(
      'mean', coalesce(v_price_mean, 0),
      'median', coalesce(v_price_median, 0)
    ),
    'tags', to_jsonb(v_tags),
    'performers', to_jsonb(v_performers)
  );

  return v_response;
end;
$$;

grant execute on function public.get_user_embedding_features(uuid) to authenticated, service_role;

create or replace function public.get_item_embedding_features(video_id uuid)
returns jsonb
language plpgsql
security definer
set search_path = public
stable
as $$
declare
  v_record record;
  v_response jsonb;
begin
  select v.*, 
    coalesce(tag_info.tags, array[]::text[]) as tag_names,
    coalesce(perf_info.performers, array[]::text[]) as performer_names
  into v_record
  from public.videos v
  left join lateral (
    select array_agg(t.name order by t.name) as tags
    from public.video_tags vt
    join public.tags t on t.id = vt.tag_id
    where vt.video_id = v.id
  ) tag_info on true
  left join lateral (
    select array_agg(pf.name order by pf.name) as performers
    from public.video_performers vp
    join public.performers pf on pf.id = vp.performer_id
    where vp.video_id = v.id
  ) perf_info on true
  where v.id = video_id;

  if not found then
    raise exception 'video % not found', video_id;
  end if;

  v_response := jsonb_build_object(
    'video_id', v_record.id,
    'price', v_record.price,
    'duration_seconds', v_record.duration_seconds,
    'product_released_at', v_record.product_released_at,
    'distribution_started_at', v_record.distribution_started_at,
    'safety_level', v_record.safety_level,
    'region_codes', to_jsonb(coalesce(v_record.region_codes, array[]::text[])),
    'tags', to_jsonb(v_record.tag_names),
    'performers', to_jsonb(v_record.performer_names)
  );

  return v_response;
end;
$$;

grant execute on function public.get_item_embedding_features(uuid) to authenticated, service_role;

create or replace function public.get_item_embedding_feature_batch(p_limit int default 100, p_offset int default 0)
returns table (video_id uuid, payload jsonb)
language sql
security definer
set search_path = public
stable
as $$
  with limited as (
    select v.id
    from public.videos v
    where v.is_active
    order by v.id
    limit greatest(1, least(coalesce(p_limit, 100), 1000))
    offset greatest(0, coalesce(p_offset, 0))
  )
  select l.id as video_id,
         public.get_item_embedding_features(l.id) as payload
  from limited l;
$$;

grant execute on function public.get_item_embedding_feature_batch(int, int) to service_role;

create or replace function public.get_user_embedding_feature_batch(p_limit int default 100, p_offset int default 0)
returns table (user_id uuid, payload jsonb)
language sql
security definer
set search_path = public
stable
as $$
  with limited as (
    select distinct d.user_id
    from public.user_video_decisions d
    order by d.user_id
    limit greatest(1, least(coalesce(p_limit, 100), 1000))
    offset greatest(0, coalesce(p_offset, 0))
  )
  select l.user_id,
         public.get_user_embedding_features(l.user_id) as payload
  from limited l;
$$;

grant execute on function public.get_user_embedding_feature_batch(int, int) to service_role;
