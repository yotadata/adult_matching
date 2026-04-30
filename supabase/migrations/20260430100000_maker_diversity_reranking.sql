-- get_videos_recommendations に maker 多様性 reranking を追加
--
-- 問題: 特定 maker（エスワン, ムーディーズ等）が推薦の 10-15% を占める偏りがある。
-- 対策: 候補を page_limit * 5 件取得し、同一 maker の k 番目の出現に
--       penalty = 1 / (1 + (k-1) * 0.3) を掛けて rerank する。
--       これにより maker シェアが分散され、多様性が向上する。

create or replace function public.get_videos_recommendations(user_uuid uuid, page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  thumbnail_vertical_url text,
  sample_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  score double precision,
  model_version text
)
language plpgsql
security definer
set search_path = public
as $$
declare
  user_vec halfvec(128);
  approx_vec halfvec(128);
  latest_version text;
  candidate_limit int;
begin
  candidate_limit := page_limit * 5;

  -- 近似ユーザー埋め込み: いいね済み動画の埋め込み平均（直近200件）
  select avg(ve.embedding)::halfvec(128)
    into approx_vec
  from (
    select uvd.video_id
    from public.user_video_decisions uvd
    where uvd.user_id = user_uuid
      and uvd.decision_type = 'like'
    order by uvd.created_at desc
    limit 200
  ) recent_likes
  join public.video_embeddings ve on ve.video_id = recent_likes.video_id;

  -- いいねがゼロの場合はバッチ生成済み埋め込みにフォールバック
  if approx_vec is null then
    select ue.embedding into user_vec
    from public.user_embeddings ue
    where ue.user_id = user_uuid;
  end if;

  user_vec := coalesce(approx_vec, user_vec);

  if user_vec is null then
    return;
  end if;

  select ve.model_version
    into latest_version
  from public.video_embeddings ve
  where ve.model_version is not null
  order by ve.updated_at desc
  limit 1;

  return query
  with candidates as (
    select
      v.id,
      v.title,
      v.description,
      v.external_id,
      v.thumbnail_url,
      v.thumbnail_vertical_url,
      v.sample_video_url,
      coalesce(v.affiliate_url, v.product_url) as product_url,
      v.product_released_at,
      coalesce(v.maker, 'unknown') as maker,
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
      ) as tags,
      1 - (ve.embedding <-> user_vec) as raw_score,
      ve.model_version
    from public.video_embeddings ve
    join public.videos v on v.id = ve.video_id
    where v.sample_video_url is not null
      and (latest_version is null or ve.model_version = latest_version)
      and not exists (
        select 1 from public.user_video_decisions uvd
        where uvd.video_id = v.id
          and uvd.user_id = user_uuid
      )
    order by ve.embedding <-> user_vec
    limit candidate_limit
  ),
  ranked as (
    select
      *,
      -- 同一 maker 内での出現順（1始まり）
      row_number() over (partition by maker order by raw_score desc) as maker_rank,
      -- penalty = 1 / (1 + (maker_rank - 1) * 0.3)
      raw_score / (1.0 + (row_number() over (partition by maker order by raw_score desc) - 1) * 0.3) as diversified_score
    from candidates
  )
  select
    r.id,
    r.title,
    r.description,
    r.external_id,
    r.thumbnail_url,
    r.thumbnail_vertical_url,
    r.sample_video_url,
    r.product_url,
    r.product_released_at,
    r.performers,
    r.tags,
    r.diversified_score as score,
    r.model_version
  from ranked r
  order by r.diversified_score desc
  limit page_limit;
end;
$$;

grant execute on function public.get_videos_recommendations(uuid, int) to anon, authenticated;
