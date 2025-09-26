-- =============================================================================
-- Personalized Feed RPC Functions for Adult Matching Backend
--
-- パーソナライゼドフィード用RPC関数群
-- ユーザーの嗜好に基づいた動画推薦機能を提供
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Function 1: get_personalized_videos_feed
-- ユーザーの過去の行動とベクター埋め込みに基づくパーソナライゼドフィード
-- -----------------------------------------------------------------------------

create or replace function public.get_personalized_videos_feed(
  user_id_param uuid default null,
  page_limit int default 20,
  diversity_factor float default 0.3
)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
  preview_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  recommendation_score float,
  recommendation_reason text
)
language plpgsql
security definer
set search_path = public
as $$
declare
  target_user_id uuid;
  user_embedding vector(768);
  has_user_history boolean := false;
  fallback_to_popular boolean := false;
begin
  -- ユーザーID決定
  target_user_id := coalesce(user_id_param, auth.uid());

  if target_user_id is null then
    -- 匿名ユーザーの場合は人気動画を返す
    return query select * from public.get_popular_videos_feed(page_limit);
    return;
  end if;

  -- ユーザーの埋め込みベクトル取得
  select embedding into user_embedding
  from public.user_embeddings
  where user_id = target_user_id
  order by created_at desc
  limit 1;

  -- ユーザーの行動履歴確認
  select exists(
    select 1 from public.user_video_decisions
    where user_id = target_user_id
    limit 1
  ) into has_user_history;

  -- 埋め込みベクトルがない、または行動履歴が少ない場合
  if user_embedding is null or not has_user_history then
    -- 新規ユーザー向けの多様性重視フィードにフォールバック
    return query select * from public.get_explore_videos_feed(page_limit);
    return;
  end if;

  -- パーソナライゼド推薦実行
  return query
  with user_preferences as (
    -- ユーザーがLIKEした動画のジャンル・出演者分析
    select
      jsonb_array_elements_text(v.tags->'genres') as genre,
      count(*) as genre_score
    from public.user_video_decisions uvd
    join public.videos v on v.id = uvd.video_id
    where uvd.user_id = target_user_id
      and uvd.decision = 'like'
      and v.tags->'genres' is not null
    group by jsonb_array_elements_text(v.tags->'genres')
  ),
  performer_preferences as (
    -- ユーザーがLIKEした出演者分析
    select
      (jsonb_array_elements(v.performers)->>'name') as performer_name,
      count(*) as performer_score
    from public.user_video_decisions uvd
    join public.videos v on v.id = uvd.video_id
    where uvd.user_id = target_user_id
      and uvd.decision = 'like'
      and v.performers is not null
    group by (jsonb_array_elements(v.performers)->>'name')
  ),
  candidate_videos as (
    select
      v.*,
      ve.embedding as video_embedding,
      -- ベクター類似度計算
      case
        when ve.embedding is not null then
          1 - (user_embedding <=> ve.embedding)
        else 0.5
      end as vector_similarity,
      -- ジャンル類似度
      coalesce(
        (select sum(up.genre_score)
         from user_preferences up
         where up.genre = any(
           select jsonb_array_elements_text(v.tags->'genres')
         )
        )::float / greatest(
          (select sum(genre_score) from user_preferences), 1
        ), 0
      ) as genre_similarity,
      -- 出演者類似度
      coalesce(
        (select sum(pp.performer_score)
         from performer_preferences pp
         where pp.performer_name = any(
           select jsonb_array_elements(v.performers)->>'name'
         )
        )::float / greatest(
          (select sum(performer_score) from performer_preferences), 1
        ), 0
      ) as performer_similarity,
      -- 新しさスコア（最近の動画を優遇）
      case
        when v.product_released_at is not null then
          greatest(0, 1 - extract(days from (now() - v.product_released_at))::float / 365)
        else 0.3
      end as recency_score,
      -- 人気度スコア
      coalesce(
        (v.tags->>'popularity_score')::float / 100.0, 0.1
      ) as popularity_score
    from public.videos v
    left join public.video_embeddings ve on ve.video_id = v.id
    where not exists (
      select 1 from public.user_video_decisions uvd
      where uvd.video_id = v.id and uvd.user_id = target_user_id
    )
  ),
  scored_recommendations as (
    select
      cv.*,
      -- 総合スコア計算（重み付き平均）
      (
        cv.vector_similarity * 0.4 +
        cv.genre_similarity * 0.25 +
        cv.performer_similarity * 0.15 +
        cv.recency_score * 0.1 +
        cv.popularity_score * 0.1
      ) * (1 + random() * diversity_factor) as final_score,
      -- 推薦理由生成
      case
        when cv.vector_similarity > 0.8 then 'あなたの嗜好と高い類似性'
        when cv.genre_similarity > 0.7 then 'お気に入りジャンルと一致'
        when cv.performer_similarity > 0.6 then 'お気に入り出演者'
        when cv.recency_score > 0.8 then '最新作品'
        when cv.popularity_score > 0.8 then '人気作品'
        else '新しい発見'
      end as reason
    from candidate_videos cv
  )
  select
    sr.id,
    sr.title,
    sr.description,
    sr.external_id,
    sr.thumbnail_url,
    sr.sample_video_url,
    sr.preview_video_url,
    sr.product_url,
    sr.product_released_at,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
        from public.video_performers vp
        join public.performers p on p.id = vp.performer_id
        where vp.video_id = sr.id
      ), '[]'::jsonb
    ) as performers,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
        from public.video_tags vt
        join public.tags t on t.id = vt.tag_id
        where vt.video_id = sr.id
      ), '[]'::jsonb
    ) as tags,
    sr.final_score as recommendation_score,
    sr.reason as recommendation_reason
  from scored_recommendations sr
  order by sr.final_score desc
  limit page_limit;

end;
$$;

grant execute on function public.get_personalized_videos_feed(uuid, int, float) to anon, authenticated;

-- -----------------------------------------------------------------------------
-- Function 2: get_popular_videos_feed
-- 人気動画フィード（パーソナライゼーション情報がない場合のフォールバック）
-- -----------------------------------------------------------------------------

create or replace function public.get_popular_videos_feed(page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
  preview_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  recommendation_score float,
  recommendation_reason text
)
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  with popular_videos as (
    select
      v.*,
      -- 人気度スコア計算
      (
        coalesce((v.tags->>'popularity_score')::float, 0) * 0.4 +
        coalesce((v.tags->>'rating')::float * 20, 0) * 0.3 +
        coalesce(
          (select count(*)::float from public.user_video_decisions uvd
           where uvd.video_id = v.id and uvd.decision = 'like'), 0
        ) * 2.0 * 0.3
      ) / 100.0 as popularity_score,
      -- 新しさボーナス
      case
        when v.product_released_at > now() - interval '30 days' then 0.2
        when v.product_released_at > now() - interval '90 days' then 0.1
        else 0
      end as recency_bonus
    from public.videos v
    where auth.uid() is null or not exists (
      select 1 from public.user_video_decisions uvd
      where uvd.video_id = v.id and uvd.user_id = auth.uid()
    )
  )
  select
    pv.id,
    pv.title,
    pv.description,
    pv.external_id,
    pv.thumbnail_url,
    pv.sample_video_url,
    pv.preview_video_url,
    pv.product_url,
    pv.product_released_at,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
        from public.video_performers vp
        join public.performers p on p.id = vp.performer_id
        where vp.video_id = pv.id
      ), '[]'::jsonb
    ) as performers,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
        from public.video_tags vt
        join public.tags t on t.id = vt.tag_id
        where vt.video_id = pv.id
      ), '[]'::jsonb
    ) as tags,
    (pv.popularity_score + pv.recency_bonus) as recommendation_score,
    '人気作品' as recommendation_reason
  from popular_videos pv
  where pv.popularity_score > 0.1
  order by (pv.popularity_score + pv.recency_bonus) desc, random()
  limit page_limit;

end;
$$;

grant execute on function public.get_popular_videos_feed(int) to anon, authenticated;

-- -----------------------------------------------------------------------------
-- Function 3: get_explore_videos_feed
-- 多様性重視のエクスプローラーフィード（新規ユーザー向け）
-- -----------------------------------------------------------------------------

create or replace function public.get_explore_videos_feed(page_limit int default 20)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
  preview_video_url text,
  product_url text,
  product_released_at timestamptz,
  performers jsonb,
  tags jsonb,
  recommendation_score float,
  recommendation_reason text
)
language plpgsql
security definer
set search_path = public
as $$
declare
  target_user_id uuid;
begin
  target_user_id := auth.uid();

  return query
  with diverse_selection as (
    select
      v.*,
      row_number() over (
        partition by
          coalesce(jsonb_array_elements_text(v.tags->'genres'), 'unknown')
        order by random()
      ) as genre_rank,
      coalesce((v.tags->>'popularity_score')::float, 0) as popularity,
      case
        when v.sample_video_url is not null then 0.3
        else 0.0
      end as sample_bonus,
      -- ジャンル多様性ボーナス
      1.0 / greatest(
        (select count(*) from public.videos v2
         where v2.tags->'genres' ?| array[
           select jsonb_array_elements_text(v.tags->'genres')
         ]
        ), 1
      ) as diversity_bonus
    from public.videos v
    where target_user_id is null or not exists (
      select 1 from public.user_video_decisions uvd
      where uvd.video_id = v.id and uvd.user_id = target_user_id
    )
  )
  select
    ds.id,
    ds.title,
    ds.description,
    ds.external_id,
    ds.thumbnail_url,
    ds.sample_video_url,
    ds.preview_video_url,
    ds.product_url,
    ds.product_released_at,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) order by p.name)
        from public.video_performers vp
        join public.performers p on p.id = vp.performer_id
        where vp.video_id = ds.id
      ), '[]'::jsonb
    ) as performers,
    coalesce(
      (
        select jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) order by t.name)
        from public.video_tags vt
        join public.tags t on t.id = vt.tag_id
        where vt.video_id = ds.id
      ), '[]'::jsonb
    ) as tags,
    (
      ds.popularity / 100.0 * 0.4 +
      ds.sample_bonus * 0.3 +
      ds.diversity_bonus * 0.3
    ) as recommendation_score,
    '新しい発見' as recommendation_reason
  from diverse_selection ds
  where ds.genre_rank <= 3  -- 各ジャンルから最大3作品
  order by random()
  limit page_limit;

end;
$$;

grant execute on function public.get_explore_videos_feed(int) to anon, authenticated;

-- -----------------------------------------------------------------------------
-- Function 4: get_similar_videos
-- 特定動画に類似した動画の取得
-- -----------------------------------------------------------------------------

create or replace function public.get_similar_videos(
  video_id_param uuid,
  page_limit int default 10
)
returns table (
  id uuid,
  title text,
  description text,
  external_id text,
  thumbnail_url text,
  sample_video_url text,
  similarity_score float,
  similarity_reason text
)
language plpgsql
security definer
set search_path = public
as $$
declare
  target_video record;
  target_embedding vector(768);
  target_user_id uuid;
begin
  target_user_id := auth.uid();

  -- 基準動画の情報取得
  select * into target_video
  from public.videos
  where id = video_id_param;

  if not found then
    raise exception 'Video not found: %', video_id_param;
  end if;

  -- 基準動画の埋め込みベクトル取得
  select embedding into target_embedding
  from public.video_embeddings
  where video_id = video_id_param;

  return query
  with similar_candidates as (
    select
      v.*,
      ve.embedding as video_embedding,
      -- ベクター類似度
      case
        when target_embedding is not null and ve.embedding is not null then
          1 - (target_embedding <=> ve.embedding)
        else 0.5
      end as vector_similarity,
      -- ジャンル類似度
      coalesce(
        (
          select count(*)::float / greatest(
            jsonb_array_length(target_video.tags->'genres') +
            jsonb_array_length(v.tags->'genres'), 1
          )
          from (
            select jsonb_array_elements_text(target_video.tags->'genres')
            intersect
            select jsonb_array_elements_text(v.tags->'genres')
          ) common_genres
        ), 0
      ) as genre_similarity,
      -- 出演者類似度
      coalesce(
        (
          select count(*)::float / greatest(
            jsonb_array_length(target_video.performers) +
            jsonb_array_length(v.performers), 1
          )
          from (
            select jsonb_array_elements(target_video.performers)->>'name'
            intersect
            select jsonb_array_elements(v.performers)->>'name'
          ) common_performers
        ), 0
      ) as performer_similarity
    from public.videos v
    left join public.video_embeddings ve on ve.video_id = v.id
    where v.id != video_id_param
      and (target_user_id is null or not exists (
        select 1 from public.user_video_decisions uvd
        where uvd.video_id = v.id and uvd.user_id = target_user_id
      ))
  ),
  scored_similar as (
    select
      sc.*,
      (
        sc.vector_similarity * 0.5 +
        sc.genre_similarity * 0.3 +
        sc.performer_similarity * 0.2
      ) as final_similarity,
      case
        when sc.performer_similarity > 0.5 then '同じ出演者'
        when sc.genre_similarity > 0.7 then '同じジャンル'
        when sc.vector_similarity > 0.8 then '内容が類似'
        else '関連作品'
      end as similarity_reason
    from similar_candidates sc
    where (
      sc.vector_similarity > 0.3 or
      sc.genre_similarity > 0.3 or
      sc.performer_similarity > 0.3
    )
  )
  select
    ss.id,
    ss.title,
    ss.description,
    ss.external_id,
    ss.thumbnail_url,
    ss.sample_video_url,
    ss.final_similarity as similarity_score,
    ss.similarity_reason
  from scored_similar ss
  order by ss.final_similarity desc
  limit page_limit;

end;
$$;

grant execute on function public.get_similar_videos(uuid, int) to anon, authenticated;

-- -----------------------------------------------------------------------------
-- Function 5: update_user_recommendation_feedback
-- ユーザーの推薦フィードバックを記録してモデル改善に活用
-- -----------------------------------------------------------------------------

create or replace function public.update_user_recommendation_feedback(
  video_id_param uuid,
  feedback_type text, -- 'click', 'view', 'like', 'skip', 'block'
  feedback_context jsonb default null
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  target_user_id uuid;
begin
  target_user_id := auth.uid();

  if target_user_id is null then
    return false;
  end if;

  -- フィードバックを記録
  insert into public.recommendation_feedback (
    user_id,
    video_id,
    feedback_type,
    feedback_context,
    created_at
  ) values (
    target_user_id,
    video_id_param,
    feedback_type,
    feedback_context,
    now()
  )
  on conflict (user_id, video_id, feedback_type)
  do update set
    feedback_context = coalesce(excluded.feedback_context, recommendation_feedback.feedback_context),
    updated_at = now();

  return true;
end;
$$;

grant execute on function public.update_user_recommendation_feedback(uuid, text, jsonb) to authenticated;

-- -----------------------------------------------------------------------------
-- Table: recommendation_feedback
-- 推薦フィードバックテーブル（存在しない場合のみ作成）
-- -----------------------------------------------------------------------------

create table if not exists public.recommendation_feedback (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  video_id uuid not null references public.videos(id) on delete cascade,
  feedback_type text not null check (feedback_type in ('click', 'view', 'like', 'skip', 'block')),
  feedback_context jsonb default null,
  created_at timestamptz default now() not null,
  updated_at timestamptz default now() not null,

  unique(user_id, video_id, feedback_type)
);

-- インデックス作成
create index if not exists idx_recommendation_feedback_user_id on public.recommendation_feedback(user_id);
create index if not exists idx_recommendation_feedback_video_id on public.recommendation_feedback(video_id);
create index if not exists idx_recommendation_feedback_created_at on public.recommendation_feedback(created_at);
create index if not exists idx_recommendation_feedback_type on public.recommendation_feedback(feedback_type);

-- RLS ポリシー
alter table public.recommendation_feedback enable row level security;

create policy if not exists "Users can manage their own feedback"
on public.recommendation_feedback
for all
using (auth.uid() = user_id);

-- 権限付与
grant select, insert, update, delete on public.recommendation_feedback to authenticated;
grant usage on sequence public.recommendation_feedback_id_seq to authenticated;

-- -----------------------------------------------------------------------------
-- Comments for documentation
-- -----------------------------------------------------------------------------

comment on function public.get_personalized_videos_feed is 'ユーザーの嗜好に基づくパーソナライゼドビデオフィード。ベクター埋め込み、ジャンル・出演者履歴、多様性を考慮';
comment on function public.get_popular_videos_feed is '人気動画フィード。新規ユーザーや匿名ユーザー向けフォールバック';
comment on function public.get_explore_videos_feed is '多様性重視のエクスプローラーフィード。新規ユーザーの発見促進';
comment on function public.get_similar_videos is '指定動画に類似した動画推薦。ベクター類似度、ジャンル、出演者の類似性を分析';
comment on function public.update_user_recommendation_feedback is '推薦システムのフィードバック記録。機械学習モデル改善に活用';
comment on table public.recommendation_feedback is '推薦フィードバックデータ。ユーザーの推薦結果に対するアクション履歴';