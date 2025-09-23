-- =============================================================================
-- Missing Database Functions and Procedures Implementation
--
-- 欠落データベース関数・プロシージャの実装
-- Edge Functions統合のための必要なRPC関数を完全実装
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Function 1: get_random_videos
-- ランダム動画取得関数（Edge Functions/content/feed/index.tsで使用）
-- -----------------------------------------------------------------------------

create or replace function public.get_random_videos(
  page_limit int default 20,
  exclude_video_ids uuid[] default array[]::uuid[]
)
returns table (
  id uuid,
  title text,
  description text,
  duration_seconds int,
  thumbnail_url text,
  preview_video_url text,
  maker text,
  genre text,
  price int,
  sample_video_url text,
  image_urls jsonb,
  published_at timestamptz,
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
    v.duration_seconds,
    v.thumbnail_url,
    v.preview_video_url,
    v.maker,
    v.genre,
    v.price,
    v.sample_video_url,
    v.image_urls,
    v.published_at,
    coalesce(
      json_agg(
        distinct jsonb_build_object('name', p.name)
      ) filter (where p.name is not null),
      '[]'::json
    )::jsonb as performers,
    coalesce(
      json_agg(
        distinct jsonb_build_object('name', t.name)
      ) filter (where t.name is not null),
      '[]'::json
    )::jsonb as tags
  from videos v
  left join video_performers vp on v.id = vp.video_id
  left join performers p on vp.performer_id = p.id
  left join video_tags vt on v.id = vt.video_id
  left join tags t on vt.tag_id = t.id
  where not (v.id = any(exclude_video_ids))
  group by v.id, v.title, v.description, v.duration_seconds, v.thumbnail_url,
           v.preview_video_url, v.maker, v.genre, v.price, v.sample_video_url,
           v.image_urls, v.published_at
  order by random()
  limit page_limit;
end;
$$;

-- -----------------------------------------------------------------------------
-- Function 2: match_videos_cosine
-- ベクター類似度検索関数（Edge Functions/content/_shared/content.tsで使用）
-- -----------------------------------------------------------------------------

create or replace function public.match_videos_cosine(
  query_embedding vector(768),
  match_threshold float default 0.1,
  match_count int default 50
)
returns table (
  id uuid,
  similarity float
)
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  select
    ve.video_id as id,
    (1 - (ve.embedding <=> query_embedding)) as similarity
  from video_embeddings ve
  where (1 - (ve.embedding <=> query_embedding)) > match_threshold
  order by ve.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- -----------------------------------------------------------------------------
-- Function 3: get_personalized_recommendations
-- パーソナライズド推薦関数（Edge Functions/content/videos-feed/index.tsで使用）
-- -----------------------------------------------------------------------------

create or replace function public.get_personalized_recommendations(
  p_user_id uuid,
  p_limit int default 20,
  p_offset int default 0,
  p_similarity_threshold float default 0.5,
  p_exclude_seen boolean default true,
  p_genre_filter text[] default null,
  p_maker_filter text[] default null
)
returns table (
  id uuid,
  title text,
  description text,
  thumbnail_url text,
  preview_video_url text,
  maker text,
  genre text,
  price int,
  rating float,
  image_urls jsonb,
  performers jsonb,
  tags jsonb,
  similarity_score float
)
language plpgsql
security definer
set search_path = public
as $$
declare
  user_embedding vector(768);
begin
  -- ユーザー埋め込みを取得
  select embedding into user_embedding
  from user_embeddings
  where user_id = p_user_id;

  if user_embedding is null then
    -- ユーザー埋め込みがない場合は空結果を返す
    return;
  end if;

  return query
  select
    v.id,
    v.title,
    v.description,
    v.thumbnail_url,
    v.preview_video_url,
    v.maker,
    v.genre,
    v.price,
    coalesce(v.rating, 0.0) as rating,
    v.image_urls,
    coalesce(
      json_agg(
        distinct jsonb_build_object('name', p.name)
      ) filter (where p.name is not null),
      '[]'::json
    )::jsonb as performers,
    coalesce(
      json_agg(
        distinct jsonb_build_object('name', t.name)
      ) filter (where t.name is not null),
      '[]'::json
    )::jsonb as tags,
    (1 - (ve.embedding <=> user_embedding)) as similarity_score
  from videos v
  inner join video_embeddings ve on v.id = ve.video_id
  left join video_performers vp on v.id = vp.video_id
  left join performers p on vp.performer_id = p.id
  left join video_tags vt on v.id = vt.video_id
  left join tags t on vt.tag_id = t.id
  left join likes l on v.id = l.video_id and l.user_id = p_user_id
  where
    (1 - (ve.embedding <=> user_embedding)) >= p_similarity_threshold
    and (not p_exclude_seen or l.id is null)
    and (p_genre_filter is null or v.genre = any(p_genre_filter))
    and (p_maker_filter is null or v.maker = any(p_maker_filter))
  group by v.id, v.title, v.description, v.thumbnail_url, v.preview_video_url,
           v.maker, v.genre, v.price, v.rating, v.image_urls, ve.embedding
  order by (1 - (ve.embedding <=> user_embedding)) desc
  offset p_offset
  limit p_limit;
end;
$$;

-- -----------------------------------------------------------------------------
-- Function 4: get_videos_feed (Enhanced version for backwards compatibility)
-- 汎用動画フィード関数（複数のEdge Functionsで使用）
-- -----------------------------------------------------------------------------

create or replace function public.get_videos_feed(
  page_limit int default 20,
  page_offset int default 0,
  feed_type text default 'latest',
  user_id_param uuid default null
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
begin
  case feed_type
    when 'personalized' then
      if user_id_param is not null then
        return query select * from public.get_personalized_videos_feed(user_id_param, page_limit);
      else
        return query select * from public.get_popular_videos_feed(page_limit);
      end if;

    when 'popular' then
      return query select * from public.get_popular_videos_feed(page_limit);

    when 'explore' then
      return query select * from public.get_explore_videos_feed(page_limit);

    when 'random' then
      return query
      select
        rv.id,
        rv.title,
        rv.description,
        null::text as external_id,
        rv.thumbnail_url,
        rv.sample_video_url,
        rv.preview_video_url,
        null::text as product_url,
        rv.published_at as product_released_at,
        rv.performers,
        rv.tags,
        0.5 as recommendation_score,
        'Random selection' as recommendation_reason
      from public.get_random_videos(page_limit) as rv;

    else -- default to latest
      return query
      select
        v.id,
        v.title,
        v.description,
        v.external_id,
        v.thumbnail_url,
        v.sample_video_url,
        v.preview_video_url,
        v.product_url,
        v.published_at as product_released_at,
        coalesce(
          json_agg(
            distinct jsonb_build_object('name', p.name)
          ) filter (where p.name is not null),
          '[]'::json
        )::jsonb as performers,
        coalesce(
          json_agg(
            distinct jsonb_build_object('name', t.name)
          ) filter (where t.name is not null),
          '[]'::json
        )::jsonb as tags,
        0.0 as recommendation_score,
        'Latest content' as recommendation_reason
      from videos v
      left join video_performers vp on v.id = vp.video_id
      left join performers p on vp.performer_id = p.id
      left join video_tags vt on v.id = vt.video_id
      left join tags t on vt.tag_id = t.id
      group by v.id, v.title, v.description, v.external_id, v.thumbnail_url,
               v.sample_video_url, v.preview_video_url, v.product_url, v.published_at
      order by v.published_at desc nulls last
      offset page_offset
      limit page_limit;
  end case;
end;
$$;

-- -----------------------------------------------------------------------------
-- Function 5: get_user_video_history
-- ユーザー動画履歴取得関数（推薦システムで使用）
-- -----------------------------------------------------------------------------

create or replace function public.get_user_video_history(
  p_user_id uuid,
  p_limit int default 100
)
returns table (
  video_id uuid,
  interaction_type text,
  created_at timestamptz,
  video_embedding vector(768)
)
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  select
    l.video_id,
    'like'::text as interaction_type,
    l.created_at,
    ve.embedding as video_embedding
  from likes l
  inner join video_embeddings ve on l.video_id = ve.video_id
  where l.user_id = p_user_id
  order by l.created_at desc
  limit p_limit;
end;
$$;

-- -----------------------------------------------------------------------------
-- Function 6: update_user_embedding_from_likes
-- いいね履歴からユーザー埋め込みを更新する関数
-- -----------------------------------------------------------------------------

create or replace function public.update_user_embedding_from_likes(
  p_user_id uuid
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  avg_embedding vector(768);
  like_count int;
begin
  -- ユーザーのいいね履歴から平均埋め込みを計算
  select
    avg(ve.embedding)::vector(768),
    count(*)
  into avg_embedding, like_count
  from likes l
  inner join video_embeddings ve on l.video_id = ve.video_id
  where l.user_id = p_user_id;

  if like_count > 0 and avg_embedding is not null then
    -- ユーザー埋め込みを挿入または更新
    insert into user_embeddings (user_id, embedding, updated_at)
    values (p_user_id, avg_embedding, now())
    on conflict (user_id) do update set
      embedding = excluded.embedding,
      updated_at = excluded.updated_at;

    return true;
  end if;

  return false;
end;
$$;

-- -----------------------------------------------------------------------------
-- Function 7: calculate_video_similarity
-- 動画間類似度計算関数
-- -----------------------------------------------------------------------------

create or replace function public.calculate_video_similarity(
  video_id1 uuid,
  video_id2 uuid
)
returns float
language plpgsql
security definer
set search_path = public
as $$
declare
  embedding1 vector(768);
  embedding2 vector(768);
begin
  select embedding into embedding1 from video_embeddings where video_id = video_id1;
  select embedding into embedding2 from video_embeddings where video_id = video_id2;

  if embedding1 is null or embedding2 is null then
    return 0.0;
  end if;

  return (1 - (embedding1 <=> embedding2));
end;
$$;

-- -----------------------------------------------------------------------------
-- Function 8: get_similar_videos_to_video
-- 指定動画に類似した動画を取得する関数
-- -----------------------------------------------------------------------------

create or replace function public.get_similar_videos_to_video(
  p_video_id uuid,
  p_limit int default 10,
  p_similarity_threshold float default 0.3
)
returns table (
  id uuid,
  title text,
  thumbnail_url text,
  similarity_score float
)
language plpgsql
security definer
set search_path = public
as $$
declare
  target_embedding vector(768);
begin
  -- ターゲット動画の埋め込みを取得
  select embedding into target_embedding
  from video_embeddings
  where video_id = p_video_id;

  if target_embedding is null then
    return;
  end if;

  return query
  select
    v.id,
    v.title,
    v.thumbnail_url,
    (1 - (ve.embedding <=> target_embedding)) as similarity_score
  from videos v
  inner join video_embeddings ve on v.id = ve.video_id
  where
    v.id != p_video_id
    and (1 - (ve.embedding <=> target_embedding)) >= p_similarity_threshold
  order by ve.embedding <=> target_embedding
  limit p_limit;
end;
$$;

-- -----------------------------------------------------------------------------
-- RLS Policies for new functions
-- 新しい関数のためのセキュリティポリシー
-- -----------------------------------------------------------------------------

-- user_embeddingsテーブルのRLSポリシー（もしまだなければ）
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
    and tablename = 'user_embeddings'
    and policyname = 'Users can access own embeddings'
  ) then
    create policy "Users can access own embeddings"
    on user_embeddings for all
    using (auth.uid() = user_id);
  end if;
end $$;

-- video_embeddingsテーブルのRLSポリシー（もしまだなければ）
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
    and tablename = 'video_embeddings'
    and policyname = 'Anyone can read video embeddings'
  ) then
    create policy "Anyone can read video embeddings"
    on video_embeddings for select
    using (true);
  end if;
end $$;

-- -----------------------------------------------------------------------------
-- Grant permissions for functions
-- 関数の権限設定
-- -----------------------------------------------------------------------------

-- Public access to all feed functions
grant execute on function public.get_random_videos to anon, authenticated;
grant execute on function public.match_videos_cosine to anon, authenticated;
grant execute on function public.get_personalized_recommendations to authenticated;
grant execute on function public.get_videos_feed to anon, authenticated;
grant execute on function public.get_user_video_history to authenticated;
grant execute on function public.update_user_embedding_from_likes to authenticated;
grant execute on function public.calculate_video_similarity to anon, authenticated;
grant execute on function public.get_similar_videos_to_video to anon, authenticated;

-- -----------------------------------------------------------------------------
-- Create indexes for performance optimization
-- パフォーマンス最適化のためのインデックス作成
-- -----------------------------------------------------------------------------

-- Video embeddings cosine similarity index (if not exists)
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'video_embeddings'
    and indexname = 'video_embeddings_embedding_cosine_idx'
  ) then
    create index video_embeddings_embedding_cosine_idx
    on video_embeddings
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
  end if;
end $$;

-- User embeddings cosine similarity index (if not exists)
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'user_embeddings'
    and indexname = 'user_embeddings_embedding_cosine_idx'
  ) then
    create index user_embeddings_embedding_cosine_idx
    on user_embeddings
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
  end if;
end $$;

-- Video performance indexes
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'videos'
    and indexname = 'videos_published_at_idx'
  ) then
    create index videos_published_at_idx on videos (published_at desc nulls last);
  end if;

  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'videos'
    and indexname = 'videos_genre_idx'
  ) then
    create index videos_genre_idx on videos (genre);
  end if;

  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'videos'
    and indexname = 'videos_maker_idx'
  ) then
    create index videos_maker_idx on videos (maker);
  end if;

  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'videos'
    and indexname = 'videos_rating_idx'
  ) then
    create index videos_rating_idx on videos (rating desc nulls last);
  end if;
end $$;

-- Likes performance index
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'likes'
    and indexname = 'likes_user_created_idx'
  ) then
    create index likes_user_created_idx on likes (user_id, created_at desc);
  end if;
end $$;

-- -----------------------------------------------------------------------------
-- Function usage documentation
-- 関数使用方法の文書化
-- -----------------------------------------------------------------------------

comment on function public.get_random_videos is 'ランダム動画取得：Edge Functions/content/feed/index.tsで使用';
comment on function public.match_videos_cosine is 'ベクター類似度検索：Edge Functions/content/_shared/content.tsで使用';
comment on function public.get_personalized_recommendations is 'パーソナライズド推薦：Edge Functions/content/videos-feed/index.tsで使用';
comment on function public.get_videos_feed is '汎用動画フィード：複数のEdge Functionsで使用（後方互換性）';
comment on function public.get_user_video_history is 'ユーザー動画履歴：推薦システム内部で使用';
comment on function public.update_user_embedding_from_likes is 'ユーザー埋め込み更新：いいね履歴から自動更新';
comment on function public.calculate_video_similarity is '動画間類似度計算：分析・推薦で使用';
comment on function public.get_similar_videos_to_video is '類似動画取得：関連動画表示で使用';

-- Migration complete
select 'Missing RPC functions migration completed successfully'::text as status;