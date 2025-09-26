-- =============================================================================
-- Schema Dependencies Fix and Missing Components
--
-- スキーマ依存関係の修正と欠落コンポーネントの補完
-- Edge Functions統合のための追加修正
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Check and add missing columns to videos table
-- videosテーブルの欠落カラムの確認と追加
-- -----------------------------------------------------------------------------

-- Add rating column if it doesn't exist (referenced in some Edge Functions)
do $$
begin
  if not exists (
    select 1 from information_schema.columns
    where table_schema = 'public'
    and table_name = 'videos'
    and column_name = 'rating'
  ) then
    alter table public.videos add column rating float;
    comment on column public.videos.rating is 'Video rating score for popularity calculation';
  end if;
end $$;

-- Add product_url column if it doesn't exist (referenced in personalized feed functions)
do $$
begin
  if not exists (
    select 1 from information_schema.columns
    where table_schema = 'public'
    and table_name = 'videos'
    and column_name = 'product_url'
  ) then
    alter table public.videos add column product_url text;
    comment on column public.videos.product_url is 'URL to product page for video purchase';
  end if;
end $$;

-- -----------------------------------------------------------------------------
-- Fix get_videos_feed function to handle new parameters
-- 新しいパラメータに対応するためのget_videos_feed関数の修正
-- -----------------------------------------------------------------------------

-- Create overloaded version that accepts offset parameter
create or replace function public.get_videos_feed(
  page_limit int default 20,
  page_offset int default 0
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
  tags jsonb
)
language plpgsql
security definer
set search_path = public
as $$
declare
  total_count int;
  start_offset int := page_offset;
begin
  -- If no specific offset provided, use random offset for discovery
  if page_offset = 0 then
    -- Candidate set: has sample url and not judged by current user
    with filtered as (
      select v.*
      from public.videos v
      where v.sample_video_url is not null
        and not exists (
          select 1 from public.user_video_decisions uvd
          where uvd.video_id = v.id
            and (auth.uid() is not null and uvd.user_id = auth.uid())
        )
    )
    select count(*) into total_count from filtered;

    if total_count > page_limit then
      start_offset := floor(random() * (total_count - page_limit))::int;
    else
      start_offset := 0;
    end if;
  end if;

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
        where vt.video_id = v.id
      ), '[]'::jsonb
    ) as tags
  from public.videos v
  where v.id in (
    select s.id from (
      select v2.id as id from public.videos v2
      where v2.sample_video_url is not null
        and not exists (
          select 1 from public.user_video_decisions uvd2
          where uvd2.video_id = v2.id
            and (auth.uid() is not null and uvd2.user_id = auth.uid())
        )
      order by v2.id
      offset start_offset limit page_limit
    ) s
  )
  order by v.id;
end;
$$;

grant execute on function public.get_videos_feed(int, int) to anon, authenticated;

-- -----------------------------------------------------------------------------
-- Create missing utility functions that might be needed
-- 必要かもしれない欠落ユーティリティ関数の作成
-- -----------------------------------------------------------------------------

-- Function to get video count for pagination
create or replace function public.get_total_video_count(
  include_decided boolean default false
)
returns int
language plpgsql
security definer
set search_path = public
as $$
declare
  count_result int;
begin
  if include_decided then
    select count(*) into count_result from public.videos;
  else
    select count(*) into count_result
    from public.videos v
    where not exists (
      select 1 from public.user_video_decisions uvd
      where uvd.video_id = v.id
        and (auth.uid() is not null and uvd.user_id = auth.uid())
    );
  end if;

  return count_result;
end;
$$;

grant execute on function public.get_total_video_count to anon, authenticated;

-- Function to get user's decision count
create or replace function public.get_user_decision_count(
  p_user_id uuid default null,
  decision_type text default null
)
returns int
language plpgsql
security definer
set search_path = public
as $$
declare
  target_user_id uuid;
  count_result int;
begin
  target_user_id := coalesce(p_user_id, auth.uid());

  if target_user_id is null then
    return 0;
  end if;

  if decision_type is null then
    select count(*) into count_result
    from public.user_video_decisions
    where user_id = target_user_id;
  else
    select count(*) into count_result
    from public.user_video_decisions
    where user_id = target_user_id
    and decision_type = decision_type;
  end if;

  return count_result;
end;
$$;

grant execute on function public.get_user_decision_count to authenticated;

-- -----------------------------------------------------------------------------
-- Create function to clean up old user decisions (maintenance)
-- 古いユーザー決定をクリーンアップする関数（メンテナンス用）
-- -----------------------------------------------------------------------------

create or replace function public.cleanup_old_user_decisions(
  days_old int default 365
)
returns int
language plpgsql
security definer
set search_path = public
as $$
declare
  deleted_count int;
begin
  -- Only allow authenticated users to clean their own data
  if auth.uid() is null then
    return 0;
  end if;

  delete from public.user_video_decisions
  where user_id = auth.uid()
    and created_at < (now() - interval '1 day' * days_old);

  get diagnostics deleted_count = ROW_COUNT;
  return deleted_count;
end;
$$;

grant execute on function public.cleanup_old_user_decisions to authenticated;

-- -----------------------------------------------------------------------------
-- Create batch operations for better performance
-- パフォーマンス向上のためのバッチ操作
-- -----------------------------------------------------------------------------

-- Function to batch update video ratings
create or replace function public.batch_update_video_ratings()
returns int
language plpgsql
security definer
set search_path = public
as $$
declare
  updated_count int := 0;
begin
  -- Calculate ratings based on likes count and recency
  update public.videos
  set rating = coalesce(
    (
      select
        (count(l.user_id)::float * 0.7) +
        (extract(epoch from (now() - v.created_at)) / -86400.0 * 0.3)
      from public.likes l
      where l.video_id = videos.id
    ), 0.0
  )
  where rating is null or rating != coalesce(
    (
      select
        (count(l.user_id)::float * 0.7) +
        (extract(epoch from (now() - videos.created_at)) / -86400.0 * 0.3)
      from public.likes l
      where l.video_id = videos.id
    ), 0.0
  );

  get diagnostics updated_count = ROW_COUNT;
  return updated_count;
end;
$$;

-- Only allow this to be called by service role or authenticated admin users
grant execute on function public.batch_update_video_ratings to service_role;

-- -----------------------------------------------------------------------------
-- Create trigger to automatically update video ratings when likes change
-- いいねが変更された時に動画レーティングを自動更新するトリガー
-- -----------------------------------------------------------------------------

create or replace function public.update_video_rating_on_like_change()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  target_video_id uuid;
  new_rating float;
begin
  -- Get the video_id from the changed record
  if TG_OP = 'DELETE' then
    target_video_id := OLD.video_id;
  else
    target_video_id := NEW.video_id;
  end if;

  -- Calculate new rating for this specific video
  select
    coalesce(
      (count(l.user_id)::float * 0.8) +
      (extract(epoch from (now() - v.created_at)) / -86400.0 * 0.2),
      0.0
    )
  into new_rating
  from public.videos v
  left join public.likes l on l.video_id = v.id
  where v.id = target_video_id
  group by v.id, v.created_at;

  -- Update the rating
  update public.videos
  set rating = new_rating
  where id = target_video_id;

  if TG_OP = 'DELETE' then
    return OLD;
  else
    return NEW;
  end if;
end;
$$;

-- Create the trigger if it doesn't exist
drop trigger if exists trigger_update_video_rating_on_like_change on public.likes;
create trigger trigger_update_video_rating_on_like_change
  after insert or update or delete on public.likes
  for each row
  execute function public.update_video_rating_on_like_change();

-- -----------------------------------------------------------------------------
-- Create indexes for improved performance
-- パフォーマンス向上のための追加インデックス
-- -----------------------------------------------------------------------------

-- Composite index for user decisions
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'user_video_decisions'
    and indexname = 'user_video_decisions_user_video_idx'
  ) then
    create index user_video_decisions_user_video_idx
    on public.user_video_decisions (user_id, video_id);
  end if;
end $$;

-- Composite index for user decisions by decision type
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'user_video_decisions'
    and indexname = 'user_video_decisions_user_type_idx'
  ) then
    create index user_video_decisions_user_type_idx
    on public.user_video_decisions (user_id, decision_type);
  end if;
end $$;

-- Index for video external_id lookups
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'videos'
    and indexname = 'videos_external_id_idx'
  ) then
    create index videos_external_id_idx on public.videos (external_id);
  end if;
end $$;

-- Index for video source lookups
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
    and tablename = 'videos'
    and indexname = 'videos_source_idx'
  ) then
    create index videos_source_idx on public.videos (source);
  end if;
end $$;

-- -----------------------------------------------------------------------------
-- Update existing RLS policies to ensure proper access
-- 適切なアクセスを確保するための既存RLSポリシーの更新
-- -----------------------------------------------------------------------------

-- Ensure videos table has proper RLS for anonymous access
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
    and tablename = 'videos'
    and policyname = 'Anyone can read videos'
  ) then
    create policy "Anyone can read videos"
    on public.videos for select
    using (true);
  end if;
end $$;

-- Ensure video_performers has proper RLS
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
    and tablename = 'video_performers'
    and policyname = 'Anyone can read video performers'
  ) then
    create policy "Anyone can read video performers"
    on public.video_performers for select
    using (true);
  end if;
end $$;

-- Ensure performers table has proper RLS
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
    and tablename = 'performers'
    and policyname = 'Anyone can read performers'
  ) then
    create policy "Anyone can read performers"
    on public.performers for select
    using (true);
  end if;
end $$;

-- Ensure video_tags has proper RLS
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
    and tablename = 'video_tags'
    and policyname = 'Anyone can read video tags'
  ) then
    create policy "Anyone can read video tags"
    on public.video_tags for select
    using (true);
  end if;
end $$;

-- Ensure tags table has proper RLS
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
    and tablename = 'tags'
    and policyname = 'Anyone can read tags'
  ) then
    create policy "Anyone can read tags"
    on public.tags for select
    using (true);
  end if;
end $$;

-- -----------------------------------------------------------------------------
-- Function to validate database schema integrity
-- データベーススキーマ整合性を検証する関数
-- -----------------------------------------------------------------------------

create or replace function public.validate_schema_integrity()
returns table (
  table_name text,
  issue_type text,
  issue_description text,
  recommended_action text
)
language plpgsql
security definer
set search_path = public
as $$
begin
  -- Check for videos without embeddings
  return query
  select
    'videos/video_embeddings'::text as table_name,
    'missing_embeddings'::text as issue_type,
    format('Found %s videos without embeddings', count(*))::text as issue_description,
    'Run embedding generation for these videos'::text as recommended_action
  from public.videos v
  left join public.video_embeddings ve on v.id = ve.video_id
  where ve.video_id is null
  having count(*) > 0;

  -- Check for users with likes but no embeddings
  return query
  select
    'users/user_embeddings'::text as table_name,
    'missing_user_embeddings'::text as issue_type,
    format('Found %s users with likes but no embeddings', count(distinct l.user_id))::text as issue_description,
    'Run user embedding generation for these users'::text as recommended_action
  from public.likes l
  left join public.user_embeddings ue on l.user_id = ue.user_id
  where ue.user_id is null
  having count(distinct l.user_id) > 0;

  -- Check for orphaned video_performers records
  return query
  select
    'video_performers'::text as table_name,
    'orphaned_records'::text as issue_type,
    format('Found %s orphaned video_performers records', count(*))::text as issue_description,
    'Clean up orphaned records'::text as recommended_action
  from public.video_performers vp
  left join public.videos v on vp.video_id = v.id
  left join public.performers p on vp.performer_id = p.id
  where v.id is null or p.id is null
  having count(*) > 0;

  -- Check for orphaned video_tags records
  return query
  select
    'video_tags'::text as table_name,
    'orphaned_records'::text as issue_type,
    format('Found %s orphaned video_tags records', count(*))::text as issue_description,
    'Clean up orphaned records'::text as recommended_action
  from public.video_tags vt
  left join public.videos v on vt.video_id = v.id
  left join public.tags t on vt.tag_id = t.id
  where v.id is null or t.id is null
  having count(*) > 0;
end;
$$;

grant execute on function public.validate_schema_integrity to authenticated, service_role;

-- -----------------------------------------------------------------------------
-- Add function comments for documentation
-- 文書化のための関数コメント追加
-- -----------------------------------------------------------------------------

comment on function public.get_total_video_count is 'Get total count of videos, optionally excluding user decisions';
comment on function public.get_user_decision_count is 'Get count of user decisions, optionally filtered by decision type';
comment on function public.cleanup_old_user_decisions is 'Clean up user decisions older than specified days';
comment on function public.batch_update_video_ratings is 'Batch update video ratings based on likes and recency';
comment on function public.validate_schema_integrity is 'Validate database schema integrity and identify issues';

-- Migration completed successfully
select 'Schema dependencies fixed and missing components added successfully'::text as status;