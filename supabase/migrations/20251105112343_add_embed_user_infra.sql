-- ユーザーの特徴量（集計済み）を保存するテーブル
create table public.user_features (
    user_id uuid not null primary key references auth.users (id) on delete cascade,
    liked_tags jsonb not null default '{}'::jsonb,
    liked_performers jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

alter table public.user_features enable row level security;

-- ユーザー本人のみ自分の特徴量を閲覧・編集可能
create policy "Allow individual user access" on public.user_features
    for all
    using (auth.uid() = user_id);

-- service_role ロールが存在しないローカル環境でも GRANT できるよう、必要に応じて作成
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'service_role') then
        create role service_role;
    end if;
end
$$;

-- service_role は全アクセス可能
grant select, insert, update, delete on table public.user_features to service_role;


-- 特定ユーザーの「いいね」履歴からタグとパフォーマーを集計し、user_featuresテーブルを更新するRPC
create or replace function public.update_user_features(p_user_id uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
    tags_summary jsonb;
    performers_summary jsonb;
begin
    -- いいねした動画のタグを集計
    select coalesce(jsonb_object_agg(tag_id, count), '{}'::jsonb)
    into tags_summary
    from (
        select t.id as tag_id, count(*) as count
        from user_video_decisions uvd
        join video_tags vt on uvd.video_id = vt.video_id
        join tags t on vt.tag_id = t.id
        where uvd.user_id = p_user_id
          and uvd.decision_type = 'like'
        group by t.id
    ) as sub;

    -- いいねした動画のパフォーマーを集計
    select coalesce(jsonb_object_agg(performer_id, count), '{}'::jsonb)
    into performers_summary
    from (
        select p.id as performer_id, count(*) as count
        from user_video_decisions uvd
        join video_performers vp on uvd.video_id = vp.video_id
        join performers p on vp.performer_id = p.id
        where uvd.user_id = p_user_id
          and uvd.decision_type = 'like'
        group by p.id
    ) as sub;

    -- user_features テーブルに upsert
    insert into public.user_features (user_id, liked_tags, liked_performers, updated_at)
    values (p_user_id, tags_summary, performers_summary, now())
    on conflict (user_id) do update
    set
        liked_tags = excluded.liked_tags,
        liked_performers = excluded.liked_performers,
        updated_at = excluded.updated_at;

end;
$$;

grant execute on function public.update_user_features(uuid) to authenticated;
