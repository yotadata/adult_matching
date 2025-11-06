create table if not exists public.ai_recommend_playlists (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  mode_id text,
  custom_intent jsonb not null default '{}'::jsonb,
  items jsonb not null,
  notes text,
  visibility text not null default 'private' check (visibility in ('private', 'link', 'public')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_ai_recommend_playlists_user_id on public.ai_recommend_playlists(user_id);

create or replace function public.set_ai_recommend_playlists_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_ai_recommend_playlists_updated_at on public.ai_recommend_playlists;
create trigger set_ai_recommend_playlists_updated_at
before update on public.ai_recommend_playlists
for each row execute function public.set_ai_recommend_playlists_updated_at();

alter table public.ai_recommend_playlists enable row level security;

drop policy if exists "Users can manage own AI playlists" on public.ai_recommend_playlists;
create policy "Users can manage own AI playlists"
on public.ai_recommend_playlists
for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

grant select, insert, update, delete on public.ai_recommend_playlists to authenticated;
