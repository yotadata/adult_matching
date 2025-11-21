-- Track which surface the decision came from (e.g., swipe feed, AI search)
alter table public.user_video_decisions
  add column if not exists recommendation_type text;

create index if not exists idx_uvd_recommendation_type on public.user_video_decisions (recommendation_type);
