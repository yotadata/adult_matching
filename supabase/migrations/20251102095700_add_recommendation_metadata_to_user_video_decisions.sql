alter table public.user_video_decisions
  add column if not exists recommendation_source text,
  add column if not exists recommendation_score double precision,
  add column if not exists recommendation_model_version text,
  add column if not exists recommendation_params jsonb;
