alter table public.video_embeddings
  add column if not exists model_version text;
