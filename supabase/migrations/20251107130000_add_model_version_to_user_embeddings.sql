-- Adds model_version column to user_embeddings for tracking embedding source model
BEGIN;

ALTER TABLE public.user_embeddings
  ADD COLUMN IF NOT EXISTS model_version text;

COMMIT;
