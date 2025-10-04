-- Change embedding dimension from 768 to 256 for pgvector columns
-- Safe order: drop indexes -> alter column type -> recreate indexes

-- 1) Drop existing IVFFLAT indexes if present
DROP INDEX IF EXISTS idx_video_embeddings_cosine;
DROP INDEX IF EXISTS idx_user_embeddings_cosine;

-- 2) Alter column types to vector(256)
ALTER TABLE public.video_embeddings
  ALTER COLUMN embedding TYPE vector(256);

ALTER TABLE public.user_embeddings
  ALTER COLUMN embedding TYPE vector(256);

-- 3) Recreate IVFFLAT indexes (cosine)
CREATE INDEX IF NOT EXISTS idx_video_embeddings_cosine
  ON public.video_embeddings USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_user_embeddings_cosine
  ON public.user_embeddings USING ivfflat (embedding vector_cosine_ops);

