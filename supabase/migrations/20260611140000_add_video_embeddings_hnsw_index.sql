-- get_videos_recommendations のベクトル近傍探索を高速化するHNSWインデックス
CREATE INDEX IF NOT EXISTS video_embeddings_embedding_hnsw_idx
  ON public.video_embeddings
  USING hnsw (embedding halfvec_ip_ops)
  WITH (m = 16, ef_construction = 64);
