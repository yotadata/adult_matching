-- フィードRPC最適化（CI対応・冪等版）
-- 20260612200000〜220000 はリモート直接適用済みのstub。
-- このマイグレーションが CI での正規適用版。
-- CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS で冪等に動作する。

-- user_excluded_tags テーブル（remote-only stubで作成済みだがCI向けに再作成）
CREATE TABLE IF NOT EXISTS public.user_excluded_tags (
  user_id    uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  tag_id     uuid NOT NULL REFERENCES public.tags(id) ON DELETE CASCADE,
  created_at timestamptz DEFAULT now(),
  PRIMARY KEY (user_id, tag_id)
);

ALTER TABLE public.user_excluded_tags ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'user_excluded_tags'
      AND policyname = 'user_excluded_tags_owner_all'
  ) THEN
    CREATE POLICY "user_excluded_tags_owner_all"
      ON public.user_excluded_tags FOR ALL USING (auth.uid() = user_id);
  END IF;
END $$;

CREATE OR REPLACE FUNCTION public.set_default_excluded_tags()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO public.user_excluded_tags (user_id, tag_id)
  SELECT NEW.user_id, t.id
  FROM public.tags t
  WHERE t.name IN ('スカトロ', '人格排泄')
  ON CONFLICT DO NOTHING;
  RETURN NEW;
END;
$$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgname = 'trg_set_default_excluded_tags'
  ) THEN
    CREATE TRIGGER trg_set_default_excluded_tags
      AFTER INSERT ON public.user_profiles
      FOR EACH ROW EXECUTE FUNCTION public.set_default_excluded_tags();
  END IF;
END $$;

CREATE OR REPLACE FUNCTION public.get_user_excluded_tags()
RETURNS json LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  RETURN (
    SELECT json_agg(json_build_object('tag_id', uet.tag_id, 'name', t.name))
    FROM public.user_excluded_tags uet
    JOIN public.tags t ON t.id = uet.tag_id
    WHERE uet.user_id = auth.uid()
  );
END; $$;
GRANT EXECUTE ON FUNCTION public.get_user_excluded_tags() TO authenticated;

CREATE OR REPLACE FUNCTION public.add_user_excluded_tag(p_tag_id uuid)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO public.user_excluded_tags (user_id, tag_id)
  VALUES (auth.uid(), p_tag_id)
  ON CONFLICT DO NOTHING;
END; $$;
GRANT EXECUTE ON FUNCTION public.add_user_excluded_tag(uuid) TO authenticated;

CREATE OR REPLACE FUNCTION public.remove_user_excluded_tag(p_tag_id uuid)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  DELETE FROM public.user_excluded_tags
  WHERE user_id = auth.uid() AND tag_id = p_tag_id;
END; $$;
GRANT EXECUTE ON FUNCTION public.remove_user_excluded_tag(uuid) TO authenticated;


-- インデックス
CREATE INDEX IF NOT EXISTS video_tags_video_id_idx
  ON public.video_tags (video_id);

CREATE INDEX IF NOT EXISTS video_performers_video_id_idx
  ON public.video_performers (video_id);

CREATE INDEX IF NOT EXISTS user_excluded_tags_user_id_idx
  ON public.user_excluded_tags (user_id);

DROP INDEX IF EXISTS public.videos_feed_eligible_idx;
CREATE INDEX IF NOT EXISTS videos_feed_eligible_idx
  ON public.videos (id)
  WHERE sample_video_url IS NOT NULL
    AND source NOT IN ('FANZA_ANIME')
    AND (has_performer = true OR source IN ('FANZA_AMATEUR', 'mgs'));


-- get_videos_feed（UUID スプリット + has_performer + バッチJOIN）
DROP FUNCTION IF EXISTS public.get_videos_feed(int);
CREATE OR REPLACE FUNCTION public.get_videos_feed(page_limit int DEFAULT 20)
RETURNS TABLE(
  id uuid, title text, description text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, preview_video_url text,
  product_url text, product_released_at timestamptz,
  performers jsonb, tags jsonb, source text, image_urls text[]
)
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  current_uid    uuid;
  excluded_tag_ids uuid[];
  cutpoint       uuid;
BEGIN
  current_uid := auth.uid();
  cutpoint    := gen_random_uuid();

  IF current_uid IS NOT NULL THEN
    SELECT array_agg(uet.tag_id) INTO excluded_tag_ids
    FROM public.user_excluded_tags uet
    WHERE uet.user_id = current_uid;
  END IF;

  RETURN QUERY
  WITH base AS (
    (
      SELECT
        v.id, v.title, v.description, v.external_id,
        v.thumbnail_url, v.thumbnail_vertical_url,
        v.sample_video_url, v.preview_video_url,
        coalesce(v.affiliate_url, v.product_url) AS product_url,
        v.product_released_at, v.source, v.image_urls
      FROM public.videos v
      WHERE v.id >= cutpoint
        AND v.sample_video_url IS NOT NULL
        AND v.source NOT IN ('FANZA_ANIME')
        AND (v.has_performer = true OR v.source IN ('FANZA_AMATEUR', 'mgs'))
        AND (current_uid IS NULL OR NOT EXISTS (
          SELECT 1 FROM public.user_video_decisions uvd
          WHERE uvd.video_id = v.id AND uvd.user_id = current_uid))
        AND (excluded_tag_ids IS NULL OR NOT EXISTS (
          SELECT 1 FROM public.video_tags vt2
          WHERE vt2.video_id = v.id AND vt2.tag_id = ANY(excluded_tag_ids)))
      ORDER BY v.id LIMIT page_limit
    )
    UNION ALL
    (
      SELECT
        v.id, v.title, v.description, v.external_id,
        v.thumbnail_url, v.thumbnail_vertical_url,
        v.sample_video_url, v.preview_video_url,
        coalesce(v.affiliate_url, v.product_url) AS product_url,
        v.product_released_at, v.source, v.image_urls
      FROM public.videos v
      WHERE v.id < cutpoint
        AND v.sample_video_url IS NOT NULL
        AND v.source NOT IN ('FANZA_ANIME')
        AND (v.has_performer = true OR v.source IN ('FANZA_AMATEUR', 'mgs'))
        AND (current_uid IS NULL OR NOT EXISTS (
          SELECT 1 FROM public.user_video_decisions uvd
          WHERE uvd.video_id = v.id AND uvd.user_id = current_uid))
        AND (excluded_tag_ids IS NULL OR NOT EXISTS (
          SELECT 1 FROM public.video_tags vt2
          WHERE vt2.video_id = v.id AND vt2.tag_id = ANY(excluded_tag_ids)))
      ORDER BY v.id LIMIT page_limit
    )
  ),
  base_ids AS (SELECT id FROM base LIMIT page_limit),
  perf_agg AS (
    SELECT vp.video_id,
           jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name) AS agg
    FROM public.video_performers vp
    JOIN public.performers p ON p.id = vp.performer_id
    WHERE vp.video_id IN (SELECT id FROM base_ids)
    GROUP BY vp.video_id
  ),
  tags_agg AS (
    SELECT vt.video_id,
           jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name) AS agg
    FROM public.video_tags vt
    JOIN public.tags t ON t.id = vt.tag_id
    LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
    WHERE vt.video_id IN (SELECT id FROM base_ids)
      AND coalesce(tg.show_in_ui, true)
    GROUP BY vt.video_id
  )
  SELECT
    b.id, b.title, b.description, b.external_id,
    b.thumbnail_url, b.thumbnail_vertical_url,
    b.sample_video_url, b.preview_video_url,
    b.product_url, b.product_released_at,
    coalesce(pa.agg, '[]'::jsonb) AS performers,
    coalesce(ta.agg, '[]'::jsonb) AS tags,
    b.source, b.image_urls
  FROM base b
  JOIN base_ids bi ON bi.id = b.id
  LEFT JOIN perf_agg pa ON pa.video_id = b.id
  LEFT JOIN tags_agg ta ON ta.video_id = b.id;
END; $$;

GRANT EXECUTE ON FUNCTION public.get_videos_feed(int) TO anon, authenticated;


-- get_videos_recommendations（バッチJOIN + decision_type バグ修正）
DROP FUNCTION IF EXISTS public.get_videos_recommendations(uuid, int);
CREATE OR REPLACE FUNCTION public.get_videos_recommendations(user_uuid uuid, page_limit int DEFAULT 20)
RETURNS TABLE (
  id uuid, title text, description text, external_id text,
  thumbnail_url text, thumbnail_vertical_url text,
  sample_video_url text, product_url text, product_released_at timestamptz,
  performers jsonb, tags jsonb,
  score double precision, model_version text, source text, image_urls text[]
)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
  user_vec       halfvec(128);
  approx_vec     halfvec(128);
  latest_version text;
BEGIN
  SELECT avg(ve.embedding)::halfvec(128) INTO approx_vec
  FROM (
    SELECT uvd.video_id
    FROM public.user_video_decisions uvd
    WHERE uvd.user_id = user_uuid
      AND uvd.decision_type IN ('swipe_like', 'grid_like')
    ORDER BY uvd.created_at DESC LIMIT 200
  ) recent_likes
  JOIN public.video_embeddings ve ON ve.video_id = recent_likes.video_id;

  IF approx_vec IS NULL THEN
    SELECT ue.embedding INTO user_vec FROM public.user_embeddings ue
    WHERE ue.user_id = user_uuid;
  END IF;

  user_vec := coalesce(approx_vec, user_vec);
  IF user_vec IS NULL THEN RETURN; END IF;

  SELECT ve.model_version INTO latest_version
  FROM public.video_embeddings ve
  WHERE ve.model_version IS NOT NULL
  ORDER BY ve.updated_at DESC LIMIT 1;

  RETURN QUERY
  WITH candidates AS (
    SELECT
      v.id, v.title, v.description, v.external_id,
      v.thumbnail_url, v.thumbnail_vertical_url, v.sample_video_url,
      coalesce(v.affiliate_url, v.product_url) AS product_url,
      v.product_released_at,
      1 - (ve.embedding <-> user_vec) AS score,
      ve.model_version, v.source, v.image_urls
    FROM public.video_embeddings ve
    JOIN public.videos v ON v.id = ve.video_id
    WHERE v.sample_video_url IS NOT NULL
      AND (latest_version IS NULL OR ve.model_version = latest_version)
      AND NOT EXISTS (
        SELECT 1 FROM public.user_video_decisions uvd
        WHERE uvd.video_id = v.id AND uvd.user_id = user_uuid)
      AND NOT EXISTS (
        SELECT 1 FROM public.user_excluded_tags uet
        JOIN public.video_tags vt ON vt.tag_id = uet.tag_id
        WHERE vt.video_id = v.id AND uet.user_id = user_uuid)
    ORDER BY ve.embedding <-> user_vec LIMIT page_limit
  ),
  perf_agg AS (
    SELECT vp.video_id,
           jsonb_agg(jsonb_build_object('id', p.id, 'name', p.name) ORDER BY p.name) AS agg
    FROM public.video_performers vp
    JOIN public.performers p ON p.id = vp.performer_id
    WHERE vp.video_id IN (SELECT id FROM candidates)
    GROUP BY vp.video_id
  ),
  tags_agg AS (
    SELECT vt.video_id,
           jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name) ORDER BY t.name) AS agg
    FROM public.video_tags vt
    JOIN public.tags t ON t.id = vt.tag_id
    LEFT JOIN public.tag_groups tg ON tg.id = t.tag_group_id
    WHERE vt.video_id IN (SELECT id FROM candidates)
      AND coalesce(tg.show_in_ui, true)
    GROUP BY vt.video_id
  )
  SELECT
    c.id, c.title, c.description, c.external_id,
    c.thumbnail_url, c.thumbnail_vertical_url, c.sample_video_url,
    c.product_url, c.product_released_at,
    coalesce(pa.agg, '[]'::jsonb) AS performers,
    coalesce(ta.agg, '[]'::jsonb) AS tags,
    c.score, c.model_version, c.source, c.image_urls
  FROM candidates c
  LEFT JOIN perf_agg pa ON pa.video_id = c.id
  LEFT JOIN tags_agg ta ON ta.video_id = c.id
  ORDER BY c.score DESC;
END; $$;

GRANT EXECUTE ON FUNCTION public.get_videos_recommendations(uuid, int) TO anon, authenticated;
