DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_ai_recommend_playlists_updated_at') THEN
    EXECUTE 'DROP TRIGGER set_ai_recommend_playlists_updated_at ON public.ai_recommend_playlists';
  END IF;
END$$;

DROP FUNCTION IF EXISTS public.set_ai_recommend_playlists_updated_at();
DROP TABLE IF EXISTS public.ai_recommend_playlists CASCADE;
