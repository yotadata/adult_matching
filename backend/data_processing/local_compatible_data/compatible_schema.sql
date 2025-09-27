-- Supabase互換テーブル構造定義
-- ローカル環境用

-- UUIDサポート
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- プロファイルテーブル
CREATE TABLE IF NOT EXISTS public.profiles (
    user_id UUID PRIMARY KEY,
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    original_reviewer_id TEXT,
    email TEXT
);

-- 動画テーブル（サブセット）
CREATE TABLE IF NOT EXISTS public.videos_subset (
    id UUID PRIMARY KEY,
    external_id TEXT UNIQUE,
    title TEXT,
    thumbnail_url TEXT,
    price NUMERIC,
    product_released_at TIMESTAMPTZ,
    original_content_id TEXT
);

-- ユーザー動画判定テーブル
CREATE TABLE IF NOT EXISTS public.user_video_decisions (
    user_id UUID REFERENCES public.profiles(user_id),
    video_id UUID REFERENCES public.videos_subset(id),
    decision_type TEXT NOT NULL CHECK (decision_type IN ('like', 'nope')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    original_rating NUMERIC,
    original_reviewer_id TEXT,
    original_content_id TEXT,
    PRIMARY KEY (user_id, video_id)
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_uvd_user_decision ON public.user_video_decisions(user_id, decision_type);
CREATE INDEX IF NOT EXISTS idx_uvd_created_at ON public.user_video_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_videos_external_id ON public.videos_subset(external_id);
