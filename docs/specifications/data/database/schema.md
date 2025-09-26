# PostgreSQL完全スキーマ仕様書

Adult Matching アプリケーションのPostgreSQL データベーススキーマ完全仕様

---

## 📋 スキーマ概要

### 拡張機能
- `uuid-ossp` - UUID生成
- `pgcrypto` - 暗号化関数
- `vector` - 埋め込みベクトル対応

### テーブル構成
```
videos (動画)
├── video_embeddings (動画埋め込み)
├── video_tags (動画タグ関係)
├── video_performers (動画出演者関係)
└── likes (ユーザーライク)

tags (タグ)
└── tag_groups (タググループ)

performers (出演者)

profiles (ユーザープロフィール)
└── user_embeddings (ユーザー埋め込み)
```

---

## 🎬 動画関連テーブル

### videos (動画メインテーブル)
```sql
CREATE TABLE public.videos (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id text UNIQUE,                    -- 外部ID (DMM/FANZA)
  title text NOT NULL,                        -- タイトル
  description text,                           -- 詳細説明
  duration_seconds int,                       -- 再生時間(秒)
  thumbnail_url text,                         -- サムネイル画像URL
  preview_video_url text,                     -- プレビュー動画URL
  distribution_code text,                     -- 配信コード
  maker_code text,                           -- メーカーコード
  director text,                             -- 監督
  series text,                               -- シリーズ
  maker text,                                -- メーカー
  label text,                                -- レーベル
  genre text,                                -- ジャンル
  price numeric,                             -- 価格
  distribution_started_at timestamptz,        -- 配信開始日
  product_released_at timestamptz,           -- 商品発売日
  sample_video_url text,                     -- サンプル動画URL
  image_urls text[],                         -- 画像URLリスト
  source text NOT NULL,                      -- データソース
  published_at timestamptz,                  -- 公開日時
  created_at timestamptz DEFAULT now(),      -- 作成日時
  UNIQUE (source, distribution_code, maker_code)
);
```

### video_embeddings (動画埋め込みベクトル)
```sql
CREATE TABLE public.video_embeddings (
  video_id uuid PRIMARY KEY REFERENCES public.videos(id) ON DELETE CASCADE,
  embedding vector(768),                     -- 768次元埋め込みベクトル
  updated_at timestamptz DEFAULT now()      -- 更新日時
);
```

---

## 🏷️ タグ・分類システム

### tag_groups (タググループ)
```sql
CREATE TABLE public.tag_groups (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL                         -- グループ名
);
```

### tags (タグ)
```sql
CREATE TABLE public.tags (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,                        -- タグ名
  tag_group_id uuid REFERENCES public.tag_groups(id)
);
```

### video_tags (動画タグ関係)
```sql
CREATE TABLE public.video_tags (
  video_id uuid REFERENCES public.videos(id) ON DELETE CASCADE,
  tag_id uuid REFERENCES public.tags(id) ON DELETE CASCADE,
  PRIMARY KEY (video_id, tag_id)
);
```

---

## 👥 出演者システム

### performers (出演者)
```sql
CREATE TABLE public.performers (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL                         -- 出演者名
);
```

### video_performers (動画出演者関係)
```sql
CREATE TABLE public.video_performers (
  video_id uuid REFERENCES public.videos(id) ON DELETE CASCADE,
  performer_id uuid REFERENCES public.performers(id) ON DELETE CASCADE,
  PRIMARY KEY (video_id, performer_id)
);
```

---

## 👤 ユーザー関連テーブル

### profiles (ユーザープロフィール)
```sql
CREATE TABLE public.profiles (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name text,                         -- 表示名
  created_at timestamptz DEFAULT now()      -- 作成日時
);
```

### user_embeddings (ユーザー埋め込みベクトル)
```sql
CREATE TABLE public.user_embeddings (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  embedding vector(768),                     -- 768次元埋め込みベクトル
  updated_at timestamptz DEFAULT now()      -- 更新日時
);
```

### likes (ユーザーライク)
```sql
CREATE TABLE public.likes (
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  video_id uuid REFERENCES public.videos(id) ON DELETE CASCADE,
  purchased boolean DEFAULT false,           -- 購入済みフラグ
  created_at timestamptz DEFAULT now(),     -- 作成日時
  PRIMARY KEY (user_id, video_id)
);
```

---

## 📊 集計・分析ビュー

### video_popularity_daily (人気動画日次集計)
```sql
CREATE MATERIALIZED VIEW public.video_popularity_daily AS
SELECT
  v.id as video_id,
  date_trunc('day', l.created_at) as d,
  count(l.video_id) as likes
FROM videos v
LEFT JOIN likes l ON l.video_id = v.id
GROUP BY v.id, date_trunc('day', l.created_at);
```

---

## 🔐 Row Level Security (RLS)

### likes テーブル
```sql
ALTER TABLE public.likes ENABLE ROW LEVEL SECURITY;

-- 自分のライクのみ参照可能
CREATE POLICY "select own likes"
ON public.likes FOR SELECT TO authenticated
USING (auth.uid() = user_id);

-- 自分のライクのみ作成可能
CREATE POLICY "insert own likes"
ON public.likes FOR INSERT TO authenticated
WITH CHECK (auth.uid() = user_id);

-- 自分のライクのみ削除可能
CREATE POLICY "delete own likes"
ON public.likes FOR DELETE TO authenticated
USING (auth.uid() = user_id);
```

### videos テーブル
```sql
ALTER TABLE public.videos ENABLE ROW LEVEL SECURITY;

-- 全ユーザーが動画閲覧可能
CREATE POLICY "read videos"
ON public.videos FOR SELECT
TO anon, authenticated
USING (true);
```

### video_embeddings テーブル
```sql
ALTER TABLE public.video_embeddings ENABLE ROW LEVEL SECURITY;

-- 全ユーザーが埋め込み閲覧可能
CREATE POLICY "read embeddings"
ON public.video_embeddings FOR SELECT
TO anon, authenticated
USING (true);
```

### user_embeddings テーブル
```sql
ALTER TABLE public.user_embeddings ENABLE ROW LEVEL SECURITY;

-- 自分の埋め込みのみ参照可能
CREATE POLICY "read own embedding"
ON public.user_embeddings FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- 自分の埋め込みのみ更新可能
CREATE POLICY "update own embedding"
ON public.user_embeddings FOR UPDATE
TO authenticated
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);
```

---

## 📈 インデックス戦略

### 基本インデックス
```sql
-- 動画検索用
CREATE INDEX idx_videos_external_id ON videos(external_id);
CREATE INDEX idx_videos_source ON videos(source);
CREATE INDEX idx_videos_published_at ON videos(published_at DESC);

-- ユーザーライク検索用
CREATE INDEX idx_likes_user_id ON likes(user_id);
CREATE INDEX idx_likes_created_at ON likes(created_at DESC);

-- 埋め込み検索用
CREATE INDEX idx_video_embeddings_updated_at ON video_embeddings(updated_at DESC);
CREATE INDEX idx_user_embeddings_updated_at ON user_embeddings(updated_at DESC);
```

### ベクトル類似検索用
```sql
-- コサイン類似度検索用インデックス
CREATE INDEX idx_video_embeddings_cosine 
ON video_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_user_embeddings_cosine 
ON user_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

**文書管理**  
**最終更新**: 2025年9月5日  
**管理者**: Claude Code