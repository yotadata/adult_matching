# データ・モデル技術設計書

**プロジェクト**: アダルト動画マッチングアプリケーション  
**文書バージョン**: v1.0  
**作成日**: 2025年9月3日  
**最終更新**: 2025年9月3日

---

## 📋 目次

1. [アーキテクチャ概要](#アーキテクチャ概要)
2. [データアーキテクチャ](#データアーキテクチャ)
3. [MLパイプライン設計](#mlパイプライン設計)
4. [API設計](#api設計)
5. [インフラストラクチャ](#インフラストラクチャ)
6. [セキュリティ設計](#セキュリティ設計)
7. [性能・スケーラビリティ](#性能スケーラビリティ)

---

## 🏗️ アーキテクチャ概要

### システム全体構成

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Supabase      │    │   ML Pipeline   │
│   (Next.js)     │◄──►│   (PostgreSQL)  │◄──►│   (Python)      │
│                 │    │   Edge Functions│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └─────────────►│  External APIs  │◄─────────────┘
                        │  (DMM/FANZA)    │
                        └─────────────────┘
```

### アーキテクチャ原則

1. **モジュラー設計**: データ処理・ML・フロントエンドの明確な分離
2. **スケーラビリティ**: 水平スケーリング可能な設計
3. **リアルタイム性**: ユーザーアクションへの即座な応答
4. **データドリブン**: 全意思決定をデータと指標に基づいて実行

---

## 🗄️ データアーキテクチャ

### データフロー概要

```
External Sources → Raw Data → Processing → Features → ML Models → Recommendations → Users
     ↓              ↓           ↓           ↓          ↓             ↓            ↓
  DMM/FANZA     raw_data/   data_cleaner  embeddings  two_tower   edge_functions frontend
  レビュー       JSON/CSV      .py         vectors     model        API         UI
```

### データベーススキーマ設計

#### Core Tables

```sql
-- ユーザーテーブル（匿名化）
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    age_verified BOOLEAN NOT NULL DEFAULT FALSE,
    preferences JSONB DEFAULT '{}', -- 初期設定情報
    status TEXT DEFAULT 'active' -- active, inactive, suspended
);

-- 動画マスターテーブル
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id TEXT UNIQUE NOT NULL, -- DMM/FANZA ID
    title TEXT NOT NULL,
    description TEXT,
    duration INTEGER, -- 秒
    release_date DATE,
    thumbnail_url TEXT,
    source_url TEXT, -- 外部サイトURL
    metadata JSONB DEFAULT '{}', -- 拡張メタデータ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- パフォーマー（出演者）テーブル
CREATE TABLE performers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    metadata JSONB DEFAULT '{}' -- 年齢、身長等
);

-- 動画-パフォーマー関係テーブル
CREATE TABLE video_performers (
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    performer_id UUID REFERENCES performers(id) ON DELETE CASCADE,
    role TEXT DEFAULT 'performer', -- main, supporting, etc.
    PRIMARY KEY (video_id, performer_id)
);

-- ジャンル・タグテーブル
CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL, -- genre, tag, studio, series
    parent_id UUID REFERENCES tags(id), -- 階層構造サポート
    metadata JSONB DEFAULT '{}'
);

-- 動画-タグ関係テーブル
CREATE TABLE video_tags (
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    weight FLOAT DEFAULT 1.0, -- タグの重要度
    PRIMARY KEY (video_id, tag_id)
);
```

#### ML & Analytics Tables

```sql
-- ユーザー行動データ
CREATE TABLE user_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    action TEXT NOT NULL, -- 'like', 'skip', 'view'
    session_id UUID NOT NULL,
    context JSONB DEFAULT '{}', -- 推薦理由、UI位置等
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ユーザー埋め込みベクトル
CREATE TABLE user_embeddings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    embedding VECTOR(768), -- pgvector拡張使用
    confidence_score FLOAT DEFAULT 0.0,
    version TEXT DEFAULT 'v1',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 動画埋め込みベクトル
CREATE TABLE video_embeddings (
    video_id UUID PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
    content_embedding VECTOR(768), -- テキストベース埋め込み
    metadata_embedding VECTOR(256), -- メタデータベース埋め込み
    version TEXT DEFAULT 'v1',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- レビューデータ
CREATE TABLE video_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    external_reviewer_id TEXT, -- 外部サイトのレビュアーID
    review_text TEXT NOT NULL,
    rating FLOAT, -- 1-5スケール
    sentiment_score FLOAT, -- -1 to 1
    helpful_votes INTEGER DEFAULT 0,
    source TEXT NOT NULL, -- 'dmm', 'fanza'
    external_url TEXT,
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Indexes & Performance

```sql
-- パフォーマンス最適化用インデックス
CREATE INDEX idx_user_decisions_user_time ON user_decisions(user_id, created_at DESC);
CREATE INDEX idx_user_decisions_video ON user_decisions(video_id);
CREATE INDEX idx_videos_release_date ON videos(release_date DESC);
CREATE INDEX idx_video_tags_tag ON video_tags(tag_id);
CREATE INDEX idx_video_reviews_video_rating ON video_reviews(video_id, rating DESC);

-- ベクトル類似度検索用インデックス（pgvector）
CREATE INDEX idx_user_embeddings_vector ON user_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_video_embeddings_content ON video_embeddings USING ivfflat (content_embedding vector_cosine_ops);
```

### データ処理パイプライン

#### 1. データ収集層 (Data Collection Layer)

```python
# data_processing/scraping/
├── cookie_dmm_scraper.py      # DMM レビュー収集
├── fanza_api_client.py        # FANZA API データ収集  
├── scheduler.py               # 定期実行スケジューラー
└── data_validator.py          # データ品質検証
```

#### 2. データ処理層 (Data Processing Layer)

```python
# data_processing/utils/
├── data_cleaner.py           # テキスト正規化・クリーニング
├── feature_extractor.py      # 特徴量エンジニアリング
├── sentiment_analyzer.py     # 感情分析
└── deduplicator.py           # 重複データ除去
```

#### 3. 埋め込み生成層 (Embedding Generation Layer)

```python
# ml_pipeline/preprocessing/
├── text_embedder.py          # BERT日本語モデル埋め込み
├── metadata_embedder.py      # カテゴリカル特徴量埋め込み
├── user_profiler.py          # ユーザープロファイル生成
└── batch_processor.py        # バッチ処理管理
```

---

## 🤖 MLパイプライン設計

### Two-Tower アーキテクチャ

```python
# ml_pipeline/models/two_tower.py

class TwoTowerModel(tf.keras.Model):
    def __init__(self, user_vocab_size, item_vocab_size, embedding_dim=768):
        super().__init__()
        
        # User Tower
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.user_dense_1 = tf.keras.layers.Dense(512, activation='relu')
        self.user_dense_2 = tf.keras.layers.Dense(256, activation='relu')
        self.user_output = tf.keras.layers.Dense(embedding_dim, activation='tanh')
        
        # Item Tower  
        self.item_embedding = tf.keras.layers.Embedding(item_vocab_size, embedding_dim)
        self.item_dense_1 = tf.keras.layers.Dense(512, activation='relu')
        self.item_dense_2 = tf.keras.layers.Dense(256, activation='relu')
        self.item_output = tf.keras.layers.Dense(embedding_dim, activation='tanh')
        
        # Interaction Layer
        self.interaction_layer = tf.keras.layers.Dot(axes=1)
        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        user_features, item_features = inputs
        
        # User tower forward pass
        user_emb = self.user_embedding(user_features)
        user_hidden = self.user_dense_1(user_emb)
        user_hidden = self.user_dense_2(user_hidden)
        user_repr = self.user_output(user_hidden)
        
        # Item tower forward pass
        item_emb = self.item_embedding(item_features)
        item_hidden = self.item_dense_1(item_emb)
        item_hidden = self.item_dense_2(item_hidden)
        item_repr = self.item_output(item_hidden)
        
        # Interaction and prediction
        interaction = self.interaction_layer([user_repr, item_repr])
        prediction = self.final_dense(interaction)
        
        return prediction
```

### 特徴量設計

#### User Features
```python
user_features = {
    'user_id': 'categorical',
    'age_group': 'categorical',  # 18-24, 25-34, 35-44, 45+
    'liked_genres': 'multi_categorical',
    'liked_performers': 'multi_categorical', 
    'avg_session_length': 'numerical',
    'like_rate': 'numerical',
    'time_of_day_preference': 'categorical',
    'behavioral_embedding': 'dense_vector'  # 768-dim
}
```

#### Item Features
```python
item_features = {
    'video_id': 'categorical',
    'duration_bucket': 'categorical',  # short, medium, long
    'release_year': 'categorical',
    'genres': 'multi_categorical',
    'performers': 'multi_categorical',
    'rating_score': 'numerical',
    'view_count_tier': 'categorical',
    'content_embedding': 'dense_vector',  # 768-dim
    'metadata_embedding': 'dense_vector'  # 256-dim
}
```

### 訓練パイプライン

```python
# ml_pipeline/training/training_pipeline.py

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.model = TwoTowerModel(config)
        self.evaluator = ModelEvaluator(config)
    
    def run_training(self):
        # 1. データ準備
        train_data, val_data, test_data = self.data_loader.load_datasets()
        
        # 2. 特徴量エンジニアリング
        train_features = self.preprocess_features(train_data)
        val_features = self.preprocess_features(val_data)
        
        # 3. モデル訓練
        history = self.model.fit(
            train_features,
            epochs=self.config.epochs,
            validation_data=val_features,
            callbacks=[
                EarlyStopping(patience=5),
                ModelCheckpoint('best_model.h5'),
                TensorBoard(log_dir='./logs')
            ]
        )
        
        # 4. モデル評価
        test_metrics = self.evaluator.evaluate(test_data)
        
        # 5. モデル保存・デプロイ
        self.save_model_artifacts()
        self.deploy_to_supabase()
        
        return history, test_metrics
```

### リアルタイム推薦システム

```python
# supabase/functions/recommendations/index.ts

export default async function handler(req: Request) {
    const { user_id, limit = 20 } = await req.json();
    
    // 1. ユーザー埋め込み取得
    const userEmbedding = await getUserEmbedding(user_id);
    
    // 2. 候補動画のフィルタリング
    const candidateVideos = await getCandidateVideos(user_id);
    
    // 3. Two-Towerモデルでスコア計算
    const scoredVideos = await scoreVideos(userEmbedding, candidateVideos);
    
    // 4. 多様性を考慮した再ランキング
    const diversifiedVideos = await diversifyRecommendations(scoredVideos);
    
    // 5. 結果返却
    return new Response(JSON.stringify({
        recommendations: diversifiedVideos.slice(0, limit),
        metadata: {
            user_id,
            generated_at: new Date().toISOString(),
            model_version: 'v1.0'
        }
    }));
}
```

---

## 🔌 API設計

### RESTful API エンドポイント

#### 推薦API
```typescript
// GET /api/recommendations
interface RecommendationRequest {
    user_id: string;
    limit?: number;          // default: 20
    offset?: number;         // default: 0
    diversify?: boolean;     // default: true
    exclude_seen?: boolean;  // default: true
}

interface RecommendationResponse {
    recommendations: VideoRecommendation[];
    metadata: {
        user_id: string;
        total_count: number;
        generated_at: string;
        model_version: string;
    };
}

interface VideoRecommendation {
    video_id: string;
    score: number;           // 0-1
    reason: string;          // 推薦理由
    video: VideoMetadata;
}
```

#### ユーザー行動記録API
```typescript
// POST /api/user-actions
interface UserActionRequest {
    user_id: string;
    video_id: string;
    action: 'like' | 'skip' | 'view';
    session_id: string;
    context?: {
        position: number;    // 推薦リスト内の位置
        source: string;      // 推薦元
        timestamp: string;
    };
}

interface UserActionResponse {
    success: boolean;
    action_id: string;
    updated_recommendations?: VideoRecommendation[];
}
```

#### フィード取得API
```typescript
// GET /api/feed
interface FeedRequest {
    user_id?: string;        // 未ログインの場合はnull
    feed_type: 'explore' | 'personalized' | 'trending';
    limit?: number;
    cursor?: string;         // ページネーション用
}

interface FeedResponse {
    videos: VideoWithMetadata[];
    next_cursor?: string;
    has_more: boolean;
}
```

### GraphQL Schema (将来拡張用)

```graphql
type User {
  id: ID!
  preferences: JSON
  recommendations(limit: Int = 20): [VideoRecommendation!]!
  likedVideos(limit: Int = 50): [Video!]!
}

type Video {
  id: ID!
  title: String!
  description: String
  duration: Int
  releaseDate: Date
  thumbnailUrl: String
  performers: [Performer!]!
  tags: [Tag!]!
  rating: Float
}

type VideoRecommendation {
  video: Video!
  score: Float!
  reason: String!
  rank: Int!
}

type Query {
  getRecommendations(userId: ID!, limit: Int = 20): [VideoRecommendation!]!
  getVideo(id: ID!): Video
  searchVideos(query: String!, limit: Int = 20): [Video!]!
}

type Mutation {
  recordUserAction(input: UserActionInput!): UserActionResponse!
  updateUserPreferences(userId: ID!, preferences: JSON!): User!
}
```

---

## 🏗️ インフラストラクチャ

### Supabase アーキテクチャ

```yaml
# supabase/config.toml
[database]
pooler_mode = "session"
default_pool_size = 20
max_pool_size = 100

[auth]
site_url = "https://your-app.com"
redirect_urls = ["https://your-app.com/**"]

[storage]
file_size_limit = "50MB"
```

### Edge Functions 構成

```
supabase/functions/
├── recommendations/         # メイン推薦API
│   ├── index.ts
│   ├── model_inference.ts
│   └── candidate_generation.ts
├── user_actions/           # ユーザー行動記録
│   ├── index.ts
│   └── embedding_update.ts
├── feed_explore/           # 探索フィード
│   └── index.ts
└── _shared/               # 共通ユーティリティ
    ├── database.ts
    ├── ml_utils.ts
    └── types.ts
```

### データベース設定

```sql
-- パフォーマンスチューニング
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements,pg_hint_plan';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = 1.1;

-- pgvector拡張有効化
CREATE EXTENSION IF NOT EXISTS vector;

-- RLS (Row Level Security) 設定
ALTER TABLE user_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_own_data ON user_decisions
    FOR ALL USING (auth.uid() = user_id);
```

### 開発・デプロイメント環境

```yaml
# .github/workflows/deploy-ml.yml
name: Deploy ML Pipeline
on:
  push:
    branches: [main]
    paths: ['ml_pipeline/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      
      - name: Run tests
        run: |
          uv run pytest ml_pipeline/tests/
      
      - name: Train and deploy model
        run: |
          uv run python ml_pipeline/training/train_two_tower_model.py
          # Supabase関数更新処理
```

---

## 🔐 セキュリティ設計

### 認証・認可

```typescript
// 年齢認証ミドルウェア
export async function ageVerificationMiddleware(req: Request) {
    const ageVerified = req.headers.get('age-verified');
    const userAgent = req.headers.get('user-agent');
    
    if (!ageVerified || ageVerified !== 'true') {
        return new Response('Age verification required', { status: 403 });
    }
    
    // 追加のbot検出ロジック
    if (detectBot(userAgent)) {
        return new Response('Access denied', { status: 403 });
    }
}
```

### データ匿名化

```python
# data_processing/utils/anonymizer.py
class UserAnonymizer:
    def __init__(self):
        self.salt = os.environ.get('ANONYMIZATION_SALT')
    
    def anonymize_user_id(self, user_id: str) -> str:
        """ユーザーIDを匿名化"""
        return hashlib.sha256(f"{user_id}{self.salt}".encode()).hexdigest()
    
    def anonymize_behavioral_data(self, data: dict) -> dict:
        """行動データから個人特定可能情報を除去"""
        return {
            'action': data['action'],
            'video_category': data.get('video_category'),
            'session_length': min(data.get('session_length', 0), 3600),  # cap at 1 hour
            'timestamp_hour': datetime.fromisoformat(data['timestamp']).hour
        }
```

### データ暗号化

```sql
-- 機密データの暗号化
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 個人設定の暗号化保存
CREATE OR REPLACE FUNCTION encrypt_user_preferences(data JSONB)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_encrypt(data::TEXT, current_setting('app.encryption_key'));
END;
$$ LANGUAGE plpgsql;
```

---

## ⚡ 性能・スケーラビリティ

### キャッシング戦略

```typescript
// Redis キャッシュ設計
interface CacheStrategy {
    user_recommendations: {
        key: `user:${user_id}:recommendations`;
        ttl: 3600; // 1時間
        refresh_on_action: true;
    };
    
    video_metadata: {
        key: `video:${video_id}:metadata`;
        ttl: 86400; // 24時間
        refresh_on_update: true;
    };
    
    trending_videos: {
        key: 'trending:videos';
        ttl: 1800; // 30分
        background_refresh: true;
    };
}
```

### データベース最適化

```sql
-- パーティショニング設定
CREATE TABLE user_decisions_partitioned (LIKE user_decisions INCLUDING ALL)
PARTITION BY RANGE (created_at);

-- 月次パーティション作成
CREATE TABLE user_decisions_2025_01 PARTITION OF user_decisions_partitioned
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- 古いデータのアーカイブ処理
CREATE OR REPLACE FUNCTION archive_old_decisions()
RETURNS void AS $$
BEGIN
    -- 6ヶ月より古いデータを別テーブルに移動
    INSERT INTO user_decisions_archive 
    SELECT * FROM user_decisions 
    WHERE created_at < NOW() - INTERVAL '6 months';
    
    DELETE FROM user_decisions 
    WHERE created_at < NOW() - INTERVAL '6 months';
END;
$$ LANGUAGE plpgsql;
```

### モニタリング・アラート

```yaml
# monitoring/alerts.yml
alerts:
  - name: High Recommendation Latency
    condition: avg(recommendation_response_time) > 500ms
    duration: 5m
    action: notify_slack
  
  - name: Low Model Accuracy
    condition: recommendation_ctr < 0.6
    duration: 1h
    action: retrain_model
  
  - name: Database Connection Pool Full
    condition: db_connections > 80
    duration: 2m
    action: scale_connections
```

---

## 📊 メトリクス・ログ設計

### アプリケーションメトリクス

```typescript
// metrics/collectors.ts
export class MetricsCollector {
    trackRecommendationServed(user_id: string, video_ids: string[]) {
        this.increment('recommendations.served', {
            user_segment: getUserSegment(user_id),
            recommendation_count: video_ids.length
        });
    }
    
    trackUserAction(action: UserAction) {
        this.increment('user_actions.total', {
            action_type: action.action,
            video_category: action.video_category,
            session_duration: action.session_duration
        });
    }
    
    trackModelPerformance(predictions: Prediction[], actual: UserAction[]) {
        const accuracy = calculateAccuracy(predictions, actual);
        this.gauge('model.accuracy', accuracy);
    }
}
```

### ログ構造化

```json
{
  "timestamp": "2025-09-03T10:30:00Z",
  "level": "INFO",
  "service": "recommendations",
  "user_id": "hashed_user_123",
  "session_id": "session_456",
  "event": "recommendation_generated",
  "data": {
    "video_count": 20,
    "model_version": "v1.0",
    "latency_ms": 245,
    "cache_hit": true
  },
  "trace_id": "abc123def456"
}
```

---

**文書管理**  
**作成者**: Claude Code  
**承認者**: -  
**次回レビュー予定**: 2025年10月