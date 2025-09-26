# Adult Matching Backend Architecture

## 📋 概要

Adult Matchingバックエンドの完全リファクタリング版のアーキテクチャ詳細仕様書です。このドキュメントでは、システムの設計思想、コンポーネント構成、データフロー、技術決定の根拠について詳細に説明します。

## 🎯 アーキテクチャ目標

### 主要目標
1. **スケーラビリティ**: 10万ユーザー、100万動画に対応
2. **パフォーマンス**: 推薦API <500ms、MLトレーニング <2時間
3. **保守性**: コード重複削減、統一パターン適用
4. **信頼性**: 99.9%稼働率、自動回復機能
5. **開発効率**: CI/CD自動化、包括的テスト

### 非機能要件
- **可用性**: 99.9%稼働率
- **スループット**: 100+ req/s
- **レイテンシ**: P95 < 500ms
- **データ整合性**: ACID準拠
- **セキュリティ**: 認証・認可・暗号化

## 🏗️ 全体アーキテクチャ

### システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                            │
├─────────────────────────────────────────────────────────────────┤
│ Frontend (Next.js) │ Mobile App │ Admin Dashboard │ External API │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                      API Gateway                                │
├─────────────────────────────────────────────────────────────────┤
│              Supabase Edge Functions                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │Recommendations││User Management││  Content   ││   _shared   ││
│  │              │ │              ││   Feed     ││ Utilities   ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                  Backend Services                               │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │     ML      │ │    Data     │ │ Monitoring  │ │Optimization │ │
│ │ Pipeline    │ │ Management  │ │ & Logging   │ │ & Caching   │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                 Infrastructure Layer                            │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ PostgreSQL  │ │ Prometheus  │ │   Docker    │ │   CI/CD     │ │
│ │ + pgvector  │ │ Monitoring  │ │ Containers  │ │ Pipeline    │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 API Layer - Supabase Edge Functions

### 設計原則
- **統合優先**: 関連機能のグループ化
- **共有リソース**: 重複コード削減
- **型安全性**: TypeScript厳密型定義
- **エラーハンドリング**: 統一エラー処理

### コンポーネント構成

#### 1. Recommendations (推薦システム)
```
supabase/functions/recommendations/
├── enhanced_two_tower/
│   ├── index.ts              # 統合推薦API
│   ├── types.ts              # 型定義
│   └── algorithms/           # 推薦アルゴリズム
│       ├── two_tower.ts      # Two-Tower実装
│       ├── collaborative.ts  # 協調フィルタリング
│       └── content_based.ts  # コンテンツベース
```

**機能:**
- 768次元Two-Towerモデル推薦
- ハイブリッド推薦アルゴリズム
- リアルタイム埋め込み更新
- A/Bテスト対応

**API仕様:**
```typescript
interface RecommendationRequest {
  user_id: string;
  num_recommendations: number;
  filters?: {
    genres?: string[];
    makers?: string[];
    exclude_liked?: boolean;
  };
  algorithm?: 'two_tower' | 'collaborative' | 'hybrid';
}

interface RecommendationResponse {
  recommendations: VideoRecommendation[];
  metadata: {
    algorithm_used: string;
    response_time_ms: number;
    cache_hit: boolean;
  };
}
```

#### 2. User Management (ユーザー管理)
```
supabase/functions/user-management/
├── likes/
│   └── index.ts              # いいね管理
├── embeddings/
│   └── index.ts              # 埋め込み更新
├── account/
│   └── index.ts              # アカウント管理
└── shared/
    ├── auth.ts               # 認証ヘルパー
    └── validation.ts         # 入力検証
```

**機能:**
- いいね/スキップ記録
- ユーザー埋め込み更新
- アカウント削除・管理
- プライバシー設定

#### 3. Content (コンテンツ配信)
```
supabase/functions/content/
├── feed/
│   └── index.ts              # 統合フィード
└── search/
    └── index.ts              # 検索機能
```

**フィードタイプ:**
- `explore`: 多様性重視フィード
- `personalized`: パーソナライズドフィード
- `latest`: 最新動画フィード
- `popular`: 人気動画フィード
- `random`: ランダムフィード

#### 4. Shared Utilities (共有ユーティリティ)
```
supabase/functions/_shared/
├── database/
│   ├── connection.ts         # DB接続管理
│   ├── queries.ts            # 共通クエリ
│   └── transactions.ts       # トランザクション管理
├── auth/
│   ├── middleware.ts         # 認証ミドルウェア
│   └── permissions.ts        # 権限管理
├── validation/
│   ├── schemas.ts            # 入力スキーマ
│   └── sanitization.ts       # データサニタイゼーション
├── monitoring/
│   └── logger.ts             # 構造化ログ
└── utils/
    ├── cache.ts              # キャッシュユーティリティ
    ├── metrics.ts            # メトリクス収集
    └── errors.ts             # エラーハンドリング
```

## 🔧 Backend Services Layer

### 1. Machine Learning Pipeline (backend/ml/)

#### アーキテクチャ
```
backend/ml/
├── models/                   # モデル管理
│   ├── two_tower/           # Two-Towerモデル
│   ├── registry.py          # モデルレジストリ
│   └── versioning.py        # バージョン管理
├── training/                # 訓練システム
│   ├── trainers/            # 訓練クラス
│   ├── configs/             # 訓練設定
│   ├── schedules/           # スケジュール管理
│   └── optimization/        # 訓練最適化
├── preprocessing/           # 前処理システム
│   ├── features/            # 特徴量エンジニアリング
│   ├── embeddings/          # 埋め込み管理
│   └── pipelines/           # 前処理パイプライン
├── inference/               # 推論システム
│   ├── serving/             # モデルサービング
│   ├── batch/               # バッチ推論
│   └── realtime/            # リアルタイム推論
└── evaluation/              # 評価システム
    ├── metrics/             # 評価メトリクス
    ├── experiments/         # 実験管理
    └── reports/             # 評価レポート
```

#### 技術スタック
- **フレームワーク**: TensorFlow 2.16+
- **分散訓練**: tf.distribute.Strategy
- **モデル最適化**: 混合精度、グラディエントチェックポイント
- **デプロイメント**: TensorFlow.js変換、ONNX対応

#### パフォーマンス最適化
```python
@dataclass
class TrainingOptimization:
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    xla_compilation: bool = True
    data_parallel: bool = True
    memory_mapping: bool = True
    batch_size_optimization: bool = True
```

### 2. Data Management System (backend/data/)

#### アーキテクチャ
```
backend/data/
├── sync/                    # API同期
│   ├── dmm/                 # DMM API統合
│   ├── schedulers/          # 同期スケジューラ
│   └── validators/          # データ検証
├── processing/              # データ処理
│   ├── pipelines/           # ETLパイプライン
│   ├── transformers/        # データ変換
│   └── aggregators/         # データ集約
├── storage/                 # ストレージ管理
│   ├── managers/            # ストレージマネージャ
│   ├── compression/         # データ圧縮
│   └── archival/            # アーカイブ管理
└── quality/                 # データ品質管理
    ├── validation/          # 品質検証
    ├── monitoring/          # 品質監視
    └── remediation/         # 品質修復
```

#### データフロー
```
External APIs → Ingestion → Validation → Processing → Storage
      ↓              ↓           ↓            ↓         ↓
  Rate Limit    Schema Check   ETL Pipeline  PostgreSQL Metrics
  Handling      Data Quality   Parallel      + pgvector Collection
```

### 3. Monitoring & Logging (backend/monitoring/)

#### 監視レイヤー
```
System Monitor (SystemMonitor)
├── CPU/Memory/Disk監視
├── プロセス監視
└── リソース使用率追跡

Integration Monitor (IntegrationMonitor)
├── Edge Functions ↔ Backend統合監視
├── MLパイプライン状態監視
└── データフロー健全性監視

Edge Function Logger (EdgeFunctionLogger)
├── 構造化ログ
├── パフォーマンスメトリクス
└── エラー追跡
```

#### メトリクス体系
```
System Metrics:
- adult_matching_system_cpu_percent
- adult_matching_system_memory_percent
- adult_matching_db_connections

Application Metrics:
- adult_matching_requests_total
- adult_matching_request_duration_seconds
- adult_matching_ml_model_accuracy

Edge Function Metrics:
- adult_matching_edge_requests_total
- adult_matching_edge_errors_total
- adult_matching_edge_*_duration_ms
```

### 4. Performance Optimization (backend/optimization/)

#### 最適化コンポーネント
```
RecommendationOptimizer:
├── キャッシュ最適化 (LRU, TTL)
├── バッチ処理最適化
├── 並列処理最適化
└── レスポンス時間最適化

MLTrainingOptimizer:
├── データローディング最適化
├── バッチ処理最適化
├── メモリ使用量最適化
└── 並列処理最適化

DatabaseOptimizer:
├── ベクター検索最適化
├── インデックス最適化
├── 接続プール最適化
└── クエリ最適化

PerformanceVerifier:
├── ベンチマークテスト
├── パフォーマンス検証
├── レポート生成
└── 継続的監視
```

## 💾 データ層設計

### データベーススキーマ

#### 主要テーブル
```sql
-- 動画メタデータ
CREATE TABLE videos (
    id UUID PRIMARY KEY,
    external_id VARCHAR UNIQUE,
    title TEXT NOT NULL,
    genre VARCHAR[],
    maker VARCHAR,
    embedding vector(768),  -- pgvector
    rating DECIMAL,
    view_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ユーザー埋め込み
CREATE TABLE user_embeddings (
    user_id UUID PRIMARY KEY,
    embedding vector(768),  -- pgvector
    preferences JSONB,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- いいね記録
CREATE TABLE likes (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    video_id UUID REFERENCES videos(id),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, video_id)
);
```

#### インデックス戦略
```sql
-- ベクター検索最適化
CREATE INDEX idx_videos_embedding_cosine 
ON videos USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

CREATE INDEX idx_user_embeddings_embedding_cosine 
ON user_embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 50);

-- 通常検索最適化
CREATE INDEX idx_videos_genre_rating ON videos(genre, rating DESC);
CREATE INDEX idx_likes_user_video ON likes(user_id, video_id);
CREATE INDEX idx_videos_view_count ON videos(view_count DESC);
```

### Row Level Security (RLS)

#### ユーザーデータ保護
```sql
-- ユーザーは自分のいいねのみアクセス可能
CREATE POLICY "Users can only access their own likes" 
ON likes FOR ALL 
USING (auth.uid() = user_id);

-- ユーザーは自分の埋め込みのみアクセス可能
CREATE POLICY "Users can only access their own embeddings" 
ON user_embeddings FOR ALL 
USING (auth.uid() = user_id);

-- 動画は全ユーザー読み取り可能
CREATE POLICY "Videos are readable by all authenticated users" 
ON videos FOR SELECT 
USING (auth.role() = 'authenticated');
```

## 🔄 データフロー

### 推薦生成フロー
```
1. User Request → Edge Function
2. Authentication & Validation
3. User Embedding Retrieval
4. Vector Search Optimization
5. Candidate Generation
6. Ranking & Filtering
7. Response Caching
8. Metrics Collection
9. Response Return
```

### MLトレーニングフロー
```
1. Data Collection & Validation
2. Feature Engineering
3. Training Data Preparation
4. Model Training (Optimized)
5. Model Evaluation
6. Model Registration
7. Deployment (TensorFlow.js)
8. Performance Monitoring
```

### データ同期フロー
```
1. Scheduled API Calls
2. Rate Limiting
3. Data Validation
4. Duplicate Detection
5. ETL Processing
6. Database Storage
7. Quality Monitoring
8. Metrics Collection
```

## 🛡️ セキュリティアーキテクチャ

### 認証・認可
- **Edge Functions**: Supabase Auth JWT検証
- **Backend Services**: サービス間認証
- **Database**: RLS + API Key認証
- **External APIs**: API Key管理

### データ保護
- **暗号化**: TLS 1.3通信、データベース暗号化
- **アクセス制御**: 最小権限原則
- **データマスキング**: ログ・メトリクスでのPII除外
- **監査ログ**: 全アクセスログ記録

### 脆弱性対策
- **入力検証**: 全APIエンドポイント
- **SQLインジェクション**: パラメータ化クエリ
- **XSS**: Content Security Policy
- **CSRF**: SameSite Cookie設定

## 📊 パフォーマンス設計

### スケーラビリティ戦略

#### 水平スケーリング
- **Edge Functions**: Supabase自動スケーリング
- **Database**: Read Replica + Connection Pooling
- **ML Training**: 分散訓練対応
- **Monitoring**: Prometheus Federation

#### キャッシュ戦略
```
L1: Edge Function Memory Cache (60s TTL)
L2: PostgreSQL Query Cache
L3: CDN Cache (Static Assets)
L4: Browser Cache (Long-term Assets)
```

#### 最適化技術
- **データベース**: pgvector最適化、インデックス戦略
- **API**: バッチリクエスト、並列処理
- **ML**: 混合精度、グラディエントチェックポイント
- **ネットワーク**: 圧縮、CDN活用

## 🔧 開発・運用

### CI/CD パイプライン
```
GitHub Actions:
├── Code Quality Checks
│   ├── TypeScript/Python型チェック
│   ├── ESLint/Flake8リンター
│   └── Security Scan (CodeQL)
├── Testing Pipeline
│   ├── Unit Tests (Jest/pytest)
│   ├── Integration Tests
│   └── E2E Tests (Playwright)
├── Performance Testing
│   ├── Load Testing
│   ├── Performance Benchmarks
│   └── Resource Usage Analysis
└── Deployment Pipeline
    ├── Edge Functions Deploy
    ├── Database Migrations
    └── Infrastructure Updates
```

### 監視・アラート
```
Monitoring Stack:
├── Prometheus (メトリクス収集)
├── Grafana (ダッシュボード)
├── AlertManager (アラート)
└── Custom Monitors (業務メトリクス)

Alert Rules:
├── System: CPU > 85%, Memory > 90%
├── API: Response Time > 2s, Error Rate > 5%
├── ML: Training Failure, Model Accuracy Drop
└── Integration: Edge Function ↔ Backend Communication Failure
```

### デプロイメント戦略
- **Blue-Green**: ゼロダウンタイムデプロイ
- **Canary**: 段階的リリース
- **Feature Flags**: 機能切り替え
- **Rollback**: 自動ロールバック機能

## 📈 未来拡張性

### 予定される拡張
1. **マルチモーダル推薦**: 画像・音声・テキスト統合
2. **リアルタイム学習**: オンライン学習対応
3. **マルチテナント**: 複数サービス対応
4. **国際化**: 多言語・多地域対応

### アーキテクチャ進化計画
- **マイクロサービス**: 更なる分離
- **イベント駆動**: 非同期処理拡充
- **サーバーレス**: 完全サーバーレス化
- **AI/ML**: GPT統合、高度AI機能

---

## 📚 関連ドキュメント

- **[API仕様書](api-specification.md)** - 詳細API仕様
- **[データベース設計](database-design.md)** - DB詳細設計
- **[運用手順書](operations-guide.md)** - 運用・保守手順
- **[セキュリティガイド](security-guide.md)** - セキュリティ仕様

---

**文書バージョン**: 2.0  
**最終更新**: 2025年9月16日  
**作成者**: Claude Code Assistant  
**レビュー**: Backend Architecture Team