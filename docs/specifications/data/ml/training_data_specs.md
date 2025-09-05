# ML学習データ仕様書

Adult Matching アプリケーションの機械学習パイプライン用データ仕様

---

## 📊 学習データ概要

### Two-Tower モデル用データ
- **ユーザータワー**: 疑似ユーザー特徴量（レビューデータ由来）
- **アイテムタワー**: 動画特徴量（**API取得データ由来**）  
- **相互作用**: Like/Skip ラベル（評価ベース変換）

### データソース
```
[DMM/FANZA API] → [動画メタデータ] → [PostgreSQL] → [アイテム特徴量]
                                                        ↓
[スクレイピングレビュー] → [Content ID紐づけ] → [疑似ユーザー生成] → [Two-Tower学習]
     38,904件                 リンキング              50ユーザー           モデル
```

### 🎯 **重要なデータポリシー**
- **動画特徴量**: API取得データ（title, genre, maker等）のみ使用
- **レビューデータ**: Content ID紐づけ + ユーザー行動変換のみ
- **学習対象動画**: PostgreSQL videosテーブル内の動画のみ

---

## 🎯 疑似ユーザー生成仕様

### 評価ベース変換ルール
```python
# Pattern 1: 評価ベース変換
if rating >= 4.0:
    action = "Like"
elif rating <= 3.0:
    action = "Skip"
else:
    action = None  # 中間評価は除外
```

### 生成統計 (2025年9月5日時点)
- **総レビュー数**: 38,904 件
- **疑似ユーザー数**: 50 名
- **Like率**: 77.7% (30,192 件)
- **Skip率**: 22.3% (8,712 件)
- **対象動画数**: 32,304 件

---

## 📋 データ形式仕様

### 疑似ユーザーデータ構造
```json
{
  "user_id": "reviewer_12345",
  "original_reviewer_info": {
    "reviewer_id": "12345",
    "total_reviews": 150,
    "avg_rating": 4.2,
    "review_span_days": 365
  },
  "interactions": [
    {
      "content_id": "video_001",
      "action": "Like",
      "original_rating": 5.0,
      "review_text": "素晴らしい作品でした...",
      "timestamp": "2025-01-15T10:30:00Z",
      "helpful_count": 5
    }
  ],
  "user_stats": {
    "total_interactions": 150,
    "like_count": 120,
    "skip_count": 30,
    "like_ratio": 0.8,
    "avg_review_length": 85
  },
  "generated_at": "2025-09-05T12:00:00Z"
}
```

### 学習用特徴量データ
```python
# ユーザー特徴量
user_features = {
    'user_id': int,                    # ユーザーID (エンコード済み)
    'total_interactions': int,         # 総インタラクション数
    'like_ratio': float,              # Like率
    'avg_rating': float,              # 平均評価値
    'review_length_avg': float,       # 平均レビュー文字数
    'activity_days': int,             # 活動日数
    'genre_preferences': List[float]  # ジャンル嗜好ベクトル
}

# アイテム特徴量（**API由来のみ**）
item_features = {
    'content_id': int,                # 動画ID (エンコード済み)
    'genre_vector': List[float],      # ジャンルベクトル（API genre）
    'maker_vector': List[float],      # メーカーベクトル（API maker）
    'price_normalized': float,        # 正規化価格（API price）
    'duration_normalized': float,     # 正規化再生時間（API duration）
    'release_recency': float,        # リリース新しさ（API release_date）
    'director_vector': List[float],   # 監督ベクトル（API director）
    'series_vector': List[float]     # シリーズベクトル（API series）
}
```

---

## 🔄 データ前処理パイプライン

### Step 1: 生データ読み込み
```python
# バッチレビューデータ統合
integrator = BatchDataIntegrator()
reviews = integrator.load_batch_reviews()
clean_reviews = integrator.clean_and_validate_reviews(reviews)
```

### Step 2: 疑似ユーザー生成
```python
# 評価ベース疑似ユーザー生成
generator = RatingBasedPseudoUserGenerator()
pseudo_users = generator.generate_pseudo_users(clean_reviews)
```

### Step 3: 特徴量エンジニアリング
```python
# ユーザー・アイテム特徴量生成
feature_generator = FeatureGenerator()
user_features = feature_generator.generate_user_features(pseudo_users)
# **重要**: アイテム特徴量はPostgreSQL videosテーブルから取得
item_features = feature_generator.generate_item_features_from_db(supabase_client)
```

### Step 4: 学習データセット作成
```python
# TensorFlowデータセット生成
dataset_creator = TwoTowerDatasetCreator()
train_dataset, val_dataset = dataset_creator.create_datasets(
    user_features, item_features, interactions
)
```

---

## 🎯 特徴量設計

### ユーザー特徴量
```python
USER_FEATURES = {
    # 基本統計量
    'total_interactions': {'type': 'int', 'range': [1, 1000]},
    'like_ratio': {'type': 'float', 'range': [0.0, 1.0]},
    'avg_rating': {'type': 'float', 'range': [1.0, 5.0]},
    
    # 行動パターン
    'review_length_avg': {'type': 'float', 'range': [10, 500]},
    'activity_span_days': {'type': 'int', 'range': [1, 365]},
    'helpful_ratio': {'type': 'float', 'range': [0.0, 1.0]},
    
    # 嗜好特徴量
    'genre_preferences': {'type': 'vector', 'dim': 20},
    'rating_variance': {'type': 'float', 'range': [0.0, 4.0]}
}
```

### アイテム特徴量（**API取得データベース**）
```python
ITEM_FEATURES = {
    # API基本メタデータ
    'genre_vector': {'type': 'vector', 'dim': 20, 'source': 'videos.genre'},
    'maker_vector': {'type': 'vector', 'dim': 50, 'source': 'videos.maker'},
    'director_vector': {'type': 'vector', 'dim': 30, 'source': 'videos.director'},
    'series_vector': {'type': 'vector', 'dim': 40, 'source': 'videos.series'},
    
    # API数値特徴量
    'price_normalized': {'type': 'float', 'range': [0.0, 1.0], 'source': 'videos.price'},
    'duration_normalized': {'type': 'float', 'range': [0.0, 1.0], 'source': 'videos.duration_seconds'},
    'release_recency': {'type': 'float', 'range': [0.0, 1.0], 'source': 'videos.product_released_at'},
    
    # データベース関連特徴量
    'tag_count': {'type': 'int', 'range': [0, 50], 'source': 'video_tags JOIN'},
    'performer_count': {'type': 'int', 'range': [0, 20], 'source': 'video_performers JOIN'}
}
```

---

## 🔍 データ品質管理

### 品質チェック項目
```python
QUALITY_CHECKS = {
    # データ完全性
    'missing_values': {'threshold': 0.05},      # 欠損値5%未満
    'duplicate_records': {'threshold': 0.01},   # 重複1%未満
    'schema_compliance': {'threshold': 0.99},   # スキーマ適合99%以上
    
    # 分布チェック
    'like_ratio_range': {'min': 0.6, 'max': 0.8},   # Like率60-80%
    'user_interaction_min': {'threshold': 10},       # 最小インタラクション数
    'genre_coverage': {'min_genres': 15},            # 最小ジャンル数
    
    # 統計的妥当性
    'rating_distribution': {'chi_square_p': 0.05},   # 評価分布の妥当性
    'temporal_consistency': {'max_gap_days': 7}      # 時系列一貫性
}
```

### 検証プロセス
```python
def validate_training_data(dataset):
    validator = DataQualityValidator()
    
    # 基本統計検証
    stats_valid = validator.check_basic_statistics(dataset)
    
    # 分布検証
    distribution_valid = validator.check_distributions(dataset)
    
    # 相関検証
    correlation_valid = validator.check_feature_correlations(dataset)
    
    return all([stats_valid, distribution_valid, correlation_valid])
```

---

## 📈 学習データセット分割

### 分割戦略
```python
DATASET_SPLIT = {
    'train': 0.7,      # 70% 学習用
    'validation': 0.15, # 15% 検証用
    'test': 0.15       # 15% テスト用
}

# 時系列分割（推奨）
# 古いデータ → 学習
# 中間データ → 検証  
# 最新データ → テスト
```

### バッチサイズ・設定
```python
TRAINING_CONFIG = {
    'batch_size': 256,
    'buffer_size': 10000,
    'prefetch_size': tf.data.experimental.AUTOTUNE,
    'shuffle_seed': 42
}
```

---

## 🔄 データ更新・版数管理

### 版数管理
```
data/training/
├── v1.0/                    # 評価ベース変換 (Pattern 1)
│   ├── pseudo_users.json
│   ├── interactions.json
│   └── metadata.json
├── v1.1/                    # 品質改善版
├── v2.0/                    # Pattern 2 実装時
└── latest -> v1.0/          # 最新版シンボリックリンク
```

### メタデータ管理
```json
{
  "version": "1.0",
  "created_at": "2025-09-05T12:00:00Z",
  "data_sources": {
    "reviews_count": 38904,
    "users_count": 50,
    "videos_count": 32304
  },
  "processing_params": {
    "like_threshold": 4.0,
    "skip_threshold": 3.0,
    "min_interactions": 10
  },
  "quality_metrics": {
    "like_ratio": 0.777,
    "coverage": 0.95,
    "completeness": 0.98
  }
}
```

---

**文書管理**  
**最終更新**: 2025年9月5日  
**管理者**: Claude Code