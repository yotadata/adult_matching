# Two-Tower Recommendation Model Design

## アーキテクチャ概要

Two-Towerモデルは、ユーザーと動画を別々のタワー（ニューラルネットワーク）でエンベディングし、コサイン類似度で推奨度を計算する手法です。

```
User Features → User Tower (768-dim embedding)
                         ↓
                    Cosine Similarity → Recommendation Score
                         ↑
Item Features → Item Tower (768-dim embedding)
```

## データベース設計

### 現在のテーブル
- `videos`: 動画メタデータ
- `video_embeddings`: 動画の768次元エンベディング
- `user_embeddings`: ユーザーの768次元エンベディング
- `likes`: ユーザーの行動履歴

### 追加テーブル（必要に応じて）
```sql
-- モデルバージョン管理
CREATE TABLE embedding_model_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  version TEXT NOT NULL,
  model_type TEXT NOT NULL, -- 'user_tower' or 'item_tower'
  model_path TEXT, -- Supabase Storage path
  is_active BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 特徴量重要度（可視化・デバッグ用）
CREATE TABLE feature_importance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_version_id UUID REFERENCES embedding_model_versions(id),
  feature_name TEXT NOT NULL,
  importance_score FLOAT NOT NULL
);
```

## 特徴量設計

### User Tower Features
- **行動履歴**: いいねした動画の特徴
  - ジャンル分布
  - メーカー分布
  - 出演者分布
  - タグ分布
  - 価格帯分布
- **時系列特徴**: いいねの時間パターン
- **統計特徴**: いいね総数、アクティビティ頻度

### Item Tower Features  
- **メタデータ**: タイトル、説明文のテキスト埋め込み
- **カテゴリ**: ジャンル、メーカー、出演者
- **数値特徴**: 価格、再生時間、人気度
- **視覚特徴**: サムネイル画像埋め込み（将来拡張）

## 実装アーキテクチャ

### 1. Supabase Edge Functions
```typescript
// 1. update_user_embedding
// - ユーザーの行動履歴から特徴量を抽出
// - User Towerで768次元エンベディングを生成
// - user_embeddingsテーブルを更新

// 2. recommendations  
// - ユーザーエンベディングと全動画エンベディングの類似度計算
// - コサイン類似度でランキング
// - 重複排除・フィルタリング
```

### 2. ローカル学習環境
```python
# scripts/train_two_tower_model.py
# - PostgreSQLから学習データ取得
# - TensorFlow/PyTorchでTwo-Towerモデル学習
# - ONNXまたはTensorFlow.js形式で保存
# - Supabase Storageにアップロード
```

### 3. モデルデプロイメント
- **初期**: ローカル学習 → Supabase Storage → Edge Functions
- **更新**: 定期バッチ処理でモデル再学習・更新
- **A/Bテスト**: 複数モデルバージョンの並列運用

## エンベディング生成方式

### User Embedding生成
```typescript
// 擬似コード
const userFeatures = {
  genre_distribution: calculateGenreDistribution(userLikes),
  maker_distribution: calculateMakerDistribution(userLikes), 
  price_preference: calculatePricePreference(userLikes),
  recency_weight: calculateRecencyWeight(userLikes),
  activity_level: calculateActivityLevel(userLikes)
};

const userEmbedding = await userTower.predict(userFeatures);
```

### Item Embedding生成（事前計算）
```python
# バッチ処理
item_features = {
    'text_embedding': encode_text(title + description),
    'genre_one_hot': encode_genre(genre),
    'maker_embedding': lookup_maker_embedding(maker),
    'price_normalized': normalize_price(price),
    'popularity_score': calculate_popularity(likes_count)
}

item_embedding = item_tower.predict(item_features)
```

## Supabase内でのモデル実行

### Option 1: ONNX.js in Edge Functions
```typescript
import { InferenceSession } from 'onnxjs';

const session = await InferenceSession.create('./models/user_tower.onnx');
const embedding = await session.run(inputFeatures);
```

### Option 2: TensorFlow.js
```typescript
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('supabase://storage/models/user_tower.json');
const embedding = model.predict(inputTensor);
```

### Option 3: 簡易版実装
```typescript
// 線形結合による簡易エンベディング
const weights = await getModelWeights();
const embedding = computeWeightedSum(features, weights);
```

## 実装フェーズ

### Phase 1: 簡易版Two-Tower
- ルールベース特徴量抽出
- 線形結合によるエンベディング生成
- コサイン類似度による推奨

### Phase 2: ML強化版
- TensorFlow.jsまたはONNX.js導入
- 学習済みニューラルネットワーク使用
- A/Bテスト機能追加

### Phase 3: 高度化
- リアルタイム学習
- マルチモーダル特徴量（画像・テキスト）
- 強化学習による最適化

## パフォーマンス考慮

### レスポンス時間目標
- User embedding更新: < 2秒
- Recommendation生成: < 1秒  

### スケーラビリティ
- Item embeddingは事前計算・キャッシュ
- User embeddingは差分更新
- 類似度計算の並列化

## 監視・評価

### メトリクス
- クリック率(CTR)
- 滞在時間  
- いいね率
- 多様性指標

### A/Bテスト
- 異なるモデルバージョンの比較
- 特徴量の重要度評価
- ハイパーパラメータ最適化