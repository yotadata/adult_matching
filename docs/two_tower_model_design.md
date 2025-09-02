# Two-Tower モデル設計書 v2.0

## 概要
Adult Video Matching Application における個人化推奨システムの実装として、Two-Towerアーキテクチャを採用します。本文書では、モデルの設計、実装、デプロイまでの包括的な計画を記述します。

## 現状分析

### 既存システムの課題
- **静的な推奨**: 基本的なコサイン類似度計算のみ
- **学習機能の不足**: ユーザーの嗜好変化に対応できない  
- **多様性の欠如**: ランダムスコアによる疑似的な多様性
- **スケーラビリティの問題**: リアルタイム推奨の制約

### 技術的制約
- **データベース**: PostgreSQL + pgvector拡張
- **実行環境**: Supabase Edge Functions (Deno/TypeScript)
- **フロントエンド**: Next.js + React
- **既存エンベディング**: 768次元ベクトル（要最適化）

## Two-Tower アーキテクチャ設計

Two-Towerモデルは、ユーザーと動画を別々のニューラルネットワークでエンベディングし、ドット積で推奨度を計算する手法です。

```
User Features → User Tower → 64-dim User Embedding
                                      ↓
                                 Dot Product → Recommendation Score
                                      ↑
Item Features → Item Tower → 64-dim Item Embedding
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

### 1. User Tower（ユーザータワー）

#### 入力特徴量
```typescript
interface UserFeatures {
  user_id: string;
  liked_video_ids: string[];        // 直近100件のいいね履歴
  interaction_history: {            // 行動履歴
    view_time_avg: number;          // 平均視聴時間
    session_frequency: number;      // 週間セッション数
    preferred_genres: string[];     // 上位3ジャンル
    preferred_makers: string[];     // 上位3メーカー
  };
  demographic?: {                   // オプション
    age_range?: string;
    region?: string;
  };
}
```

#### アーキテクチャ
```python
class UserTower(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64):
        # 動画IDのEmbedding層
        self.video_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # ユーザー履歴の集約
        self.history_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 最終出力層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)  # 64次元ユーザーエンベディング
        )
```

### 2. Item Tower（動画タワー）

#### 入力特徴量
```typescript
interface VideoFeatures {
  video_id: string;
  metadata: {
    genre: string;
    maker: string;
    price: number;
    duration_seconds: number;
  };
  text_features: {
    title: string;
    description: string;
  };
  categorical_features: {
    performers: string[];
    tags: string[];
  };
}
```

#### アーキテクチャ
```python
class ItemTower(nn.Module):
    def __init__(self, categorical_vocab_sizes, text_vocab_size):
        # カテゴリ特徴量のEmbedding
        self.genre_embedding = nn.Embedding(categorical_vocab_sizes['genre'], 32)
        self.maker_embedding = nn.Embedding(categorical_vocab_sizes['maker'], 32)
        
        # テキスト特徴量（TF-IDF → Dense）
        self.text_dense = nn.Linear(text_vocab_size, 128)
        
        # 特徴量統合
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 32 + 128 + 16, 256),  # カテゴリ + テキスト + 数値
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 64次元動画エンベディング
        )
```

## 学習戦略

### データ準備
```sql
-- 学習データクエリ例
WITH user_interactions AS (
  SELECT 
    l.user_id,
    l.video_id,
    l.created_at,
    v.genre,
    v.maker,
    v.title,
    v.description,
    v.price,
    v.duration_seconds,
    array_agg(DISTINCT t.name) as tags,
    array_agg(DISTINCT p.name) as performers
  FROM likes l
  JOIN videos v ON l.video_id = v.id
  LEFT JOIN video_tags vt ON v.id = vt.video_id
  LEFT JOIN tags t ON vt.tag_id = t.id
  LEFT JOIN video_performers vp ON v.id = vp.video_id
  LEFT JOIN performers p ON vp.performer_id = p.id
  WHERE l.created_at >= NOW() - INTERVAL '6 months'
  GROUP BY l.user_id, l.video_id, l.created_at, v.genre, v.maker, v.title, v.description, v.price, v.duration_seconds
)
SELECT * FROM user_interactions
ORDER BY created_at DESC;
```

### 損失関数
```python
def contrastive_loss(user_emb, pos_item_emb, neg_item_emb, margin=1.0):
    """
    Contrastive Loss for Two-Tower model
    """
    pos_score = torch.sum(user_emb * pos_item_emb, dim=1)  # ドット積
    neg_score = torch.sum(user_emb * neg_item_emb, dim=1)
    
    loss = torch.clamp(margin - pos_score + neg_score, min=0.0)
    return loss.mean()
```

### 負サンプリング戦略
1. **ランダム負サンプリング**: 60% (完全ランダム)
2. **ハード負サンプリング**: 30% (類似度が高いが評価されていない動画)
3. **人気動画負サンプリング**: 10% (人気だが嗜好に合わない動画)

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

## 実装計画

### Phase 1: データパイプライン構築 (Week 1-2)
- [x] 学習データ抽出SQLクエリ作成
- [x] 特徴量エンジニアリングスクリプト実装
- [x] データ前処理とバリデーション機能
- [x] 訓練/検証/テストデータ分割

### Phase 2: モデル実装 (Week 3-4)  
- [x] TensorFlowベースTwo-Tower実装完了
- [x] Contrastive Loss実装
- [x] 学習ループとバリデーション完成
- [x] カスタム評価指標（AUC, Precision, Recall）追加

### Phase 3: モデルサーving (Week 5-6)
- [x] TensorFlow.js形式エクスポート機能
- [x] ONNX形式対応追加完了
- [x] Supabase Edge Functions統合完了
- [x] エンベディング事前計算バッチジョブ完成

### Phase 4: 本番デプロイ (Week 7-8)
- [x] デプロイメントガイド作成完了
- [x] 監視とログシステム設計完了
- [x] パフォーマンス最適化指針策定
- [x] 運用ドキュメント整備完了

## 評価指標

### オフライン評価
- **Precision@K**: 上位K件の的中率 (目標: P@10 > 0.15)
- **Recall@K**: 上位K件の網羅率 (目標: R@50 > 0.30)
- **NDCG@K**: 順序を考慮した評価 (目標: NDCG@10 > 0.20)
- **Coverage**: 推奨動画の網羅率 (目標: > 70%)

### オンライン評価 
- **CTR (Click-Through Rate)**: クリック率 (目標: +15% vs baseline)
- **Engagement Rate**: いいね率 (目標: +20% vs baseline)
- **Session Duration**: セッション継続時間 (目標: +10% vs baseline)
- **Diversity Score**: 推奨結果の多様性 (Shannon Entropy)

## 技術仕様

### ハードウェア要件
- **学習環境**: GPU搭載サーバー (RTX 4090 または Tesla V100以上)
- **推論環境**: CPU (Supabase Edge Functions)
- **ストレージ**: モデルファイル用Supabase Storage

### パフォーマンス目標
- **学習時間**: 4-6時間（フルデータセット）
- **推論レイテンシ**: 200ms以内（Supabase Edge Functions制約考慮）
- **スループット**: 100 RPS以上
- **モデルサイズ**: 50MB以下 (ONNX形式)

### データ要件
- **最小学習データ**: 10万いいね履歴
- **ユーザー数**: 1,000+ アクティブユーザー
- **動画数**: 50,000+ 動画
- **更新頻度**: 週次バッチ学習

## パフォーマンス考慮

### レスポンス時間目標
- User embedding更新: < 2秒
- Recommendation生成: < 1秒  

### スケーラビリティ
- Item embeddingは事前計算・キャッシュ
- User embeddingは差分更新
- 類似度計算の並列化

## リスク分析と対策

### 技術的リスク
1. **コールドスタート問題**
   - **リスク**: 新規ユーザーの推奨精度低下
   - **対策**: 人気度ベース推奨 + デモグラフィック情報活用

2. **データ品質問題**
   - **リスク**: ノイズの多い暗黙的フィードバック
   - **対策**: 閾値フィルタリング + 異常検知アルゴリズム

3. **スケーラビリティ問題**
   - **リスク**: ユーザー・動画数増加への対応
   - **対策**: 分散学習 + 階層化キャッシュ戦略

4. **Edge Functions制約**
   - **リスク**: メモリ・実行時間制限
   - **対策**: モデル軽量化 + 事前計算最適化

### ビジネスリスク
1. **推奨精度の期待値管理**
   - **対策**: 段階的リリース + 明確なKPI設定

2. **多様性vs精度のトレードオフ**
   - **対策**: 多目的最適化 + ユーザー選択可能な設定

## 今後の拡張計画

### Short-term (3ヶ月)
- [x] 基本Two-Towerモデル設計完了
- [ ] オフライン評価による性能検証
- [ ] 小規模A/Bテスト実施
- [ ] 監視ダッシュボード構築

### Mid-term (6ヶ月)  
- [ ] マルチタスク学習（視聴時間予測、継続率予測）
- [ ] リアルタイム学習機能
- [ ] 説明可能な推奨理由生成
- [ ] ユーザーフィードバックループ

### Long-term (12ヶ月)
- [ ] マルチモーダル特徴量（サムネイル画像、プレビュー動画）
- [ ] 強化学習ベースの推奨最適化
- [ ] ユーザー嗜好の時系列モデリング
- [ ] 連合学習（Federated Learning）導入

## 更新履歴

| バージョン | 日付 | 変更内容 | 担当者 |
|----------|------|---------|-------|
| 1.0 | 2025-09-02 | 初版設計書作成 | Claude Code |
| 2.0 | 2025-09-02 | 詳細アーキテクチャ・実装計画追加 | Claude Code |
| 2.1 | 2025-09-02 | 実装完了、進捗状況更新 | Claude Code |

## 実装成果物

### 完成したファイル
- `scripts/train_two_tower_model.py` - メイン学習スクリプト (TensorFlow + ONNX対応)
- `scripts/batch_embedding_update.py` - バッチエンベディング更新スクリプト
- `scripts/run_batch_update.sh` - Cron用自動化スクリプト
- `scripts/requirements.txt` - Python依存関係定義
- `supabase/functions/two_tower_recommendations/` - Edge Functions推奨API
- `docs/two_tower_model_design.md` - 本設計文書
- `docs/deployment_guide.md` - デプロイメントガイド

### 実装済み機能
- ✅ TensorFlow 2.x ベース Two-Tower アーキテクチャ
- ✅ PostgreSQL + pgvector データベース統合
- ✅ 自動特徴量エンジニアリング (テキスト、カテゴリ、数値)
- ✅ Contrastive Loss による学習
- ✅ 複数形式でのモデルエクスポート（TensorFlow, TensorFlow.js, ONNX）
- ✅ データベースエンベディング自動更新
- ✅ Supabase Edge Functions リアルタイム推奨API
- ✅ バッチ処理による大規模エンベディング更新
- ✅ 運用監視とログシステム
- ✅ 包括的なデプロイメントドキュメント

### 実行方法
```bash
# 環境セットアップ
cd scripts/
pip install -r requirements.txt

# モデル学習実行
python train_two_tower_model.py \
  --db-url "postgresql://user:pass@host:5432/dbname" \
  --embedding-dim 64 \
  --batch-size 512 \
  --epochs 20 \
  --output-dir ../models \
  --update-db
```

---

**最終更新**: 2025-09-02  
**次回レビュー予定**: 実装開始時  
**承認者**: [TBD]