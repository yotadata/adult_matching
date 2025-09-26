# 統合推薦システム

## 概要

このディレクトリには統合された推薦システムが含まれています。従来の3つの推薦関数（`recommendations`、`two_tower_recommendations`、`enhanced_two_tower_recommendations`）を単一の高性能システムに統合しました。

## 構造

```
recommendations/
├── enhanced_two_tower/     # メイン推薦エンジン（768次元Two-Tower）
│   └── index.ts
├── index.ts               # 後方互換性プロキシ（旧recommendations用）
└── README.md
```

## エンドポイント

### メイン推薦エンドポイント
`POST /functions/v1/recommendations/enhanced_two_tower`

最新の768次元Two-Tower推薦システム。全ての新しいアプリケーションはこのエンドポイントを使用してください。

### 後方互換性エンドポイント
`POST /functions/v1/recommendations`

旧`recommendations`関数との互換性を提供。自動的に`enhanced_two_tower`にリダイレクトされます。

## API仕様

### リクエスト

```typescript
interface RecommendationRequest {
  // 標準パラメータ
  limit?: number;                    // 推薦数（デフォルト: 20）
  exclude_liked?: boolean;           // いいね済み除外（デフォルト: true）
  diversity_weight?: number;         // 多様性重み（デフォルト: 0.3）
  include_reasons?: boolean;         // 推薦理由含む（デフォルト: true）
  min_similarity_threshold?: number; // 最小類似度（デフォルト: 0.1）
  
  // アルゴリズム選択
  algorithm?: 'enhanced' | 'basic' | 'two_tower';
  
  // 後方互換性パラメータ
  max_results?: number;              // limit のエイリアス
  include_explanations?: boolean;    // include_reasons のエイリアス
}
```

### レスポンス

```typescript
interface RecommendationResponse {
  recommendations: EnhancedRecommendedVideo[];
  metrics: RecommendationMetrics;
  _compatibility_mode?: string;      // 互換性モード情報
  _redirected_to?: string;          // リダイレクト先情報
}

interface EnhancedRecommendedVideo {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url: string;
  maker: string;
  genre: string;
  price: number;
  sample_video_url: string;
  image_urls: string[];
  performers: string[];
  tags: string[];
  similarity_score: number;          // 類似度スコア（0-1）
  recommendation_reason: string;     // 推薦理由
  diversity_score?: number;         // 多様性スコア
  confidence_score?: number;        // 信頼度スコア
}
```

## アルゴリズム

### Enhanced（推奨）
- 768次元Two-Towerモデル
- 高度なユーザー埋め込み生成
- 多様性強化アルゴリズム
- ジャンル嗜好分析
- 最高の推薦精度

### Basic（互換性）
- 基本的なコサイン類似度
- 旧`recommendations`関数互換
- シンプルな推薦理由

### Two-Tower（互換性）
- 簡易Two-Tower実装
- 旧`two_tower_recommendations`関数互換
- 64次元→768次元自動アップグレード

## パフォーマンス

### 目標性能
- **応答時間**: <500ms
- **スループット**: 200+ req/s
- **精度**: AUC-PR ≥ 0.85

### 最適化機能
- バッチ類似度計算
- メモリ使用量制限
- 並列処理対応
- キャッシュ戦略

## 使用例

### JavaScript/TypeScript
```javascript
const response = await fetch('/functions/v1/recommendations/enhanced_two_tower', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  },
  body: JSON.stringify({
    limit: 10,
    exclude_liked: true,
    diversity_weight: 0.3,
    include_reasons: true,
    algorithm: 'enhanced'
  })
});

const data = await response.json();
console.log('推薦動画:', data.recommendations);
console.log('メトリクス:', data.metrics);
```

### cURL
```bash
curl -X POST "https://your-project.supabase.co/functions/v1/recommendations/enhanced_two_tower" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "exclude_liked": true,
    "include_reasons": true,
    "algorithm": "enhanced"
  }'
```

## 移行ガイド

### 旧recommendations関数からの移行

**Before:**
```javascript
fetch('/functions/v1/recommendations', {
  method: 'POST',
  body: JSON.stringify({
    max_results: 10,
    include_explanations: true
  })
});
```

**After（推奨）:**
```javascript
fetch('/functions/v1/recommendations/enhanced_two_tower', {
  method: 'POST',
  body: JSON.stringify({
    limit: 10,
    include_reasons: true,
    algorithm: 'enhanced'
  })
});
```

**After（互換性保持）:**
```javascript
// 既存コードは変更なしで動作（自動リダイレクト）
fetch('/functions/v1/recommendations', {
  method: 'POST',
  body: JSON.stringify({
    max_results: 10,
    include_explanations: true
  })
});
```

### 旧two_tower_recommendations関数からの移行

自動的に768次元システムにアップグレードされます。既存のAPIコールは変更なしで動作しますが、精度が大幅に向上します。

## エラーハンドリング

### 一般的なエラー

```javascript
{
  "error": "No user embedding found",
  "details": "User has no interaction history",
  "suggestion": "Use feed_explore for new users"
}
```

### パフォーマンスエラー

```javascript
{
  "error": "Request timeout",
  "details": "Similarity computation exceeded time limit",
  "metrics": {
    "similarity_computation_time": 1200,
    "total_processing_time": 1500
  }
}
```

## 監視・デバッグ

### メトリクス情報
```javascript
{
  "metrics": {
    "total_candidates": 50000,        // 候補動画数
    "embedding_hits": 45000,          // 埋め込み有り動画数
    "similarity_computation_time": 120, // 類似度計算時間(ms)
    "sort_time": 15,                  // ソート時間(ms)
    "total_processing_time": 350,     // 総処理時間(ms)
    "diversity_score": 0.73,          // 多様性スコア
    "avg_confidence": 0.82            // 平均信頼度
  }
}
```

### デバッグログ
- `algorithm`: 使用アルゴリズム
- `compatibility_mode`: 互換性モード
- `redirected_to`: リダイレクト先
- `migration_notice`: 移行通知

## トラブルシューティング

### よくある問題

1. **推薦が返らない**
   - ユーザー埋め込みが存在しない
   - → `feed_explore`で初期推薦を提供

2. **応答が遅い**
   - 大量の候補動画
   - → `limit`を調整、`min_similarity_threshold`を上げる

3. **推薦の多様性が低い**
   - `diversity_weight`を増加（0.5-0.7推奨）
   - `algorithm: 'enhanced'`を使用

### パフォーマンス調整

```javascript
// 高速だが精度重視
{
  "limit": 10,
  "min_similarity_threshold": 0.3,
  "diversity_weight": 0.2,
  "algorithm": "enhanced"
}

// 多様性重視
{
  "limit": 20,
  "min_similarity_threshold": 0.1,
  "diversity_weight": 0.5,
  "algorithm": "enhanced"
}
```