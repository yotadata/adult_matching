# 🚀 Adult Matching API仕様書

## 📋 概要

Adult Matchingバックエンドの統合API仕様書です。リファクタリング後の全APIエンドポイントを網羅しています。

**APIバージョン**: 2.0
**最終更新**: 2025年9月17日
**ベースURL**: `https://your-project-ref.supabase.co/functions/v1`

## 🔐 認証

全APIエンドポイントはSupabase Authenticationを使用します。

```http
Authorization: Bearer <supabase-jwt-token>
```

### 認証方式
- **匿名ユーザー**: `anon_key`を使用
- **認証済みユーザー**: JWTトークンを使用
- **管理者**: `service_role_key`を使用（管理API用）

## 🎯 統合Edge Functions API

### 1. 推薦システム API

#### 1.1 Enhanced Two-Tower 推薦 [統合]

**エンドポイント**: `/enhanced_two_tower_recommendations`
**メソッド**: `POST`
**説明**: 768次元Two-Towerモデルによる高精度推薦

**リクエスト**:
```json
{
  "user_id": "string (UUID)",
  "num_recommendations": "number (1-50, default: 10)",
  "algorithm": "string (two_tower|collaborative|hybrid, default: two_tower)",
  "filters": {
    "genres": ["string"],
    "makers": ["string"],
    "exclude_liked": "boolean (default: true)",
    "exclude_ids": ["string (UUID)"]
  },
  "context": {
    "session_id": "string",
    "device_type": "string (mobile|desktop)",
    "timestamp": "string (ISO 8601)"
  }
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "video_id": "string (UUID)",
        "title": "string",
        "thumbnail_url": "string (URL)",
        "preview_video_url": "string (URL)",
        "maker": "string",
        "genre": "string[]",
        "performers": "string[]",
        "tags": "string[]",
        "score": "number (0-1)",
        "reasoning": "string"
      }
    ],
    "total_count": "number",
    "algorithm_used": "string",
    "diversity_score": "number (0-1)"
  },
  "metadata": {
    "response_time_ms": "number",
    "cache_hit": "boolean",
    "model_version": "string",
    "timestamp": "string (ISO 8601)"
  }
}
```

**エラーレスポンス**:
```json
{
  "success": false,
  "error": {
    "code": "string (ERROR_CODE)",
    "message": "string",
    "details": "object"
  }
}
```

### 2. ユーザー管理 API

#### 2.1 いいね管理

**エンドポイント**: `/user-management/likes`
**メソッド**: `POST`
**説明**: 動画へのいいね追加・削除

**リクエスト**:
```json
{
  "user_id": "string (UUID)",
  "video_id": "string (UUID)",
  "action": "string (like|unlike)",
  "context": {
    "session_id": "string",
    "source": "string (swipe|button|auto)",
    "timestamp": "string (ISO 8601)"
  }
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "like_id": "string (UUID)",
    "user_id": "string (UUID)",
    "video_id": "string (UUID)",
    "action": "string",
    "created_at": "string (ISO 8601)"
  },
  "metadata": {
    "total_likes": "number",
    "embedding_updated": "boolean"
  }
}
```

**エンドポイント**: `/user-management/likes`
**メソッド**: `GET`
**説明**: ユーザーのいいね履歴取得

**クエリパラメータ**:
```
user_id: string (UUID, required)
limit: number (1-100, default: 20)
offset: number (default: 0)
sort: string (newest|oldest, default: newest)
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "likes": [
      {
        "like_id": "string (UUID)",
        "video_id": "string (UUID)",
        "video_title": "string",
        "video_thumbnail": "string (URL)",
        "liked_at": "string (ISO 8601)"
      }
    ],
    "pagination": {
      "total": "number",
      "limit": "number",
      "offset": "number",
      "has_more": "boolean"
    }
  }
}
```

#### 2.2 埋め込み更新

**エンドポイント**: `/user-management/embeddings`
**メソッド**: `PUT`
**説明**: ユーザー埋め込みベクターの更新

**リクエスト**:
```json
{
  "user_id": "string (UUID)",
  "interaction_data": {
    "liked_videos": ["string (UUID)"],
    "skipped_videos": ["string (UUID)"],
    "viewing_history": [
      {
        "video_id": "string (UUID)",
        "watch_duration": "number (seconds)",
        "timestamp": "string (ISO 8601)"
      }
    ]
  },
  "update_mode": "string (incremental|full, default: incremental)"
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "user_id": "string (UUID)",
    "embedding_vector": "number[] (768 dimensions)",
    "updated_at": "string (ISO 8601)",
    "embedding_version": "string"
  },
  "metadata": {
    "processing_time_ms": "number",
    "interactions_processed": "number"
  }
}
```

#### 2.3 アカウント管理

**エンドポイント**: `/user-management/account`
**メソッド**: `DELETE`
**説明**: ユーザーアカウント削除

**リクエスト**:
```json
{
  "user_id": "string (UUID)",
  "confirmation": "string (DELETE_MY_ACCOUNT)",
  "reason": "string (optional)"
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "user_id": "string (UUID)",
    "deleted_at": "string (ISO 8601)",
    "data_retained": "boolean",
    "retention_period_days": "number"
  }
}
```

### 3. コンテンツ配信 API

#### 3.1 統合フィード

**エンドポイント**: `/content/feed`
**メソッド**: `POST`
**説明**: 全フィードタイプに対応した統合コンテンツAPI

**リクエスト**:
```json
{
  "feed_type": "string (explore|personalized|latest|popular|random)",
  "user_id": "string (UUID, optional for anonymous)",
  "limit": "number (1-50, default: 20)",
  "offset": "number (default: 0)",
  "filters": {
    "genres": ["string"],
    "makers": ["string"],
    "tags": ["string"],
    "exclude_ids": ["string (UUID)"],
    "min_rating": "number (0-5)",
    "max_price": "number"
  },
  "options": {
    "include_metadata": "boolean (default: true)",
    "diversity_factor": "number (0-1, default: 0.3)",
    "freshness_factor": "number (0-1, default: 0.2)"
  }
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "videos": [
      {
        "id": "string (UUID)",
        "title": "string",
        "description": "string",
        "thumbnail_url": "string (URL)",
        "preview_video_url": "string (URL)",
        "maker": "string",
        "genre": "string[]",
        "performers": "string[]",
        "tags": "string[]",
        "price": "number",
        "rating": "number (0-5)",
        "duration_seconds": "number",
        "published_at": "string (ISO 8601)",
        "metadata": {
          "score": "number (0-1)",
          "reasoning": "string",
          "view_count": "number",
          "like_count": "number"
        }
      }
    ],
    "feed_info": {
      "feed_type": "string",
      "total_available": "number",
      "diversity_score": "number (0-1)",
      "personalization_score": "number (0-1)"
    },
    "pagination": {
      "limit": "number",
      "offset": "number",
      "has_more": "boolean",
      "next_offset": "number"
    }
  },
  "metadata": {
    "response_time_ms": "number",
    "cache_hit": "boolean",
    "algorithm_used": "string",
    "timestamp": "string (ISO 8601)"
  }
}
```

#### フィードタイプ詳細

**explore**: 多様性重視の探索フィード
- 新規ユーザー向け
- ジャンル・メーカーの多様性を重視
- 人気コンテンツとニッチコンテンツのバランス

**personalized**: パーソナライズドフィード
- ユーザー埋め込みベースの推薦
- Two-Towerモデル使用
- 過去のいいね履歴を考慮

**latest**: 最新動画フィード
- 公開日時順（新しい順）
- フィルタリング対応
- 定期的な更新

**popular**: 人気動画フィード
- いいね数・視聴数ベース
- 時間窓（24時間、週間、月間）
- トレンド分析

**random**: ランダムフィード
- 完全ランダム選択
- フィルタリング後のランダム化
- 偶発的発見促進

## 🗃️ Database RPC Functions

### 推薦関連

#### get_personalized_videos_feed
```sql
SELECT * FROM get_personalized_videos_feed(
  p_user_id UUID,
  p_limit INTEGER DEFAULT 20,
  p_offset INTEGER DEFAULT 0,
  p_diversity_factor FLOAT DEFAULT 0.3
);
```

#### get_popular_videos_feed
```sql
SELECT * FROM get_popular_videos_feed(
  p_limit INTEGER DEFAULT 20,
  p_offset INTEGER DEFAULT 0,
  p_time_window TEXT DEFAULT '7d'
);
```

#### get_explore_videos_feed
```sql
SELECT * FROM get_explore_videos_feed(
  p_limit INTEGER DEFAULT 20,
  p_offset INTEGER DEFAULT 0,
  p_diversity_factor FLOAT DEFAULT 0.5
);
```

#### get_similar_videos
```sql
SELECT * FROM get_similar_videos(
  p_video_id UUID,
  p_limit INTEGER DEFAULT 10,
  p_similarity_threshold FLOAT DEFAULT 0.7
);
```

#### get_random_videos
```sql
SELECT * FROM get_random_videos(
  p_limit INTEGER DEFAULT 20,
  p_filters JSONB DEFAULT '{}'
);
```

### ユーザー関連

#### update_user_recommendation_feedback
```sql
SELECT update_user_recommendation_feedback(
  p_user_id UUID,
  p_video_id UUID,
  p_action TEXT,
  p_context JSONB DEFAULT '{}'
);
```

#### get_user_video_decisions
```sql
SELECT * FROM get_user_video_decisions(
  p_user_id UUID,
  p_limit INTEGER DEFAULT 100
);
```

#### get_videos_feed
```sql
SELECT * FROM get_videos_feed(
  p_feed_type TEXT,
  p_user_id UUID DEFAULT NULL,
  p_limit INTEGER DEFAULT 20,
  p_offset INTEGER DEFAULT 0,
  p_filters JSONB DEFAULT '{}'
);
```

## 📊 レスポンスコード

| コード | 説明 | 詳細 |
|--------|------|------|
| 200 | Success | リクエスト成功 |
| 400 | Bad Request | 無効なリクエストパラメータ |
| 401 | Unauthorized | 認証が必要 |
| 403 | Forbidden | アクセス権限なし |
| 404 | Not Found | リソースが見つからない |
| 429 | Too Many Requests | レート制限超過 |
| 500 | Internal Server Error | サーバー内部エラー |

## 🔄 エラーハンドリング

### 標準エラーレスポンス
```json
{
  "success": false,
  "error": {
    "code": "string",
    "message": "string",
    "details": "object",
    "timestamp": "string (ISO 8601)",
    "request_id": "string (UUID)"
  }
}
```

### エラーコード一覧

| コード | 説明 |
|--------|------|
| `INVALID_USER_ID` | 無効なユーザーID |
| `INVALID_VIDEO_ID` | 無効な動画ID |
| `INSUFFICIENT_DATA` | 推薦に必要なデータ不足 |
| `RATE_LIMIT_EXCEEDED` | レート制限超過 |
| `UNAUTHORIZED_ACCESS` | 認証エラー |
| `RESOURCE_NOT_FOUND` | リソース見つからず |
| `INTERNAL_ERROR` | 内部処理エラー |
| `VALIDATION_ERROR` | バリデーションエラー |

## 📈 パフォーマンス仕様

### レスポンス時間目標
- **推薦API**: < 300ms (P95)
- **フィードAPI**: < 200ms (P95)
- **ユーザー管理API**: < 150ms (P95)
- **RPC Functions**: < 50ms (P95)

### レート制限
- **認証済みユーザー**: 1000 req/hour
- **匿名ユーザー**: 100 req/hour
- **管理者**: 10000 req/hour

### キャッシュ戦略
- **フィードデータ**: 5分TTL
- **推薦結果**: 15分TTL
- **ユーザープロファイル**: 1時間TTL

## 🔐 セキュリティ仕様

### 入力検証
- **SQLインジェクション防止**: パラメータ化クエリ
- **XSS防止**: 入力サニタイゼーション
- **認証**: JWT検証・RLS適用

### データ保護
- **個人情報**: 暗号化保存
- **ログ**: PII除外
- **アクセス制御**: 最小権限原則

## 📚 使用例

### JavaScript/TypeScript
```typescript
// 推薦取得
const recommendations = await fetch('/functions/v1/enhanced_two_tower_recommendations', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_id: 'user-uuid',
    num_recommendations: 10,
    filters: {
      genres: ['action', 'drama'],
      exclude_liked: true
    }
  })
});

// フィード取得
const feed = await fetch('/functions/v1/content/feed', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    feed_type: 'personalized',
    user_id: 'user-uuid',
    limit: 20
  })
});
```

### Python
```python
import requests

# 推薦取得
response = requests.post(
    'https://your-project.supabase.co/functions/v1/enhanced_two_tower_recommendations',
    headers={
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    },
    json={
        'user_id': 'user-uuid',
        'num_recommendations': 10,
        'algorithm': 'two_tower'
    }
)

recommendations = response.json()
```

### cURL
```bash
# 推薦取得
curl -X POST "https://your-project.supabase.co/functions/v1/enhanced_two_tower_recommendations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "num_recommendations": 10,
    "filters": {
      "exclude_liked": true
    }
  }'

# フィード取得
curl -X POST "https://your-project.supabase.co/functions/v1/content/feed" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "feed_type": "explore",
    "limit": 20
  }'
```

## 🛠️ 開発・テスト

### ローカル開発
```bash
# Supabase起動
supabase start

# Edge Functions起動
supabase functions serve

# テストURL
http://localhost:54321/functions/v1/enhanced_two_tower_recommendations
```

### テスト実行
```bash
# 統合テスト
python backend/tests/integration/content/run_content_api_tests.py all

# APIテスト
pytest backend/tests/integration/content/ -v
```

## 📞 サポート

### 技術サポート
- **ドキュメント**: [開発者ガイド](developer-onboarding.md)
- **アーキテクチャ**: [バックエンド設計](backend-architecture.md)
- **トラブルシューティング**: [問題解決ガイド](troubleshooting.md)

### 連絡先
- **Email**: backend-team@adult-matching.com
- **GitHub Issues**: リポジトリのIssues
- **Emergency**: 24/7 On-call Support

---

**API仕様書バージョン**: 2.0
**最終更新**: 2025年9月17日
**作成者**: Claude Code Assistant
**ステータス**: ✅ 本番稼働中