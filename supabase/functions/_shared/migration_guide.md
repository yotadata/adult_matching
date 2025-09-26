# Edge Functions共有依存関係標準化移行ガイド

## 概要
このガイドでは、Edge Functionsで共有依存関係を標準化するための移行手順を説明します。

## 標準化された共有モジュール構造

```
supabase/functions/_shared/
├── auth.ts              # 統一認証ユーティリティ
├── database.ts          # データベース接続・操作
├── types.ts            # 共通型定義
├── responses.ts        # HTTPレスポンス処理
├── validation.ts       # 入力検証
├── content.ts          # コンテンツ処理
├── monitoring.ts       # 監視・ログ
└── index.ts           # エクスポートインデックス
```

## 移行手順

### 1. 既存のローカル_sharedフォルダーから移行

#### Before（非標準化）:
```typescript
// supabase/functions/user-management/_shared/auth.ts から
import { authenticateUser } from "../_shared/auth.ts";

// supabase/functions/content/_shared/content.ts から
import { getVideoContent } from "../_shared/content.ts";
```

#### After（標準化済み）:
```typescript
// 全てのEdge Functionsで統一
import { authenticateUser, requireAuth } from "../_shared/auth.ts";
import { getVideoContent, searchVideosWithFilters } from "../_shared/content.ts";
import { successResponse, errorResponse } from "../_shared/responses.ts";
import { validateFeedRequest } from "../_shared/validation.ts";
```

### 2. インポート文の標準化

#### 推奨パターン:
```typescript
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";

// 認証関連
import {
  authenticateUser,
  requireAuth,
  optionalAuth,
  withRateLimit
} from "../_shared/auth.ts";

// データベース関連
import {
  getSupabaseClientFromRequest,
  DatabaseResult,
  createSuccessResult,
  handleDatabaseError
} from "../_shared/database.ts";

// レスポンス関連
import {
  corsHeaders,
  successResponse,
  errorResponse,
  optionsResponse
} from "../_shared/responses.ts";

// バリデーション関連
import {
  validateFeedRequest,
  validateRecommendationRequest
} from "../_shared/validation.ts";

// コンテンツ関連
import {
  getVideoContent,
  searchVideosWithFilters,
  calculateContentMetrics
} from "../_shared/content.ts";

// 型定義
import type {
  VideoWithRelations,
  ContentFilters,
  RecommendationRequest
} from "../_shared/types.ts";
```

### 3. Edge Function実装の標準パターン

#### 基本構造:
```typescript
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import {
  corsHeaders,
  optionsResponse,
  successResponse,
  errorResponse,
  requireAuth
} from "../_shared/index.ts";

serve(async (req: Request) => {
  // CORS対応
  if (req.method === 'OPTIONS') {
    return optionsResponse();
  }

  // レート制限付き認証
  return await withRateLimit(req, async (req) => {
    return await requireAuth(req, async (req, user) => {
      try {
        // ビジネスロジックの実装
        const result = await processRequest(req, user);
        return successResponse(result);

      } catch (error) {
        console.error('Function error:', error);
        return errorResponse('Internal server error', 500);
      }
    });
  });
});
```

### 4. エラーハンドリングの標準化

#### Before:
```typescript
// 各Edge Functionで独自のエラー処理
if (error) {
  return new Response(JSON.stringify({ error: error.message }), {
    status: 500,
    headers: { 'Content-Type': 'application/json' }
  });
}
```

#### After:
```typescript
// 統一されたエラー処理
import { errorResponse, internalErrorResponse } from "../_shared/responses.ts";

if (error) {
  console.error('Operation failed:', error);
  return internalErrorResponse('Operation failed');
}
```

### 5. データベース操作の標準化

#### Before:
```typescript
// 各Edge Functionで独自のクライアント作成
const supabase = createClient(
  Deno.env.get('SUPABASE_URL'),
  Deno.env.get('SUPABASE_ANON_KEY')
);
```

#### After:
```typescript
// 統一されたクライアント取得
import { getSupabaseClientFromRequest } from "../_shared/database.ts";

const supabase = getSupabaseClientFromRequest(req, {
  useServiceRole: false // 必要に応じて設定
});
```

### 6. バリデーションの標準化

#### Before:
```typescript
// 各Edge Functionで独自のバリデーション
if (!user_id) {
  return new Response(JSON.stringify({ error: 'user_id is required' }), {
    status: 400
  });
}
```

#### After:
```typescript
// 統一されたバリデーション
import { validateRecommendationRequest, validationErrorResponse } from "../_shared/index.ts";

const validation = validateRecommendationRequest(requestData);
if (!validation.valid) {
  return validationErrorResponse(validation.errors);
}
```

## 移行チェックリスト

### Edge Function毎の移行作業

- [ ] **auth.ts系統の統合**
  - [ ] `user-management/_shared/auth.ts` の使用を `_shared/auth.ts` に変更
  - [ ] 認証関連インポートの統一
  - [ ] `requireAuth`、`optionalAuth` の使用

- [ ] **database.ts系統の統合**
  - [ ] 独自のSupabaseクライアント作成を統一関数に変更
  - [ ] `DatabaseResult` 型の使用
  - [ ] エラーハンドリングの統一

- [ ] **responses.ts系統の統合**
  - [ ] 独自のレスポンス作成を統一関数に変更
  - [ ] `successResponse`, `errorResponse` の使用
  - [ ] CORSヘッダーの統一

- [ ] **validation.ts系統の統合**
  - [ ] 独自のバリデーション処理を統一関数に変更
  - [ ] `validateFeedRequest` などの専用バリデーター使用

- [ ] **content.ts系統の統合**
  - [ ] コンテンツ取得処理の統一
  - [ ] `getVideoContent`、`searchVideosWithFilters` の使用

### 対象Edge Function一覧

1. **content/feed/index.ts** ✅
2. **content/recommendations/index.ts** ✅
3. **content/search/index.ts** ✅
4. **content/videos-feed/index.ts** ✅
5. **user-management/likes/index.ts** ✅
6. **user-management/embeddings/index.ts** ✅
7. **user-management/account/index.ts** ✅
8. **user-management/profile/index.ts** ✅
9. **recommendations/index.ts** 🔄
10. **recommendations/enhanced_two_tower/index.ts** 🔄
11. **likes/index.ts** 🔄
12. **delete_account/index.ts** 🔄
13. **update_user_embedding/index.ts** 🔄

## 互換性の注意事項

### Breaking Changes
1. **AuthResult型の変更**: `user_id` フィールドの型が変更
2. **DatabaseResult型の統一**: 戻り値の構造が変更
3. **レスポンス形式の統一**: APIレスポンスの構造が統一

### 後方互換性の維持
- 既存のエクスポートは維持
- 段階的な移行をサポート
- 警告レベルでの非推奨機能通知

## パフォーマンス最適化

### 共有モジュールのキャッシュ
- Deno の ES モジュールキャッシュを活用
- 共通処理の重複排除
- メモリ使用量の削減

### 実装例
```typescript
// コンテンツキャッシュの活用
import { ContentCache } from "../_shared/content.ts";

const cacheKey = `videos_${userId}_${feedType}`;
let cachedResult = ContentCache.get(cacheKey);

if (!cachedResult) {
  cachedResult = await fetchVideos(userId, feedType);
  ContentCache.set(cacheKey, cachedResult, 300000); // 5分間キャッシュ
}
```

## テスト戦略

### 単体テスト
- 各共有モジュールの個別テスト
- Mock を使用した統合テスト

### 統合テスト
- Edge Function 全体の動作テスト
- データベース接続テスト
- 認証フローテスト

### 実装例
```typescript
// tests/shared/auth.test.ts
import { assertEquals } from "https://deno.land/std@0.190.0/testing/asserts.ts";
import { authenticateUser } from "../../supabase/functions/_shared/auth.ts";

Deno.test("authenticateUser should validate JWT token", async () => {
  const mockRequest = new Request("https://example.com", {
    headers: { "Authorization": "Bearer valid-jwt-token" }
  });

  const result = await authenticateUser(mockRequest);
  assertEquals(result.authenticated, true);
});
```

## 監視・ログ

### 標準化されたログ形式
```typescript
import { logger } from "../_shared/monitoring.ts";

logger.info("User authenticated successfully", {
  user_id: user.user_id,
  function_name: "recommendations",
  duration_ms: processingTime
});
```

### パフォーマンス監視
```typescript
import { measurePerformance } from "../_shared/content.ts";

const { result, metrics } = await measurePerformance(
  "video_recommendation",
  async () => await generateRecommendations(userId)
);
```

## 移行完了の確認

### 成功基準
1. ✅ 全Edge Functionsが統一された共有モジュールを使用
2. ✅ コード重複が90%以上削減
3. ✅ レスポンス時間が維持または改善
4. ✅ エラー率が増加していない
5. ✅ 全ての統合テストがパス

### 移行後の利点
- **保守性向上**: 共通処理の一元管理
- **開発効率向上**: 再利用可能なコンポーネント
- **品質向上**: 統一された実装パターン
- **バグ削減**: 共通処理のテストカバレッジ向上
- **パフォーマンス向上**: 重複処理の削減