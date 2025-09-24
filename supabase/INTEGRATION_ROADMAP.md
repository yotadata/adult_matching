# Edge Functions 統合ロードマップ

## 統合概要

**現在**: 11個のEdge Functions → **目標**: 6個の統合関数

## 統合マッピング

### 削除対象（2個）
```
❌ recommendations/                    → 削除（enhanced_two_tower_recommendationsに統合済み）
❌ two_tower_recommendations/          → 削除（64次元・旧実装）
```

### 保持・強化対象（1個）
```
⭐ enhanced_two_tower_recommendations/ → recommendations/enhanced_two_tower/ に再編成
```

### 統合対象（5個）
```
✅ likes/                             → user-management/likes/
✅ update_user_embedding/              → user-management/embeddings/
✅ delete_account/                     → user-management/account/

✅ feed_explore/                       → content/feed/
✅ videos-feed/                        → content/feed/ (統合)
```

### backend/移行対象（2個）
```
⚠️ dmm_sync/                          → backend/data/sync/dmm/
⚠️ update_embeddings/                  → backend/ml/preprocessing/embeddings/
```

### 強化対象（1個）
```
🔧 _shared/                           → 共通ライブラリ大幅強化
```

## 統合後の構造

```
supabase/functions/
├── recommendations/
│   └── enhanced_two_tower/           # メイン推薦システム
├── user-management/
│   ├── likes/                        # いいね管理
│   ├── embeddings/                   # ユーザー埋め込み更新
│   └── account/                      # アカウント管理
├── content/
│   └── feed/                         # 統合フィード（explore + videos）
└── _shared/
    ├── database/                     # DB接続・クエリ
    ├── auth/                         # 認証ヘルパー
    ├── validation/                   # 入力検証
    └── monitoring/                   # ログ・メトリクス
```

## 機能保持チェックリスト

### ✅ 推薦機能（完全保持）
- [x] 768次元Two-Tower推薦
- [x] コサイン類似度計算
- [x] 多様性強化
- [x] 推薦理由生成
- [x] ジャンル嗜好分析

### ✅ ユーザー管理（完全保持）
- [x] いいね追加・削除・取得
- [x] ユーザー埋め込み更新
- [x] アカウント削除機能

### ✅ コンテンツ配信（機能統合）
- [x] 探索的フィード（多様性重視）
- [x] 基本フィード機能
- [x] 新規ユーザー対応

### ⚠️ データ処理（backend移行）
- [x] DMM API同期（Node.js版使用）
- [x] 埋め込みバッチ更新

## API互換性マトリクス

| 旧エンドポイント | 新エンドポイント | 互換性 | 対応 |
|------------------|------------------|--------|------|
| `/recommendations` | `/recommendations/enhanced_two_tower` | 🔄 Redirect | プロキシ設定 |
| `/two_tower_recommendations` | `/recommendations/enhanced_two_tower` | 🔄 Redirect | プロキシ設定 |
| `/likes` | `/user-management/likes` | 🔄 Redirect | プロキシ設定 |
| `/update_user_embedding` | `/user-management/embeddings` | 🔄 Redirect | プロキシ設定 |
| `/delete_account` | `/user-management/account` | 🔄 Redirect | プロキシ設定 |
| `/feed_explore` | `/content/feed` | 🔄 Redirect | プロキシ設定 |
| `/videos-feed` | `/content/feed` | 🔄 Redirect | プロキシ設定 |

## リスク軽減策

### 1. 段階的移行
```
Phase 1: 新構造作成（旧構造並行稼働）
Phase 2: プロキシ設定でリダイレクト
Phase 3: フロントエンド更新
Phase 4: 旧関数削除
```

### 2. 互換性保持
```typescript
// プロキシ関数例
export default async function handler(req: Request) {
  // 旧エンドポイントへのリクエストを新エンドポイントにリダイレクト
  const url = new URL(req.url);
  url.pathname = '/functions/v1/recommendations/enhanced_two_tower';
  
  return fetch(url, req);
}
```

### 3. テスト戦略
```
- 統合前後の機能テスト
- API互換性テスト
- パフォーマンス回帰テスト
- エンドツーエンドテスト
```

## 実装優先順位

### 高優先度（Week 1-2）
1. **enhanced_two_tower_recommendations 強化**
2. **user-management グループ作成**
3. **content フィード統合**

### 中優先度（Week 3）
4. **_shared ライブラリ強化**
5. **プロキシ設定実装**

### 低優先度（Week 4）
6. **backend/ 移行**
7. **旧関数削除**
8. **最終最適化**

## 成功指標

### パフォーマンス
- [ ] 応答時間 <500ms 維持
- [ ] エラー率 <1% 維持
- [ ] スループット向上 >20%

### 保守性
- [ ] コード重複 削減 >50%
- [ ] 関数数削減 11個→6個
- [ ] 共通ライブラリ利用率 >80%

### 安定性
- [ ] 機能回帰 0件
- [ ] API互換性 100%
- [ ] データ損失 0件

この統合により、システム全体の保守性と性能が大幅に向上します。