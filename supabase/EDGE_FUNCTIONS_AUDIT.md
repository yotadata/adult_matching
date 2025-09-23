# Edge Functions 監査・分析レポート

## 概要

現在のSupabase Edge Functions（11個）の包括的監査を実施し、統合・最適化計画を策定します。

## 現在のEdge Functions一覧

### 1. 推薦システム関連（3個）

#### enhanced_two_tower_recommendations ⭐ **[保持・強化]**
- **目的**: 最新の768次元Two-Tower推薦システム
- **主要機能**:
  - 高度なユーザー埋め込み生成
  - バッチ類似度計算
  - 多様性強化アルゴリズム
  - 推薦理由生成
  - ジャンル嗜好分析
- **依存関係**: user_embeddings, video_embeddings テーブル
- **状態**: 🟢 **最新・活用中**

#### recommendations ❌ **[削除対象]**
- **目的**: 基本的な推薦システム（旧実装）
- **主要機能**:
  - 基本的なコサイン類似度計算
  - 多様性選択
  - 推薦理由生成
- **問題**: enhanced_two_tower_recommendations と機能重複
- **状態**: 🔴 **冗長・削除予定**

#### two_tower_recommendations ❌ **[削除対象]**
- **目的**: 簡易Two-Tower実装（64次元）
- **主要機能**:
  - ドット積計算
  - 基本的なTwo-Tower推論
  - 特徴量計算
- **問題**: 64次元実装（現在は768次元が標準）
- **状態**: 🔴 **旧実装・削除予定**

### 2. ユーザー管理関連（3個）

#### likes ✅ **[user-management/に移動]**
- **目的**: いいね機能の管理
- **主要機能**:
  - いいね追加・削除
  - いいね一覧取得
  - レスポンス形式統一
- **依存関係**: likes テーブル、認証
- **状態**: 🟡 **移動・再編成対象**

#### update_user_embedding ✅ **[user-management/に移動]**
- **目的**: ユーザー埋め込みの更新
- **主要機能**:
  - ユーザー特徴量抽出
  - 埋め込み生成
  - データベース更新
- **依存関係**: user_embeddings テーブル
- **状態**: 🟡 **移動・再編成対象**

#### delete_account ✅ **[user-management/に移動]**
- **目的**: アカウント削除機能
- **主要機能**:
  - 関連データの削除
  - 認証情報削除
- **依存関係**: 複数テーブル、認証
- **状態**: 🟡 **移動・再編成対象**

### 3. コンテンツ配信関連（2個）

#### feed_explore ✅ **[content/に統合]**
- **目的**: 探索的フィード生成
- **主要機能**:
  - 多様性スコア計算
  - 新規ユーザー向けフィード
- **依存関係**: videos テーブル
- **状態**: 🟡 **統合対象**

#### videos-feed ✅ **[content/に統合]**
- **目的**: 基本的なビデオフィード
- **主要機能**:
  - 基本的なフィード生成
- **依存関係**: videos テーブル
- **状態**: 🟡 **統合対象**

### 4. データ同期・処理関連（2個）

#### dmm_sync ⚠️ **[backend/data/に移行]**
- **目的**: DMM API同期
- **主要機能**:
  - DMM APIデータ取得
  - データベース挿入
- **問題**: Node.js版(real_dmm_sync.js)の方が安定
- **状態**: 🟠 **backend移行対象**

#### update_embeddings ⚠️ **[backend/ml/に移行]**
- **目的**: 埋め込みバッチ更新
- **主要機能**:
  - ユーザー・ビデオ埋め込み更新
  - バッチ処理機能
- **依存関係**: 複数埋め込みテーブル
- **状態**: 🟠 **backend移行対象**

### 5. 共有ユーティリティ（1個）

#### _shared ✅ **[強化・拡張]**
- **目的**: 共有型定義とユーティリティ
- **主要機能**:
  - 共通型定義（ApiResponse, VideoData等）
  - CORSヘッダー設定
- **状態**: 🟢 **強化対象**

## 重複機能分析

### 1. 推薦システムの重複

| 関数 | 実装方式 | 次元数 | 状態 | 対応 |
|------|----------|--------|------|------|
| enhanced_two_tower_recommendations | 高度Two-Tower | 768次元 | 最新 | **保持** |
| recommendations | 基本コサイン類似度 | N/A | 旧式 | **削除** |
| two_tower_recommendations | 簡易Two-Tower | 64次元 | 旧式 | **削除** |

### 2. 埋め込み更新の重複

| 関数 | 対象 | 処理方式 | 対応 |
|------|------|----------|------|
| update_user_embedding | ユーザーのみ | 単体更新 | **user-management/へ移動** |
| update_embeddings | ユーザー・ビデオ | バッチ更新 | **backend/ml/へ移行** |

### 3. フィード生成の重複

| 関数 | 特徴 | 対応 |
|------|------|------|
| feed_explore | 多様性重視 | **content/へ統合** |
| videos-feed | 基本的 | **content/へ統合** |

## 統合計画

### フェーズ1: 機能統合

#### 推薦システム統合
```
supabase/functions/recommendations/
└── enhanced_two_tower/
    └── index.ts  # enhanced_two_tower_recommendations のみ保持
```

#### ユーザー管理統合
```
supabase/functions/user-management/
├── likes/
│   └── index.ts
├── embeddings/
│   └── index.ts  # update_user_embedding
└── account/
    └── index.ts  # delete_account
```

#### コンテンツ統合
```
supabase/functions/content/
└── feed/
    └── index.ts  # feed_explore + videos-feed 統合
```

### フェーズ2: backend/移行

#### データ同期移行
```
backend/data/sync/dmm/
└── dmm_sync.js  # real_dmm_sync.js を標準化
```

#### ML処理移行
```
backend/ml/preprocessing/embeddings/
└── batch_update.ts  # update_embeddings 移行
```

### フェーズ3: 共有ライブラリ強化

#### 強化された_shared
```
supabase/functions/_shared/
├── database/
│   ├── connection.ts
│   └── queries.ts
├── auth/
│   └── helpers.ts
├── validation/
│   └── schemas.ts
└── monitoring/
    └── logging.ts
```

## 機能損失防止チェック

### 削除対象の機能確認

#### recommendations 削除影響
- ✅ **コサイン類似度**: enhanced_two_tower_recommendations でカバー
- ✅ **多様性選択**: enhanced_two_tower_recommendations でカバー
- ✅ **推薦理由**: enhanced_two_tower_recommendations でカバー

#### two_tower_recommendations 削除影響
- ✅ **Two-Tower推論**: enhanced_two_tower_recommendations (768次元) でカバー
- ✅ **特徴量計算**: enhanced_two_tower_recommendations でカバー
- ⚠️ **64次元サポート**: 廃止（768次元に統一）

### 移行時の互換性確保

#### API エンドポイント互換性
```typescript
// 既存のAPIコントラクトを維持
POST /functions/v1/recommendations → /functions/v1/recommendations/enhanced_two_tower
POST /functions/v1/likes → /functions/v1/user-management/likes
POST /functions/v1/feed_explore → /functions/v1/content/feed
```

#### リクエスト・レスポンス形式
- 全ての型定義を_shared/types.tsで管理
- 後方互換性を保持
- 段階的移行により影響最小化

## 最適化機会

### 1. コード重複削除
- **CORS設定**: 全関数で共通化
- **認証処理**: 共通ミドルウェア化
- **エラーハンドリング**: 統一パターン
- **ログ記録**: 共通ログ関数

### 2. パフォーマンス向上
- **接続プール**: データベース接続の最適化
- **キャッシュ戦略**: 頻繁なクエリのキャッシュ
- **バッチ処理**: 複数リクエストの効率化

### 3. 監視・運用性
- **統一ログ形式**: 構造化ログ
- **メトリクス収集**: パフォーマンス測定
- **エラー追跡**: 統一エラー管理

## 移行スケジュール

### Week 1: 準備・分析
- [ ] 依存関係詳細調査
- [ ] テストケース作成
- [ ] 移行スクリプト準備

### Week 2: 統合実装
- [ ] enhanced_two_tower_recommendations 強化
- [ ] user-management/ グループ作成
- [ ] content/ グループ作成

### Week 3: backend移行
- [ ] data/sync/ 移行
- [ ] ml/preprocessing/ 移行
- [ ] 旧関数削除

### Week 4: 最適化・検証
- [ ] _shared 強化
- [ ] パフォーマンステスト
- [ ] 本番デプロイ

## リスク評価

### 高リスク
- ❗ **API互換性**: フロントエンド影響
- ❗ **データ整合性**: 移行中のデータ損失
- ❗ **パフォーマンス**: 統合による性能劣化

### 中リスク
- ⚠️ **依存関係**: 未発見の依存関係
- ⚠️ **テスト不足**: エッジケース見落とし

### 低リスク
- ✅ **文書化**: 不十分な文書化
- ✅ **監視**: 一時的な監視空白

## 結論

**統合により11個→6個のEdge Functionsに削減**
- **保持**: enhanced_two_tower_recommendations（強化）
- **統合**: user-management（3個）, content（1個）
- **移行**: 2個をbackend/へ
- **削除**: 2個の冗長関数
- **強化**: _shared共通ライブラリ

この統合により、コード重複削減、保守性向上、パフォーマンス最適化を実現します。