# データ仕様書群

Adult Matching アプリケーションの全データ仕様・管理指針を定義する文書群

---

## 📚 仕様書カテゴリ

### 🗄️ [Database（データベース）](./database/)
PostgreSQL中心のデータベース設計・管理仕様
- **[schema.md](./database/schema.md)** - 完全スキーマ仕様書
- **[migrations.md](./database/migrations.md)** - マイグレーション管理指針  
- **[rls_policies.md](./database/rls_policies.md)** - Row Level Security設定

### 🕷️ [Scraped（スクレイピング）](./scraped/)
大規模レビューデータ収集・管理システム仕様
- **[reviews_format.md](./scraped/reviews_format.md)** - レビューデータ形式仕様
- **[batch_collection_process.md](./scraped/batch_collection_process.md)** - バッチ収集プロセス仕様
- **[data_quality_standards.md](./scraped/data_quality_standards.md)** - データ品質基準・検証

### 🔌 [API（API統合）](./api/)
外部API・内部API統合仕様
- **[dmm_fanza_integration.md](./api/dmm_fanza_integration.md)** - DMM/FANZA API仕様
- **[supabase_functions.md](./api/supabase_functions.md)** - Edge Functions仕様
- **[external_apis.md](./api/external_apis.md)** - 外部API統合仕様

### 🤖 [ML（機械学習）](./ml/)
機械学習パイプライン・データ仕様
- **[training_data_specs.md](./ml/training_data_specs.md)** - ML学習データ仕様
- **[pseudo_user_generation.md](./ml/pseudo_user_generation.md)** - 疑似ユーザー生成仕様
- **[embeddings_management.md](./ml/embeddings_management.md)** - 埋め込みベクトル管理
- **[model_artifacts.md](./ml/model_artifacts.md)** - モデルファイル管理

### 📄 [Formats（データ形式）](./formats/)
データ形式・構造・ファイルシステム仕様
- **[json_schemas.md](./formats/json_schemas.md)** - JSONファイル形式定義
- **[file_system_organization.md](./formats/file_system_organization.md)** - ファイルシステム構成
- **[data_flow_diagrams.md](./formats/data_flow_diagrams.md)** - データフロー仕様

---

## 🔄 データフロー概要

```
[DMM/FANZA API] ──── [PostgreSQL Database] ──── [ML Training]
                              │                      │
[Scraped Reviews] ──── [Content ID Linking] ────────┘
       │                      │
       └── [Pseudo User Generation] ───────────────────┘
                              │
[User Interactions] ──────────┘     │
                                    │
[Trained Models] ──────────────── [Recommendations]
```

### 主要データパス
1. **API動画取得** → PostgreSQL videos テーブル（**メインデータソース**）
2. **レビュースクレイピング** → **Content ID紐づけのみ** → 疑似ユーザー生成
3. **疑似ユーザー生成** → 評価ベース変換 → Two-Tower モデル学習
4. **推論・推薦** → 埋め込みベクトル → ユーザー体験

### 🎯 **重要なデータポリシー変更**
- **動画データソース**: API取得データのみを利用（レビューからの動画データは使用しない）
- **レビューデータ用途**: Content ID紐づけ + 疑似ユーザー生成のみ
- **学習データ**: API動画データ + レビュー由来疑似ユーザーデータの組み合わせ

---

## 📊 データ規模・統計 (2025年9月5日時点)

### スクレイピング済みデータ
- **レビュー数**: 38,904 件
- **対象動画数**: 32,304 件  
- **レビュワー数**: 50 名（トップレビュワー）
- **平均レビュー/動画**: 1.2 件
- **データサイズ**: 約 50MB (JSON)

### データベース格納データ
- **動画レコード**: 6 件（テスト用）
- **ユーザーライク**: 2 件
- **出演者**: 2 名
- **スキーマバージョン**: 最新 (2025-08-27)

### ML学習データ
- **疑似ユーザー**: 50 名分生成済み
- **Like率**: 77.7% (評価4.0+基準)
- **Skip率**: 22.3% (評価3.0以下基準)
- **Two-Towerモデル**: 学習完了

---

## 🛠️ データ管理コマンド

### 収集・処理
```bash
# 大規模データ収集
make collect-all                    # トップレビュワー→大規模収集
make integrate-batch-data          # バッチデータ統合・クリーニング
make generate-rating-users         # 評価ベース疑似ユーザー生成

# データ統計確認
make data-stats                    # データファイル統計
make collect-status                # 収集進捗確認
```

### データベース
```bash
# Supabase管理
supabase status                    # DB稼働状況
supabase db dump --local           # DBダンプ
supabase functions list            # Edge Functions確認
```

### ML処理
```bash
# モデル学習
make train-rating                  # 評価ベース学習
make train-full-rating             # フルパイプライン
```

---

## 🔍 仕様書の使い方

### 開発時
1. **該当カテゴリの仕様書を参照**（database/ 、ml/ など）
2. **データ形式・構造を確認** (formats/)
3. **品質基準・検証方法を確認** (scraped/data_quality_standards.md)

### 新機能追加時
1. **影響する仕様書を特定**
2. **データフロー変更を検討**
3. **関連仕様書を同期更新**

### トラブルシューティング時
1. **データフロー図で問題箇所を特定** (formats/data_flow_diagrams.md)
2. **該当システムの仕様書で詳細確認**
3. **データ品質基準と照合** (scraped/data_quality_standards.md)

---

## 📈 データ品質指標

### 基本品質
- **完全性**: 必須フィールドの充足率 95%以上
- **一意性**: 重複データ率 5%未満
- **整合性**: スキーマ適合率 99%以上
- **適時性**: データ更新遅延 24時間未満

### ML品質
- **バランス**: Like/Skip比 60-80% / 20-40%
- **多様性**: ジャンル分布偏差 20%未満  
- **代表性**: ユーザー行動分布の妥当性

---

## 🔐 データセキュリティ

### アクセス制御
- **Row Level Security** (RLS) 全テーブル適用
- **API Key管理** - 環境変数による分離
- **Cookie情報** - 暗号化保存・定期更新

### プライバシー
- **匿名化処理** - 個人特定情報の除去
- **データ保持期間** - 最大1年間
- **削除ポリシー** - ユーザー要求時の完全削除

---

## 📅 更新履歴

| 日付 | 更新内容 | 担当 |
|------|----------|------|
| 2025-09-05 | データ仕様書群の構造設計・作成 | Claude Code |
| 2025-09-03 | 大規模データ収集完了・ML学習実装 | Claude Code |
| 2025-08-27 | データベーススキーマ完成 | Development Team |

---

**文書管理**  
**最終更新**: 2025年9月5日  
**管理者**: Claude Code  
**レビュー頻度**: 週次  
**承認**: 必要時