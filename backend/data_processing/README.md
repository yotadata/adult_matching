# データ処理パイプライン

## 🎯 現在の成果物: local_compatible_data

**最重要フォルダ**: `local_compatible_data/` - Supabase互換のユーザーデータセット

### 📊 データ概要
- **ユーザー数**: 5人（認証済み疑似ユーザー）
- **動画数**: 40,036本（DMM APIデータ）
- **ユーザー判定**: 208件（Like: 160件、Nope: 48件）
- **処理日時**: 2025-09-27

### 📁 local_compatible_data構成
```
local_compatible_data/
├── profiles.json              # ユーザープロファイル（5ユーザー）
├── user_video_decisions.json  # ユーザー動画判定データ（208件）
├── videos_subset.json         # 対象動画サブセット（207本）
├── compatible_schema.sql      # Supabase互換スキーマ定義
├── import_data.sql           # データインポート用SQL
└── data_stats.json          # データ統計情報
```

### 🔄 データ変換プロセス
1. **元データ**: DMMレビューデータ（38,904件、50人のレビュアー）
2. **変換ロジック**:
   - 評価4-5 → "like"判定
   - 評価1-3 → "nope"判定
3. **マッチング**: content_id ↔ external_id（11.6%マッチ率）
4. **結果**: Supabase完全互換のローカルデータセット

### 🚀 使用方法
```bash
# PostgreSQLでスキーマ作成
psql -f local_compatible_data/compatible_schema.sql

# データインポート
psql -f local_compatible_data/import_data.sql
```

## 📦 アーカイブデータ

### archive/ - 過去の処理段階
- `raw_data/`: スクレイピングした生データ
- `processed_data/`: 統合・クリーニング済みデータ
- `converted_data/`: Supabase形式変換データ
- `scraping/`: データ収集スクリプト群
- `config/`: スクレイピング設定

### archive/utils/ - データ処理ツール
主要なデータ変換スクリプト群（開発完了済み）

## 📈 データ品質
- **DMM APIデータ**: 40,036本の動画（100%完全メタデータ）
- **レビューマッチング**: 3,742本の動画で判定データ利用可能
- **ユーザー認証**: Supabase Auth APIで実際のユーザー作成済み
- **データ整合性**: 外部キー制約、RLS対応

## 🎯 次のステップ
1. local_compatible_dataをSupabaseにアップロード
2. MLモデル訓練用データとして活用
3. レコメンデーションシステムのベースライン作成