# Supabase整合性チェッカー

汎用的なSupabaseプロジェクト整合性検証ツールです。フロントエンドとバックエンド間のAPI整合性、データベーススキーマ、認証パターンを包括的にチェックします。

## 🚀 特徴

- **汎用設計**: 任意のSupabaseプロジェクトで使用可能
- **包括的検証**: Edge Functions、RPC関数、API応答、認証を一括チェック
- **複数レポート形式**: JSON、Markdown、HTML形式でレポート出力
- **実環境テスト**: 実際のAPIエンドポイントを呼び出して検証
- **設定可能**: プロジェクトに応じた柔軟な設定

## 📦 構成

### コアツール

- `supabase-analyzer.js` - Supabase環境解析
- `edge-function-validator.js` - Edge Functions検証
- `database-schema-checker.js` - データベーススキーマ整合性チェック
- `api-response-validator.js` - API実環境検証

### 統合CLI

- `supabase-consistency-checker.js` - メインCLIツール

### 設定

- `scripts/config/supabase-consistency.config.js` - 設定ファイル

## 🛠️ セットアップ

### 1. 依存関係

```bash
npm install @ast-grep/napi magic-string zx
```

### 2. 環境変数設定

```bash
export SUPABASE_URL="your-supabase-url"
export SUPABASE_ANON_KEY="your-supabase-anon-key"
```

### 3. 設定ファイル

`scripts/config/supabase-consistency.config.js`を編集：

```javascript
export default {
    frontendPath: "frontend/src",
    backendPath: "supabase/functions",
    databasePath: "supabase/migrations",
    expectedFunctions: [
        "your-function-1",
        "your-function-2"
    ],
    expectedRpcFunctions: [
        "your_rpc_function_1",
        "your_rpc_function_2"
    ]
};
```

## 🚀 使用方法

### 基本実行

```bash
node scripts/supabase-consistency-checker.js
```

### 個別ツール実行

```bash
# Edge Functions検証のみ
node -e "
import { validateEdgeFunctions } from './scripts/utilities/edge-function-validator.js';
await validateEdgeFunctions();
"

# データベーススキーマチェックのみ
node -e "
import { checkDatabaseSchema } from './scripts/utilities/database-schema-checker.js';
await checkDatabaseSchema();
"
```

## 📊 レポート

実行後、`reports/`ディレクトリに以下のファイルが生成されます：

- `supabase-consistency-report.json` - 詳細JSON形式
- `supabase-consistency-summary.md` - Markdownサマリー
- `supabase-consistency-report.html` - HTMLレポート（設定で有効化時）

## 🔍 検証項目

### 1. Supabase環境解析

- Edge Functions定義の検出
- フロントエンドAPI呼び出しの抽出
- RPC関数使用箇所の特定
- 型定義の整合性確認
- 認証パターンの解析

### 2. Edge Functions検証

- HTTPメソッド対応確認
- レスポンス形式の統一性チェック
- エラーハンドリングの検証
- 認証処理の確認
- パフォーマンステスト

### 3. データベーススキーマ整合性

- RPC関数定義と使用箇所の照合
- パラメータ整合性の確認
- 未使用関数の検出
- 期待される関数の存在確認

### 4. API実環境検証

- 実際のエンドポイント呼び出し
- レスポンススキーマ検証
- 環境間整合性チェック
- エラーレスポンス形式の確認

## ⚙️ 設定オプション

| オプション | デフォルト値 | 説明 |
|-----------|-------------|------|
| `frontendPath` | `"frontend/src"` | フロントエンドコードディレクトリ |
| `backendPath` | `"supabase/functions"` | Edge Functionsディレクトリ |
| `databasePath` | `"supabase/migrations"` | マイグレーションファイルディレクトリ |
| `enableLiveTests` | `true` | 実環境APIテストの有効/無効 |
| `generateHtmlReport` | `true` | HTMLレポート生成の有効/無効 |
| `testTimeout` | `15000` | APIテストタイムアウト（ms） |
| `maxRetries` | `3` | 失敗時の最大リトライ回数 |

## 🔧 カスタマイズ

### 独自チェック追加

新しい検証ロジックを追加する場合：

1. `scripts/utilities/`に新しい検証クラスを作成
2. `supabase-consistency-checker.js`の`performAnalysis()`メソッドに統合
3. 設定ファイルに必要なオプションを追加

### 例：独自バリデーター

```javascript
// scripts/utilities/custom-validator.js
export class CustomValidator {
    async validate() {
        // 独自の検証ロジック
        return { issues: [], results: {} };
    }
}

// supabase-consistency-checker.jsに追加
import { CustomValidator } from './utilities/custom-validator.js';

// performAnalysis()内に追加
const customValidator = new CustomValidator(this.config);
this.results.custom = await customValidator.validate();
this.allIssues.push(...customValidator.issues);
```

## 🐛 トラブルシューティング

### よくある問題

**1. "Supabase URL または API Key が設定されていません"**
- 環境変数`SUPABASE_URL`と`SUPABASE_ANON_KEY`を確認
- 設定ファイルで直接指定も可能

**2. "マイグレーションディレクトリが見つかりません"**
- `databasePath`設定を確認
- プロジェクト構造に合わせてパス調整

**3. API呼び出しエラー**
- ネットワーク接続を確認
- Supabaseプロジェクトの稼働状況を確認
- APIキーの権限を確認

**4. パフォーマンス問題**
- `parallelExecution: false`に設定してシーケンシャル実行
- `testTimeout`を増加
- 不要な検証項目を無効化

## 📝 ライセンス

MITライセンス

## 🤝 コントリビューション

1. フォーク
2. フィーチャーブランチ作成
3. コミット
4. プルリクエスト作成

## 🔄 アップデート履歴

### v1.0.0
- 初回リリース
- 基本的な整合性チェック機能
- JSON/Markdown/HTMLレポート生成
- 汎用設計によるプロジェクト間再利用

---

*Generated by Supabase Consistency Checker - 汎用Supabaseプロジェクト整合性検証ツール*