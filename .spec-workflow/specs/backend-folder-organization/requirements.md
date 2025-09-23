# バックエンドフォルダ組織化 - 要件仕様書（Supabase統合版）

## 1. 問題の定義

Adult Matching Applicationのバックエンドは現在、Supabaseとバックエンド間の責任範囲が不明確で、以下の問題を抱えています：

### 1.1 構造的問題
- **重複フォルダ**：ルートディレクトリ（`/home/devel/dev/adult_matching/`）とバックエンドディレクトリ間の重複
- **Supabase vs Backend混同**：`/supabase/functions/`（TypeScript Edge Functions）と`/backend/edge_functions/`（Python共有ユーティリティ）の役割混同
- **散在するPythonコード**：ML pipeline、data processing、scriptsが複数場所に分散
- **命名の混乱**：`/backend/edge_functions/`は実際にはPython共有ユーティリティ

### 1.2 現在の問題構造
```
/home/devel/dev/adult_matching/
├── ml_pipeline/           # 重複 - backend/ に統合すべき
├── models/                # 重複 - backend/ に統合すべき  
├── scripts/               # 重複 - backend/ に統合すべき
├── data_processing/       # 重複 - backend/ に統合すべき
├── supabase/              # 【分離維持】Supabase固有領域
│   ├── functions/         # TypeScript Edge Functions（25個）
│   ├── migrations/        # データベースマイグレーション
│   └── config.toml        # Supabase設定
└── backend/               # 【統合対象】Python領域
    ├── edge_functions/    # 【名前変更必要】実際は共有ユーティリティ
    ├── ml-pipeline/       # ML関連
    ├── data-processing/   # データ処理
    └── scripts/           # スクリプト
```

## 2. アーキテクチャ原則

### 2.1 責任分離原則
- **`/supabase/`領域**：Supabase固有機能（Edge Functions、マイグレーション、設定）
- **`/backend/`領域**：Python環境での開発・ML・データ処理
- **`/frontend/`領域**：Next.js React アプリケーション

### 2.2 技術スタック分離
- **Supabase Edge Functions**：TypeScript/Deno環境でのサーバーレス関数
- **Backend Python**：ML pipeline、データ処理、バッチ処理
- **フロントエンド**：Next.js/React/TypeScript

## 3. ビジネス要件

### BR-01: Supabase分離維持
**優先度**: 重要  
**説明**: Supabaseエコシステム固有の機能は`/supabase/`配下に分離維持し、Supabaseデプロイメントとの整合性を保つ。

**受入条件**:
- `/supabase/functions/` - TypeScript Edge Functionsが適切に分離されている
- `/supabase/migrations/` - データベースマイグレーションが保持されている  
- `/supabase/config.toml` - Supabase設定が維持されている
- Supabase固有ファイルが他の場所に移動されていない

### BR-02: Python Backend統合
**優先度**: 重要  
**説明**: すべてのPython関連コード（ML、データ処理、共有ユーティリティ、スクリプト）を`/backend/`配下に統合する。

**受入条件**:
- ルートディレクトリのPython関連フォルダがすべて`/backend/`に統合される
- `/backend/edge_functions/` → `/backend/shared/` に名前変更される
- MLモデル、データ処理パイプラインがすべて`/backend/`配下に配置される

### BR-03: 重複除去と役割明確化
**優先度**: 重要  
**説明**: Supabase Edge FunctionsとPython共有ユーティリティの混同を解消し、重複を除去する。

**受入条件**:
- TypeScript Edge Functions（`/supabase/functions/`）とPython共有ユーティリティ（`/backend/shared/`）が明確に分離される
- 機能的重複が存在しない
- 各コンポーネントの役割が明確に定義される

### BR-04: 統合後の動作保証
**優先度**: 高  
**説明**: 再編成後もSupabase Edge Functions、Python ML pipeline、フロントエンドがすべて正常動作する。

**受入条件**:
- Supabase Edge Functionsが正常デプロイ・実行される
- Python MLパイプラインが正常動作する
- フロントエンドからの API呼び出しが正常動作する
- データベース接続とマイグレーションが正常実行される

## 4. 機能要件

### FR-01: Supabase vs Backend分析
**説明**: Supabase固有機能とPython backend機能を明確に分析・分離する。

**Supabase領域（分離維持）**:
```
/supabase/
├── functions/                    # TypeScript Edge Functions
│   ├── content/                 # コンテンツ管理API
│   ├── recommendations/         # 推薦API
│   ├── user-management/         # ユーザー管理API
│   ├── dmm_sync/               # DMM API同期
│   └── _shared/                # TypeScript共有ユーティリティ
├── migrations/                  # データベースマイグレーション
├── config.toml                 # Supabase設定
└── README_CONFIG.md            # Supabase設定ドキュメント
```

**Backend領域（統合対象）**:
```
/backend/
├── shared/                     # Python共有ユーティリティ（旧edge_functions）
│   ├── auth.py                 # 認証ヘルパー
│   ├── database.py             # データベースヘルパー
│   ├── cache.py                # キャッシュユーティリティ
│   ├── embedding.py            # エンベディングユーティリティ
│   └── validation.py           # バリデーション
├── ml_pipeline/                # 機械学習パイプライン
│   ├── training/               # モデル訓練
│   ├── inference/              # 推論エンジン
│   ├── preprocessing/          # 前処理
│   └── evaluation/             # 評価
├── data_processing/            # データ処理パイプライン
│   ├── scraping/               # Webスクレイピング
│   ├── cleaning/               # データクリーニング
│   └── validation/             # データ検証
├── scripts/                    # 各種スクリプト
│   ├── dmm-sync/               # DMM API同期スクリプト
│   ├── data-migration/         # データマイグレーション
│   └── deployment/             # デプロイメント
├── models/                     # MLモデル成果物
├── config/                     # Backend設定
└── tests/                      # テストスイート
```

### FR-02: Edge Functions責任分離
**説明**: TypeScript Edge FunctionsとPython共有ユーティリティの明確な責任分離。

**TypeScript Edge Functions（`/supabase/functions/`）**:
- HTTPエンドポイントとしてのAPIサービス
- リアルタイムリクエスト処理
- Supabaseサービスとの直接統合
- ユーザー向けAPI提供

**Python共有ユーティリティ（`/backend/shared/`）**:
- バックエンド内部での共有ロジック
- MLパイプラインサポート機能
- データ処理支援ユーティリティ
- バッチ処理向けヘルパー関数

### FR-03: インポート・参照更新
**説明**: 再編成に伴うすべてのインポート、設定、参照パスの更新。

**更新対象**:
- Python内部インポート（`from backend.shared import ...`）
- Supabase設定ファイル参照
- テストファイル内参照
- ドキュメント内パス参照
- CI/CD設定ファイル

### FR-04: デプロイメント整合性
**説明**: Supabase Edge FunctionsとPython backendのそれぞれのデプロイメントプロセス維持。

**Supabaseデプロイメント**:
- `supabase functions deploy` コマンドでのEdge Functions配置
- マイグレーションファイルの自動適用
- 設定ファイルの環境別管理

**Backendデプロイメント**:
- Python環境でのパッケージ管理
- MLモデルファイルのバージョン管理
- バッチ処理スクリプトの配置

## 5. 非機能要件

### NFR-01: Supabase互換性維持
**説明**: Supabaseエコシステムとの完全な互換性維持。
- Edge Functions デプロイメントプロセスが変更されない
- Supabase CLI コマンドが正常動作する
- データベースマイグレーションが正常実行される

### NFR-02: Python環境統合性
**説明**: Python開発環境の統合性向上。
- 統一された依存関係管理
- 明確なモジュール構造
- 効率的なテスト実行

### NFR-03: 開発体験向上
**説明**: 開発者の作業効率向上。
- Supabase機能とPython機能の明確な区別
- 適切なIDE支援
- 明確なドキュメント構造

## 6. 技術要件

### TR-01: Supabase固有要件
**要件**:
- `/supabase/functions/` 配下のTypeScript Edge Functionsが変更されない
- `deno.json` 設定ファイルが適切に維持される
- Supabase プロジェクト設定との整合性が保たれる

### TR-02: Python統合要件
**要件**:
- `/backend/shared/` への名前変更に伴うインポート更新
- `pyproject.toml` での統合パッケージ管理
- `pytest` テストの正常実行

### TR-03: 相互連携要件
**要件**:
- TypeScript Edge Functionsからのデータベースアクセス
- Python scriptsからのSupabaseクライアント使用
- 共通データ形式での連携

## 7. 制約と前提

### 制約
- **Supabase デプロイメント**: Edge Functionsのデプロイメントプロセスを変更しない
- **API互換性**: 既存のAPIエンドポイントが影響を受けない
- **データベース**: マイグレーションとスキーマが維持される
- **Python互換性**: 既存のPythonスクリプトが動作し続ける

### 前提
- TypeScript Edge FunctionsとPython共有ユーティリティは異なる実行環境
- Supabase設定ファイルは環境管理に必須
- Python MLパイプラインは独立した実行環境を持つ
- 両システム間でのデータ交換は主にデータベース経由

## 8. 成功基準

### 主要成功指標
1. **責任分離達成**: Supabase機能とPython機能が明確に分離される
2. **重複除去完了**: 機能的重複が存在しない
3. **統合Python環境**: すべてのPython関連コードが`/backend/`配下に統合される
4. **動作保証**: Supabase、Python、フロントエンドがすべて正常動作する

### 技術指標
1. **Supabase デプロイメント**: `supabase functions deploy` が正常実行される
2. **Python テスト**: 全pytest テストがパスする
3. **API疎通**: フロントエンドからのAPI呼び出しが正常動作する
4. **MLパイプライン**: Two-Towerモデルの訓練・推論が正常動作する

## 9. 実装フェーズ

### Phase 1: 分析・計画
- 現在のSupabase vs Backend機能マッピング
- 依存関係分析
- 移行計画策定

### Phase 2: Python Backend統合
- ルートディレクトリのPython関連フォルダ統合
- `/backend/edge_functions/` → `/backend/shared/` 名前変更
- インポート文更新

### Phase 3: 検証・テスト
- Supabase Edge Functions動作確認
- Python MLパイプライン動作確認
- 統合テスト実行

### Phase 4: ドキュメント更新
- アーキテクチャドキュメント更新
- 開発ガイドライン更新
- デプロイメント手順更新

## 10. 受入条件まとめ

バックエンドフォルダ組織化は以下の条件が満たされた時点で完了とみなされます：

### Supabase領域
✅ `/supabase/functions/` のTypeScript Edge Functionsが正常動作する  
✅ `/supabase/migrations/` のマイグレーションが正常実行される  
✅ Supabaseデプロイメントプロセスが維持される  

### Backend領域  
✅ すべてのPython関連コードが `/backend/` 配下に統合される  
✅ `/backend/edge_functions/` が `/backend/shared/` に名前変更される  
✅ ルートディレクトリの重複フォルダが削除される  

### 統合検証
✅ 全Python テストスイートがエラーなしでパスする  
✅ MLパイプライン（Two-Tower）がエンドツーエンドで正常動作する  
✅ フロントエンドからのAPI呼び出しが正常動作する  
✅ DMM API同期スクリプトが正常動作する  

### ドキュメント
✅ アーキテクチャドキュメントが更新される  
✅ 開発ガイドラインが明確化される  
✅ デプロイメント手順が文書化される  

## 11. リスク軽減

### Supabase関連リスク
- **Edge Functions破損**: TypeScript Edge Functionsの変更による機能停止
- **マイグレーション失敗**: データベーススキーマの不整合
- **設定ファイル破損**: Supabase設定の不適切な変更

### Python Backend関連リスク
- **インポートエラー**: モジュール移動に伴うインポート破損
- **MLパイプライン停止**: モデルファイルパスの不整合
- **テスト失敗**: テスト環境設定の不備

### 軽減戦略
- **段階的実装**: 小規模バッチでの変更・検証
- **包括的バックアップ**: 変更前の完全プロジェクトバックアップ
- **継続的テスト**: 各段階での動作確認・テスト実行
- **ロールバック準備**: 問題発生時の迅速な復旧計画