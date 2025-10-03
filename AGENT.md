# エージェントメモ

## プロジェクト概要
- AVレコメンド / マッチングサービスのPoC。FANZA（DMM）由来の動画メタデータとユーザー行動ログを元にスワイプUIでレコメンドを提示する。
- フロントエンドはNext.js 15 + React 19で実装。Supabase Auth/Database/Functionsと連携し、動画フィードはEdge Function経由で取得する。
- データ層はSupabase/PostgreSQLを前提に、`backend/data_processing/local_compatible_data`で生成した疑似ユーザー・動画データを利用する。
- 将来的にMLレコメンド（`supabase/functions/ai-recommend*`）やデータパイプライン強化を想定しており、`docs/openapi.yaml`にAPI仕様、`docs/ui_design.md`にUI要件がまとまっている。

## ディレクトリと役割
- `frontend/` : Next.jsアプリ本体。スワイプUIや動画詳細モーダル、Supabaseクライアント初期化など。
- `supabase/` : Supabase CLIプロジェクト。Edge Function (`functions/videos-feed` 等)、DBマイグレーション、`config.toml`。
- `backend/data_processing/` : DMMデータ → Supabase互換データへの変換パイプラインと成果物 (`local_compatible_data/`)。
- `scripts/` : Supabaseへデータ投入・アフィリエイトリンク整備用のNodeスクリプト群。
- `docs/` : API仕様（Redoc用`openapi.yaml`）、UIデザイン要件、バッチジョブ定義などの資料。

## 技術スタックと開発要件
- Node.js 20.x推奨（Next.js 15とSupabase JS 2系の推奨バージョン帯）。npmを使用（`package-lock.json` 管理）。
- Deno (>=1.40) + Supabase CLI (>=1.185) を利用してEdge Functionsをローカル実行。
- Supabaseローカル環境は Postgres 17 を想定（`supabase/config.toml`）。マイグレーションは `supabase/migrations/` に配置。
- フロントのスタイリングはTailwind CSS、UI操作はFramer Motion / Headless UIを使用。
- データ処理スクリプトはTypeScript(Node)ベースで`.env.remote`を読み込む設計。

## 環境変数（主要）
- フロント/共通: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_FANZA_AFFILIATE_ID`, `NEXT_PUBLIC_GUEST_DECISIONS_LIMIT`, `NEXT_PUBLIC_PERSONALIZE_TARGET`, `NEXT_PUBLIC_DIAGNOSIS_TARGET`。
- サーバーサイド/サービスロール: `SUPABASE_SERVICE_ROLE_KEY` (もしくは `SUPABASE_SERVICE_KEY`)。
- データ取得スクリプト: `FANZA_API_ID`, `FANZA_API_AFFILIATE_ID`, `FANZA_LINK_AFFILIATE_ID`（後方互換で `FANZA_AFFILIATE_ID` を許容）。
- その他: ローカルでSupabaseを動かす際は `supabase/.env` にJWTシークレット等を設定（Supabase CLIが自動生成）。

## ローカル開発フロー
1. ルートで `npm install`（scripts用）を実行。フロント用に `cd frontend && npm install`。
2. Supabaseローカル: `supabase start` でDBとEdge Functions環境を起動。必要に応じ `supabase db reset` / `supabase db push` でスキーマ同期。
3. フロント起動: `cd frontend && npm run dev`。Supabase Edge Functionをローカルで使う場合は別ターミナルで `supabase functions serve --no-verify-jwt` を実行。
4. 疑似データ導入: `backend/data_processing/local_compatible_data` の `compatible_schema.sql` と `import_data.sql` を使い、Supabase(Postgres)に投入。
5. 追加データ取得が必要な場合は `node --loader ts-node/esm scripts/ingestFanzaData.ts` を実行（`.env.remote` に各種FANZA APIキーを配置）。

## 動作確認・テスト
- フロント lint: `cd frontend && npm run lint`。
- UI確認: `npm run dev` 実行後に `http://localhost:3000` へアクセス。ゲスト判定はローカルStorage、ログイン後は`user_video_decisions`テーブルに書き込まれる。
- Supabase Edge Functions: `supabase functions serve` で `videos-feed` などを確認。エンドポイントは `http://localhost:54321/functions/v1/videos-feed`。
- Denoテスト雛形: `deno test supabase/functions/videos-feed/index.test.ts`（現状`index.ts`が`serve`をエクスポートしていないため、実行前にエクスポート補完が必要）。
- データ整合性: Supabase Studioまたは `psql` で `videos`, `user_video_decisions`, `tags` 等の外部キーを確認。
- アフィリエイトリンク整備: `node scripts/backfillAffiliateLinks.mjs` 実行で `videos.product_url` を一括整形。

## 参考資料
- API仕様: `docs/openapi.yaml`（Redocで閲覧可能）
- UI設計: `docs/ui_design.md`
- データ概要: `backend/data_processing/README.md`
- Supabase設定: `supabase/config.toml`

開発時は上記情報を常に参照し、要件や環境を変更した場合は本ファイルと関連ドキュメントを更新すること。

## Two-Tower v2 運用メモ
- Python学習パイプライン: `backend/ml/two_tower_v2` に収容。`train.py` → `export.py` → `generate_embeddings.py` の順で利用する。
- 一括実行スクリプト: `backend/ml/two_tower_v2/run_local_pipeline.py` を実行すると上記3ステップが連続で走る。
- 成果物は `artifacts/` 以下に出力し、CDN（例: Supabase Storage）へ配置。Edge Functions は `MODEL_ARTIFACT_BASE_URL` から `user_tower.onnx` / `feature_schema.json` 等を取得する。
- Supabase RPC: `recommend_videos_ann`（HNSW×cosine）。`videos.safety_level`・`videos.region_codes`・`video_tags` をwhere句で制御。
- Edge Functions:
  - `embed-user`: `user_id` を受け取り、サービスロール経由で履歴を取得 → 埋め込み生成・`user_embeddings` に保存。
  - `ai-recommend-v5`: `user_id` とフィルタを受け、埋め込み参照（必要なら生成）→ `recommend_videos_ann` RPC → リランキングしてレスポンス。
  - `recommend`: ユーザー埋め込み → `recommend_videos_ann` RPC → 簡易リランキングしてレスポンス。
- WASMアーティファクト（`ort-wasm-*.wasm`）もCDNに配置し、Edge側の `onnxruntime-web` 初期化で利用すること。
- GitHub Actions `train-two-tower.yml` でパイプライン実行・成果物配布・Supabase反映まで自動化可能（Secrets設定必須）。

## 動作確認チェックリスト
1. `python src/train.py ...` 実行で `checkpoints/latest.pt` が生成されること。
2. `python src/export.py ...` 後、`artifacts/user_tower.onnx` と各種JSONが作成されること。
3. `python src/generate_embeddings.py ...` で `video_embeddings.parquet` が出力され、pgvectorへのアップサートに利用できること。
4. Supabaseローカルで `supabase db reset` → 新規マイグレーション適用後、`recommend_videos_ann` RPC が呼び出せること。
5. Edge Functions (`supabase functions serve embed-user` / `ai-recommend-v5`) を起動し、`curl` で埋め込み生成と推薦結果が得られること。
6. CDNに成果物を配置した上で `.env` に `MODEL_ARTIFACT_BASE_URL` を設定し、本番のEdge Functionsが起動できること。
