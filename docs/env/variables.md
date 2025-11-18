# 環境変数一覧（共通参照）

フロントエンド、バックエンド（Docker スクリプトを含む）、GitHub Actions で利用する主要な環境変数をまとめました。  
各カテゴリの **設定場所** 列を参考に、`.env` / Secrets を統一的に管理してください。

## 1. Supabase 共通

| 変数名 | 説明 | 主な利用先 | 推奨設定場所 |
| --- | --- | --- | --- |
| `SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_URL` | Supabase プロジェクト URL（`https://<project-ref>.supabase.co`） | フロント, バックエンド, GHA | `docker/env/*.env`, `frontend/.env.local`, GitHub Secrets |
| `SUPABASE_ANON_KEY` / `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon key（公開可） | フロント, CLI | `frontend/.env.local`, GitHub Secrets |
| `SUPABASE_SERVICE_ROLE_KEY` | Service Role key（非公開） | Docker スクリプト, GHA | `docker/env/prd.env`, GitHub Secrets |
| `SUPABASE_SERVICE_ROLE_JWT_AUDIENCE` | Supabase JS audience（多くは `authenticated`） | フロント, バックエンド | `docker/env/*.env`, `frontend/.env.local` |
| `REMOTE_DATABASE_URL` | 本番 Postgres 接続文字列（`sslmode=require` 推奨） | Docker スクリプト, GHA | `docker/env/prd.env`, GitHub Secrets |
| `SUPABASE_DB_URL` | Postgres 接続文字列のエイリアス（未指定でも `REMOTE_DATABASE_URL` から補完） | Docker スクリプト | `docker/env/*.env` |
| `SUPABASE_PROJECT_ID` (旧: `SUPABASE_PROJECT_REF`) | Supabase プロジェクトの ref（CLI/API の project-id と同値） | Docker スクリプト, GHA | `docker/env/*.env`, GitHub Secrets |
| `SUPABASE_REGION` | Supabase リージョン（例: `aws-1-ap-northeast-1`） | Docker スクリプト, GHA | `docker/env/*.env`, GitHub Secrets |
| `SUPABASE_POOLER_HOST` / `SUPABASE_POOLER_PORT` | プール接続先を固定したい場合に指定（未指定なら自動解決） | Docker スクリプト | 任意 |
| `PGSSLMODE` / `PGSSLROOTCERT` | SSL 接続用設定（GHA 等で `require` / `/etc/ssl/certs/ca-certificates.crt` を推奨） | Docker スクリプト, GHA | `docker/env/*.env`, GitHub Secrets |

## 2. FANZA / 取得系

| 変数名 | 説明 | 主な利用先 | 推奨設定場所 |
| --- | --- | --- | --- |
| `FANZA_API_ID` | DMM/FANZA API のアプリケーション ID | `scripts/ingest_fanza` | `docker/env/*.env`, GitHub Secrets |
| `FANZA_API_AFFILIATE_ID` | API 用アフィリエイト ID（動画取得時に使用） | `scripts/ingest_fanza` | `docker/env/*.env`, GitHub Secrets |
| `FANZA_LINK_AFFILIATE_ID` | リンク生成用アフィリエイト ID（`product_url` 作成で利用） | `scripts/ingest_fanza` | `docker/env/*.env`, GitHub Secrets |
| `GTE_RELEASE_DATE` / `LTE_RELEASE_DATE` | FANZA 取得の期間指定。手動実行時に上書きする | `scripts/ingest_fanza`, `scripts/sync_video_embeddings` | 実行時にエクスポート |
| `SYNC_VIDEO_LOOKBACK_DAYS` | 差分同期のデフォルト対象日数（既定: 3） | `scripts/sync_video_embeddings`, GHA | `docker/env/*.env`, GitHub Secrets |

## 3. 推薦・ML スクリプト

| 変数名 | 説明 | 主な利用先 | 推奨設定場所 |
| --- | --- | --- | --- |
| `SUPABASE_PROJECT_ID` / `SUPABASE_PROJECT_REF` / `SUPABASE_REGION` | プール接続補助（上段と共通） | `prep_two_tower`, `gen_user_embeddings`, `gen_video_embeddings` | `docker/env/*.env` |
| `INGEST_FANZA_ENV_FILE`, `FANZA_BACKFILL_ENV_FILE`, `SYNC_VIDEO_ENV_FILE`, `UPSERT_TT_ENV_FILE`, `GEN_USER_ENV_FILE` など | 各スクリプトの `--env-file` オーバーライド用。省略時は `docker/env/prd.env` を参照 | Docker スクリプト | 実行時に指定 |
| `SYNC_USER_ENV_FILE`, `SYNC_USER_OUTPUT_ROOT`, `SYNC_USER_RECENT_HOURS` | ユーザー埋め込み増分同期（`scripts/sync_user_embeddings`）向けのデフォルト設定 | Docker スクリプト, GHA | 実行時に指定 |
| `SUPABASE_DB_DISABLE_SSL` | ローカル Postgres など SSL 非対応環境で `gen_user_embeddings` / `update_user_embeddings` からの接続時に SSL を無効化（`1` で無効） | `gen_user_embeddings`, `update_user_embeddings`, `sync_user_embeddings` | `.env` / 実行時に指定 |
| `POSTGRES_KEEP_CONTAINER`, `LOCAL_DB_STARTUP_TIMEOUT`, `POSTGRES_PLATFORM` | 一部スクリプトで Postgres コンテナを制御するためのオプション | `scripts/prep_two_tower/run_with_remote_db.sh` など | 実行時に必要なら設定 |

## 4. GitHub Actions 用 Secrets チェックリスト

GitHub リポジトリの Secrets / Variables として、少なくとも以下を登録しておくと各ワークフロー（モデル学習・埋め込み同期・フロントデプロイ等）で再利用できます。

| Secret 名 | 推奨値 |
| --- | --- |
| `SUPABASE_URL` | 上記 `SUPABASE_URL` と同値 |
| `SUPABASE_ANON_KEY` | anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | service role key |
| `REMOTE_DATABASE_URL` | SSL 必須の Postgres 接続文字列 |
| `SUPABASE_PROJECT_ID` | プロジェクト ref |
| `SUPABASE_REGION` | リージョン名 |
| `SUPABASE_ACCESS_TOKEN` | Supabase CLI 用 Personal Access Token（`deploy-db`, `deploy-functions`, `release` で利用） |
| `SUPABASE_DB_PASSWORD` | Supabase DB パスワード（`supabase db remote connect` 等で利用） |
| `FANZA_API_ID` | FANZA API ID |
| `FANZA_API_AFFILIATE_ID` | FANZA API アフィリエイト ID |
| `FANZA_LINK_AFFILIATE_ID` | FANZA リンクアフィリエイト ID |
| `VERCEL_TOKEN` | Vercel CLI トークン（`deploy-frontend` で利用） |
| `VERCEL_ORG_ID` | Vercel 組織 ID（`deploy-frontend` で利用） |
| `VERCEL_PROJECT_ID` | Vercel プロジェクト ID（`deploy-frontend` で利用） |

必要に応じて `PGSSLMODE`, `PGSSLROOTCERT`, `SYNC_VIDEO_LOOKBACK_DAYS` なども Secrets として登録すると、ワークフローから直接参照できます。各ワークフローごとの詳細な入力や追加 Secrets は `docs/env/github_actions.md` を参照してください。

---

**メモ:**  
- プロジェクト内の `.env` は `docker/env/` に集約し、ローカル確認は `dev.env`、本番相当は `prd.env` を参照する運用を推奨します。  
- Supabase 関連では `SUPABASE_PROJECT_ID` と `SUPABASE_PROJECT_REF` を同義として扱っています（レガシーなスクリプトは `..._REF` を参照しますが、内部で `..._ID` から自動補完されます）。
- フロントエンドの追加環境変数が必要になった場合は `frontend/.env.local` とこのドキュメントを併せて更新してください。  
- 新しいスクリプトやパイプラインを追加した際は、本表に変数定義を追記すると管理が楽になります。
