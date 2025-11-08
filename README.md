# Adult Matching — 開発ガイド（Docker）

本リポジトリは Next.js フロントエンド、Supabase（DB/Studio/Edge Functions）、および ML（学習/前処理）で構成されます。開発・実行は Docker 前提です。

## ディレクトリ構成

- `docker/`
  - `compose.yml`: フロントエンド + Supabase + Edge Functions の起動定義
  - `frontend.dev.Dockerfile`: フロント開発用イメージ
  - `env/dev.env`: 開発用の環境変数（実値を設定して使用）
  - `env/prd.env.example`: 本番用の雛形
- `frontend/`: Next.js 15 + React 19 のアプリ本体
- `supabase/`: Supabase 設定・マイグレーション・Edge Functions（`supabase/functions/*`）
- `scripts/`: スクリプトごとの専用フォルダ（Dockerfile + run.sh + スクリプト本体）
  - `prep_two_tower/`: 学習データ前処理（Parquet 生成）
  - `train_two_tower/`: Two-Tower 学習
  - `scrape_dmm_reviews/`: DMM レビューの収集
  - `ingest_fanza/`: FANZA API からの動画取り込み（TS）
  - `sync_remote_db/`: リモート DB → ローカル DB 同期（Supabase CLI）
- `ml/`
  - `data/`: 入力データ置き場（CSV/Parquet など）
  - `artifacts/`: 学習成果物（ONNX/pt/メタデータ など）
- `docs/`: ドキュメント（API 仕様、学習ガイド、Docker 手順など）
- `AGENTS.md`: エージェント運用ルール

## 事前準備

1) Docker/Docker Compose をインストール
2) 環境変数ファイルを作成（self-hosted Supabase 用の必須キーを含む）

```bash
cp docker/env/dev.env docker/env/dev.env.local  # 例: バックアップ
# docker/env/dev.env に実値を設定（下記の必須キーを参照）
## 自動生成も可能（開発用）
# 直接 dev.env を上書き/追記する場合（推奨）
bash scripts/setup_dev_env/run.sh --write

# 生成内容をファイルで確認したい場合
# bash scripts/setup_dev_env/run.sh  # docker/env/dev.env.generated に出力（Realtime の SECRET_KEY_BASE も含む）
```

必須キー（dev.env）
- `POSTGRES_PASSWORD`: 任意の強固な文字列
- `SUPABASE_JWT_SECRET`: JWT 署名用の秘密鍵（十分長いランダム文字列）
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`: 上記秘密鍵で署名した anon ロールの JWT
- `SUPABASE_SERVICE_ROLE_KEY`: 上記秘密鍵で署名した service_role ロールの JWT
- `NEXT_PUBLIC_SUPABASE_URL`: `http://localhost:54321` を推奨（Kong 経由・ブラウザ用）
- `SUPABASE_INTERNAL_URL`: `http://kong:8000`（サーバー/SSR/API からの内部アクセス用）

補足
- Edge Functions は検証を無効化して起動（`VERIFY_JWT=false`）。開発用途に限って利用してください。
- 機密情報は `docker/env/*.env` に置き、Git にはコミットしないでください。

主要な値:
- `NEXT_PUBLIC_SUPABASE_URL`（例: `http://localhost:54321`）
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `FANZA_API_ID`, `FANZA_API_AFFILIATE_ID`, `FANZA_LINK_AFFILIATE_ID`

## 起動/停止

すべてのサービス（フロントエンド + Supabase 自前スタック + Edge Functions）をまとめて起動します。

```bash
# 事前にイメージを取得（任意だが推奨）
docker compose -f docker/compose.yml pull

# フォアグラウンド起動（-d なし）: ログをその場で確認できます
docker compose --env-file docker/env/dev.env -f docker/compose.yml up --build

# 停止方法（フォアグラウンド起動時）: ターミナルで Ctrl + C

# バックグラウンドで起動した場合の停止:
# docker compose --env-file docker/env/dev.env -f docker/compose.yml stop
# 完全停止（コンテナ削除）:
# docker compose --env-file docker/env/dev.env -f docker/compose.yml down
# ボリュームも削除（DB初期化などリセットしたい時のみ推奨）:
# docker compose --env-file docker/env/dev.env -f docker/compose.yml down -v

# フロントにアクセス
# http://localhost:3000
```

補助コマンド
```bash
# 稼働状況
docker compose --env-file docker/env/dev.env -f docker/compose.yml ps
# ログ追尾
docker compose --env-file docker/env/dev.env -f docker/compose.yml logs -f
```

## マイグレーション適用（未適用のみ）

開発中に `supabase/migrations/*.sql` が増えた場合、未適用分だけを適用するには以下を実行します。

```bash
docker compose --env-file docker/env/dev.env -f docker/compose.yml up db-migrate

# REST のスキーマキャッシュを更新（必要時）
docker compose --env-file docker/env/dev.env -f docker/compose.yml restart rest
```

初回セットアップやボリューム消去後（完全初期化）の場合は、`db-init` が `docker/db/init/*.sql` と `supabase/migrations/*.sql` を昇順で全適用します。

## スクリプトの実行

各スクリプトは専用コンテナで実行します（Compose は使いません）。

- 前処理（reviews → interactions）

```bash
# Remote DB から動画メタデータを取り込みつつデータ生成
bash scripts/prep_two_tower/run_with_remote_db.sh \
  --remote-db-url "$REMOTE_DATABASE_URL" \
  --mode reviews \
  --input ml/data/raw/reviews/dmm_reviews_videoa_YYYY-MM-DD.csv \
  --min-stars 4 --neg-per-pos 3 --val-ratio 0.2 \
  --run-id auto
# reviews モードでは reviews 由来の interactions から擬似 user_features を自動合成します（DB 接続なしでも可）
```

- 学習

```bash
bash scripts/train_two_tower/run.sh --embedding-dim 128 --epochs 5
```

- スクレイピング（DMM レビュー）

```bash
bash scripts/scrape_dmm_reviews/run.sh --output ml/data/raw/reviews/dmm_reviews.csv
```

- データ取り込み（FANZA API）

```bash
bash scripts/ingest_fanza/run.sh --help  # 必要な引数を確認
```

- リモート DB → ローカル DB 同期（上書き注意）

```bash
bash scripts/sync_remote_db/run.sh \
    --env-file docker/env/prd.env \
    --local-env-file docker/env/dev.env \
    --yes \
    --exclude-embeddings \
    --project-ref mfleexehdteobgsyokex
# 事後適用（デフォルト）: 既存のローカルCompose上のDBに対してデータのみ流し込みます
# Supabase CLI のローカルスタック起動や db reset は行いません
# CLI 管理のローカルスタックを使いたい場合のみ --start-local を付与
# 例: bash scripts/sync_remote_db/run.sh --start-local --yes
```

- マイグレーション適用（未適用のみ）

```bash
# 既存DBに対して、supabase/migrations/*.sql の未適用分だけを順に適用
bash scripts/db_migrate/run.sh
# REST のスキーマキャッシュを明示的に更新したい場合
RELOAD_REST=false bash scripts/db_migrate/run.sh && \
  docker compose -f docker/compose.yml restart rest

補足
- 初期起動（`docker compose up --build`）では DB の前提だけ整えます（拡張権限・スキーマなど）。
- アプリの RPC/関数（例: `get_videos_feed`）は、上記の `db_migrate` 実行後に利用可能になります。
- db-migrate は Compose サービスとしては保持していません。スクリプトがワンオフの Postgres コンテナを起動し、既存の Compose ネットワーク上で `psql` を実行します。
```

- DB リセット（2通り）

```bash
# ソフトリセット: DBボリュームのみ削除 → init SQL → 未適用マイグレーション → 依存サービス再起動
bash scripts/db_reset/run.sh

# ハードリセット: 全コンテナ停止 + 全ボリューム削除 + 再起動（破壊的）
HARD=true bash scripts/db_reset/run.sh
```

## 補足

- すべて Docker 前提です。ローカルの `.nvmrc`, `.python-version`, `.venv` は不要です。
- 機密情報は `docker/env/*.env` に保持し、Git にはコミットしないでください。
- Supabase は self-hosted 構成（db/rest/auth/realtime/storage/kong/studio + edge-runtime）。Edge Functions は `VERIFY_JWT=false` で起動します。
- 詳細は `docs/docker.md`, `docs/ml/two_tower_training.md` も参照してください。
# 補足: 自動初期化
# - `db` の起動をヘルスチェックで待ち、`db-init` ジョブが init SQL を idempotent に適用します
#   （空ボリュームでも、既存ボリュームでも安全に前提を整備します）
