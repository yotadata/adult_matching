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
2) 環境変数ファイルを作成

```bash
cp docker/env/dev.env docker/env/dev.env.local  # 例: バックアップ
# docker/env/dev.env に実値を設定
```

主要な値:
- `NEXT_PUBLIC_SUPABASE_URL`（例: `http://host.docker.internal:54321`）
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `FANZA_API_ID`, `FANZA_API_AFFILIATE_ID`, `FANZA_LINK_AFFILIATE_ID`

## 起動方法

すべてのサービス（フロントエンド + Supabase + Edge Functions）をまとめて起動します。

```bash
docker compose -f docker/compose.yml up -d --build
# http://localhost:3000 にアクセス
```

## スクリプトの実行

各スクリプトは専用コンテナで実行します（Compose は使いません）。

- 前処理（reviews → interactions）

```bash
bash scripts/prep_two_tower/run.sh \
  --mode reviews \
  --input ml/data/dmm_reviews_videoa_YYYY-MM-DD.csv \
  --min-stars 4 --neg-per-pos 3 \
  --val-ratio 0.2 \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet
```

- 学習

```bash
bash scripts/train_two_tower/run.sh --embedding-dim 256 --epochs 5
```

- スクレイピング（DMM レビュー）

```bash
bash scripts/scrape_dmm_reviews/run.sh --output ml/data/dmm_reviews.csv
```

- データ取り込み（FANZA API）

```bash
bash scripts/ingest_fanza/run.sh --help  # 必要な引数を確認
```

- リモート DB → ローカル DB 同期（上書き注意）

```bash
bash scripts/sync_remote_db/run.sh --env-file docker/env/dev.env --yes
```

## 補足

- すべて Docker 前提です。ローカルの `.nvmrc`, `.python-version`, `.venv` は不要です。
- 機密情報は `docker/env/*.env` に保持し、Git にはコミットしないでください。
- 詳細は `docs/docker.md`, `docs/two_tower_training.md` も参照してください。
