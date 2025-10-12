# AGENTS ガイドライン

このリポジトリでエージェント（AI支援）を利用する際の共通ルールです。実装・運用は Docker 前提です。

## 基本方針

- 言語: 原則 日本語。
- 実行環境: すべて Docker で実行（ローカル Node/Python は前提にしない）。
- コミット: 自動コミットはしない。変更はレビュー方針に従い手動でコミットする。
- 要件: 作業前に `docs/requirements.md` を確認・更新する（存在しない場合は作成）。

## 作業ルール

- 変更点の確認は `rg`/`cat` などでソースを読み、差分は PR/コミットで提示する。
- 画像やアセットの拡張子は勝手に変更しない（必要時は事前に合意）。
- Tailwind 設定（例: `tailwind.config.js`）と CSS は別ファイルとして管理し混在させない。
- 機密情報は `docker/env/dev.env` などの `.env` に置き、Git にはコミットしない。

## 実行方法（要点）

- フロント: `docker compose -f docker/compose.yml up --build`
- Supabase（DB/Studio）+ Functions: `docker compose -f docker/compose.yml up -d supabase functions`
- スクリプト群は `scripts/<name>/run.sh` から専用 Docker コンテナで実行する。

## ディレクトリの約束

- `scripts/<name>/` にスクリプト本体、`Dockerfile`、`requirements.txt`（必要に応じて）を置く。
- 学習データは `ml/data/`、生成物は `ml/artifacts/` に配置する。
- 環境変数ファイルは `docker/env/` に配置し、例は `*.env.example`（例: `dev.env.example`, `prd.env.example`）を用意する。

不明点や例外運用が必要な場合は、この文書と `docs/requirements.md` をまず更新してから作業すること。
