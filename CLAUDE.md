# CLAUDE.md — Claude Code 運用ルール

## スコープ制限

- **このリポジトリ外のファイル・システムへのアクセスは禁止。**
- 外部 API・URL へのリクエストは、明示的に指示された場合のみ許可（デバッグ・調査目的に限る）。
- `.env` ファイル（`docker/env/*.env`）は読み取り可能だが、Git にコミットしてはならない。
- `ml/data/`, `ml/artifacts/` 配下の大容量ファイルは Git にコミットしない。

## ブランチ・PR 運用

作業を始める前に **必ず専用ブランチを切る**。`feature-kamiura` などの長命ブランチを使い回さない。

```bash
git checkout main && git pull
git checkout -b fix/内容   # バグ修正
git checkout -b feature/内容  # 新機能
```

- 1ブランチ = 1テーマ。完了したらすぐ PR を作成して main にマージする。
- PR はマージ後にブランチを削除する。

## タスク完了時のコミット

**各タスクが完了するたびに**コミットを行う（複数タスクをまとめてコミットしない）。

手順:
1. `git status` で変更ファイルを確認する。
2. 機密ファイル（`docker/env/*.env`、`*.env`、`*.pem` など）が含まれていないことを確認する。
3. 変更内容を適切な粒度でステージング（`git add <files>`）し、コミットする。
4. コミットメッセージは日本語で、変更内容を端的に表す。

```
git commit -m "変更の要約（日本語）"
```

## 基本方針

- 言語: 日本語（コード内コメントも日本語可）。
- 実行環境: すべて Docker 前提。ローカルの Node / Python は使用しない。
- 作業前に `docs/requirements.md` を確認する（なければ作成する）。

## 機能仕様書

実装前に必ず対応する仕様書を確認すること。

| 機能 | 仕様書 |
|---|---|
| キュレーター機能 MVP | [`docs/features/curator_mvp.md`](docs/features/curator_mvp.md) |
| AI推薦 | [`docs/features/ai_recommend.md`](docs/features/ai_recommend.md) |

## コーディングルール

- Tailwind 設定（`tailwind.config.js`）と CSS は別ファイルで管理し、混在させない。
- 画像・アセットの拡張子は勝手に変更しない。
- 機密情報は `docker/env/dev.env` などの `.env` に置き、コードにハードコードしない。

## ディレクトリ規約

| ディレクトリ | 用途 |
|---|---|
| `frontend/` | Next.js 15 + React 19 アプリ |
| `supabase/migrations/` | DBマイグレーション SQL |
| `supabase/functions/` | Edge Functions |
| `scripts/<name>/` | スクリプト本体 + Dockerfile + run.sh |
| `ml/data/` | 学習用入力データ（CSV/Parquet） |
| `ml/artifacts/` | 学習成果物（ONNX/pt） |
| `docker/env/` | 環境変数ファイル（`.env` 系） |
| `docs/` | ドキュメント |

## 実行コマンド（参考）

```bash
# 全サービス起動
docker compose --env-file docker/env/dev.env -f docker/compose.yml up --build

# マイグレーション適用
docker compose --env-file docker/env/dev.env -f docker/compose.yml up db-migrate

# スクリプト実行例
bash scripts/<name>/run.sh
```

## Supabase マイグレーション運用

### リモートのみ適用されたマイグレーションの扱い

Supabase ダッシュボードや MCP (`apply_migration`) でリモートに直接適用したマイグレーションは、ローカルの `supabase/migrations/` に存在しないため `db push` がエラーになる。**必ず以下の手順をセットで行う。**

#### 毎回やること（リモート直接適用のたびに）

1. **stub ファイルを作成する**

```bash
echo "-- Remote-only migration stub (applied directly to remote)" \
  > supabase/migrations/<timestamp>_remote_only_stub.sql
```

2. **deploy ワークフローの repair リストに追加する**

`.github/workflows/cicd-deploy-db.yml` の `Repair remote-only migrations` ステップにバージョン番号を追記する。

```yaml
for v in ... \
         <timestamp>; do
```

3. **コミットに含める**

stub ファイルと `cicd-deploy-db.yml` の変更を同じコミットに含める。

#### なぜ必要か

`supabase db push --include-all` は、リモートの migration 履歴テーブルにあってローカルに存在しないバージョンがあるとエラーになる。`repair --status reverted` でそのバージョンを「取り消し済み」として扱わせることで回避できる。stub ファイルはローカルの tracking 記録として残す。

## 禁止事項

- `--no-verify` によるフックのスキップ。
- `git push --force`（明示的指示がない限り）。
- `docker/env/*.env` のコミット。
- このリポジトリ外のパスへの書き込み。
