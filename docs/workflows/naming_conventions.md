# GitHub Actions 命名規約

ワークフロー名は運用目的とカテゴリが一目で分かるよう、以下のルールで統一します。

## ルール

1. **カテゴリを `[Category]` 形式で先頭に付ける**（例: `[ML]`, `[AppRelease]`, `[CiCd]`, `[FetchData]`）
2. **役割を表す接頭辞を続ける**
   - `Ops - ...` : 日常運用で直接実行するワークフロー（手動実行・定期実行）
   - `Sub - ...` : Ops から呼び出される補助ワークフローや内部処理
3. **ファイル名も `<category>-<description>.yml` とし、検索しやすくする**（例: `ml-sync-video-embeddings.yml`）
4. **カテゴリとトリガーの対応**
   - `[AppRelease]` : main へのマージやリリースフローで実行されるワークフロー
   - `[CiCd]` : push / PR で走るテストやビルド確認のワークフロー
   - `[ML]`, `[FetchData]` など、用途が明確な専用カテゴリは用途に合わせて定義

## 現在のワークフロー対応表

| ファイル | `name:` | カテゴリ / 種別 |
| --- | --- | --- |
| `.github/workflows/ml-sync-video-embeddings.yml` | `[ML] Ops - Sync Video Embeddings` | ML / Ops |
| `.github/workflows/ml-update-user-embeddings.yml` | `[ML] Ops - Update User Embeddings` | ML / Ops |
| `.github/workflows/apprelease-deploy-frontend.yml` | `[AppRelease] Ops - Deploy Frontend to Vercel` | AppRelease / Ops |
| `.github/workflows/apprelease-release.yml` | `[AppRelease] Ops - Release Pipeline` | AppRelease / Ops |
| `.github/workflows/cicd-deploy-functions.yml` | `[CiCd] Ops - Deploy Supabase Edge Functions` | CiCd / Ops |
| `.github/workflows/cicd-deploy-db.yml` | `[CiCd] Sub - Deploy DB Migrations` | CiCd / Sub |
| `.github/workflows/cicd-deploy-docs.yml` | `[CiCd] Sub - Deploy Docs` | CiCd / Sub |
| `.github/workflows/cicd-deploy-db-docs.yml` | `[CiCd] Sub - Deploy DB Docs` | CiCd / Sub |

### 未整備のワークフロー

以下のワークフローは命名／配置が未確定です。追加・改名する際は本規約に沿って命名し、この一覧を更新してください。

| 想定カテゴリ | 想定ファイル名 | 想定 name | メモ |
| --- | --- | --- | --- |
| `[CiCd]` | `cicd-performance-benchmark.yml` | `[CiCd] Ops - Performance Benchmark` | push / PR 時の性能ベンチマークを想定 |
| `[ML]` | `ml-train-two-tower.yml` | `[ML] Ops - Train Two-Tower` | ML モデル学習パイプライン |
| `[AppRelease]` | `apprelease-pages-build-deployment.yml` | `[AppRelease] Sub - Pages Build Deployment` | GitHub Pages 自動デプロイ（必要であれば手動ワークフロー化） |

今後ワークフローを追加・改名する場合は、この表を更新しながら命名規約に沿って運用してください。
