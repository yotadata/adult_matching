# GitHub Actions で必要な環境変数 / Secrets 一覧

各ワークフローが参照する Secrets（もしくは Workflow 用 Variables）と、手動トリガー時の入力パラメータを整理しました。  
Secrets 名はリポジトリ設定（Settings → Secrets and variables → Actions）で事前に登録してください。

命名規約や Ops/Sub の分類は `docs/workflows/naming_conventions.md` を参照してください。

## 共通で使う Secrets

以下は複数ワークフローで共有されるため、まず最初に登録しておくことを推奨します。

| Secret 名 | 用途 |
| --- | --- |
| `SUPABASE_URL` | Supabase プロジェクトのベース URL（`https://<project-ref>.supabase.co`） |
| `SUPABASE_ANON_KEY` または `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon key（公開用） |
| `SUPABASE_SERVICE_ROLE_KEY` | Service Role key（RLS 無視、Storage 書き込み等に利用） |
| `REMOTE_DATABASE_URL` | Supabase Postgres の接続文字列（`sslmode=require` 推奨） |
| `SUPABASE_PROJECT_REF` | Supabase プロジェクトの ref（CLI・API で使用する ID） |
| `SUPABASE_REGION` | Supabase リージョン名（例: `aws-1-ap-northeast-1`） |
| `SUPABASE_DB_PASSWORD` | `supabase db remote connect` 等で利用するデータベースパスワード |
| `SUPABASE_ACCESS_TOKEN` | Supabase CLI で `supabase login` する際の Personal Access Token |

## ワークフロー別の追加要素

### 1. `.github/workflows/cicd-deploy-db.yml`
- **Secrets**（上記共通に加えて必須）
  - `SUPABASE_DB_PASSWORD`
  - `SUPABASE_ACCESS_TOKEN`
- **workflow_dispatch inputs**
  | 入力名 | 説明 | 既定値 |
  | --- | --- | --- |
  | `target` | デプロイ先の project ref。未指定時は `SUPABASE_PROJECT_REF` を使用 | 空 |
  | `schema` | 適用する schema（例: `public`）。空なら全体 | 空 |

### 2. `.github/workflows/cicd-deploy-functions.yml`
- **Secrets**
  - 共通のほか `SUPABASE_DB_PASSWORD`, `SUPABASE_ACCESS_TOKEN`
- **workflow_dispatch inputs**
  | 入力名 | 説明 | 既定値 |
  | --- | --- | --- |
  | `function` | デプロイする Edge Function 名。空なら `supabase/functions/` 配下を全てデプロイ | 空 |

### 3. `.github/workflows/apprelease-deploy-frontend.yml`
- **Secrets**
  - `VERCEL_TOKEN`
  - `VERCEL_ORG_ID`
  - `VERCEL_PROJECT_ID`
  - （必要に応じて）`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- **workflow_dispatch inputs**
  なし（手動トリガー時もデフォルト設定でデプロイ）

### 4. `.github/workflows/apprelease-release.yml`
- **Secrets**
  - `SUPABASE_ACCESS_TOKEN`
  - `SUPABASE_PROJECT_REF`
  - `SUPABASE_DB_PASSWORD`
- **workflow_dispatch inputs**
  | 入力名 | 説明 | 既定値 |
  | --- | --- | --- |
  | `targets` | デプロイ対象の環境（例: `staging,production`） | `staging` |

### 5. `.github/workflows/ml-sync-video-embeddings.yml`
- **Secrets**
  - 共通のもの + FANZA 系
  - `SUPABASE_URL`
  - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
  - `SUPABASE_SERVICE_ROLE_KEY`
  - `REMOTE_DATABASE_URL`
  - `SUPABASE_PROJECT_REF`
  - `SUPABASE_REGION`
  - `FANZA_API_ID`
  - `FANZA_API_AFFILIATE_ID`
  - `FANZA_LINK_AFFILIATE_ID`
- **workflow_dispatch inputs**（すべて任意）
  | 入力名 | 説明 | 既定値 |
  | --- | --- | --- |
  | `start_date` | FANZA 取得範囲の開始日（`YYYY-MM-DD` JST） | 空 |
  | `end_date` | FANZA 取得範囲の終了日 | 空 |
| `lookback_days` | 上記が未指定の場合の取得日数 | `3` |
| `skip_ingest` | FANZA 取得ステップをスキップするか | `false` |
| `skip_fetch` | Storage からモデルを取得するステップをスキップするか | `false` |
| `mode` | `full` (取得+埋め込み) / `embeddings` (埋め込みのみ) | `full` |

### 6. その他ワークフロー（例: `cicd-deploy-docs.yml`, `cicd-deploy-db-docs.yml`）
- それぞれ README / ワークフローコメントに記載されている Secrets を参照してください。  
 追加で共通化したい変数が発生した場合は、このドキュメントを更新して一覧化してください。

---

**運用メモ**
- Secrets 名は可能な限り既存の `.env` 名と合わせています。新しく追加する場合は `docs/env/variables.md` にも反映してください。  
- 手動トリガー時の入力（workflow_dispatch inputs）は、空を許容するものが多いため必要時のみ指定すれば問題ありません。  
- 複数ワークフローで同じ Secrets を参照する場合は、組織レベルの Secrets を使うと管理コストを下げられます。
