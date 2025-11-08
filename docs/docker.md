# Docker Compose Dev Setup

This repo includes a Docker setup to run:
- Next.js frontend (hot reload)
- ML workspace container for training/inference
- Supabase local stack managed via Supabase CLI inside a container

-## Prerequisites
- Docker + Docker Compose
- Copy `docker/env/dev.env.example` to `docker/env/dev.env` and fill values
- Copy `docker/env/prd.env.example` to `docker/env/prd.env` when接続テストや本番用スクリプトを実行する（`REMOTE_DATABASE_URL` や Supabase の鍵を設定）
- Login to GitHub Container Registry (for Supabase CLI image):
  - Create a GitHub Personal Access Token with `read:packages`
  - `echo <TOKEN> | docker login ghcr.io -u <GITHUB_USERNAME> --password-stdin`

## Start All Services (Dev)
- Run: `docker compose -f docker/compose.yml up -d --build`
- Open: `http://localhost:3000`

The container uses Node 20, mounts the `frontend` directory for live reload, and persists `node_modules` in a named volume.

## Supabase
- Compose brings up the local stack and Edge Functions automatically.
- API: `http://localhost:54321`
- Edge Functions: `http://localhost:54321/functions/v1/<function>`
- From other containers, set in `docker/env/dev.env`:
  - `NEXT_PUBLIC_SUPABASE_URL=http://host.docker.internal:54321`
  - `NEXT_PUBLIC_SUPABASE_ANON_KEY=...` (from logs of supabase services)

## Commands (per-script Docker)
- Start all services: `docker compose -f docker/compose.yml up -d --build` (uses `docker/env/dev.env`)
- Stop all services: `docker compose -f docker/compose.yml down`
- Prep dataset: `bash scripts/prep_two_tower/run_with_remote_db.sh --remote-db-url "$REMOTE_DATABASE_URL" --mode reviews --input ml/data/raw/reviews/dmm_reviews_videoa_....csv --run-id auto`
- Train model: `bash scripts/train_two_tower/run.sh --train ml/data/processed/two_tower/latest/interactions_train.parquet --val ml/data/processed/two_tower/latest/interactions_val.parquet --user-features ml/data/processed/two_tower/latest/user_features.parquet --item-features ml/data/processed/two_tower/latest/item_features.parquet --item-key video_id --embedding-dim 256 --hidden-dim 512 --epochs 5 --batch-size 1024 --lr 1e-3 --max-tag-features 2048 --max-performer-features 512 --out-dir ml/artifacts`
- Evaluate model: `bash scripts/eval_two_tower/run.sh python scripts/eval_two_tower/evaluate_two_tower.py --item-key video_id --recall-k 20 --out-json ml/artifacts/metrics.json`
- Streamlit review: `bash scripts/streamlit_qual_eval/run.sh`（ブラウザで `http://localhost:8501/` を開き、Model A/B を指定するとユーザー別の推薦比較やバイアス分布が確認できる。タブを切り替えることで maker / source / label / series / tags / performer_ids などの偏りを確認可能）
- Scrape reviews: `bash scripts/scrape_dmm_reviews/run.sh --output ml/data/raw/reviews/dmm_reviews.csv`

## Notes
- Edge Functions: In dev, you can run `supabase functions serve` on the host alongside `supabase start`. The frontend will call them via the Supabase API URL (use the same `host.docker.internal` host if needed).
- If you change dependencies in `frontend/package.json`, re-run with `--build` to refresh the image layer cache, or run `npm i` in the container shell.
- To stop: `docker compose -f docker/compose.yml down`

## Env files policy
- Canonical location for container envs: `docker/env/dev.env` (real file, ignored by git)
- Commit examples only: `docker/env/*.env.example` (e.g., `dev.env.example`, `prd.env.example`)
- Legacy files like `/.env.remote` or `frontend/.env.local` are no longer referenced anywhere.

## ML workspace
- Container image: Python 3.11 (see per-script requirements under `scripts/*/requirements.txt`)
- Default working dir: `/workspace` (repo root mounted)
- Data directory: `ml/data/`
- Artifacts directory: `ml/artifacts/`
- Example:
  - `bash scripts/run_train_two_tower.sh --embedding-dim 256 --epochs 5`
  - Customize data paths via args `--train ml/data/... --val ml/data/... --out-dir ml/artifacts`

## IPv6 設定（Docker Desktop）
`scripts/sync_remote_db` からリモート Supabase へ `pg_dump` を実行する際は、Docker が IPv6 を扱えることが前提です。Docker Desktop を利用している場合は、以下の設定を `Settings → Docker Engine` に追加してください。

```jsonc
{
  // 既存の設定に追記する
  "ipv6": true,
  "fixed-cidr-v6": "fd00:dead:beef::/64"
}
```

- `fixed-cidr-v6` は任意の Unique-Local アドレス帯を指定してください（他のネットワークと重複しない範囲を推奨）。
- 保存後に Docker Desktop の再起動が必要です。
- 反映確認: `docker info | grep -i IPv6` や `docker network inspect bridge | rg IPv6Gateway` が値を返すこと。
- `scripts/sync_remote_db` は IPv6 接続での `pg_dump` が必須です。設定不足で `pg_dump` が失敗した場合、スクリプトは即終了しリトライを促す仕様になっています。

Linux ホストの場合は上記に加えて `sysctl net.ipv6.conf.all.disable_ipv6=0` など OS レベルで IPv6 が有効化されているかを確認してください。`docker run --rm --network host ...` が利用できない/IPv6 が未整備な環境では `scripts/sync_remote_db` を実行できません。
