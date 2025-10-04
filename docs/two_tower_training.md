# Two‑Tower 初期学習ガイド（DMMレビュー活用）

本ドキュメントは、DMMレビューのスクレイプ結果を用いて Two‑Tower（ユーザー塔×アイテム塔）モデルの初期学習を行い、学習済みモデルを Supabase のストレージへ配置するまでを一気通貫でまとめたものです。

## 目的とアウトプット

- 目的: DMM レビュー由来のユーザー×動画相性を Two‑Tower で学習し、初期の個人化推薦を可能にする。
- 学習アウトプット（いずれか、または両方）
  - A) 学習済みモデルファイル（例: `models/two_tower_latest.pkl`）を Supabase Storage へアップロード
  - B) `video_embeddings` / `user_embeddings` を再計算し、Supabase DB（pgvector）へ反映

現行の Edge Function（`supabase/functions/ai-recommend`）はスタブのため、ローンチ段階は B の「埋め込みDB参照」方式を推奨します。モデルファイル（A）は将来的なオンライン更新や再学習容易化のために保管します。

## データソースとスキーマ前提

- 学習元: `data/dmm_reviews.csv`（スクレイパ出力）
  - 列: `product_url, reviewer_id, stars`
  - すべてのレビューを取得済み（URL検証なし）
- 動画テーブル（Supabase）: `public.videos`
  - 主に `id (uuid)`, `product_url`, `created_at` などを使用
- 埋め込みテーブル（Supabase）: `public.video_embeddings`, `public.user_embeddings`
  - いずれも `embedding vector(256)` を前提（次元は必要に応じて調整）

補足: `product_url` が `videos` に未登録の場合は、最小項目で新規レコードを作る「ingest」工程を挟みます（タイトル等は後続のクローラで補完）。

## ラベル設計（初期案）

- 暗黙フィードバック化（バイナリ化）
  - 正例: `stars >= 4`
  - 負例: レビュワー未観測アイテムからランダムサンプル（正例:負例=1:K 例: K=3）
- 代替案
  - 重み付き（回帰的）: `weight = stars / 5.0`（損失に重み付け）

まずは暗黙×負例サンプリングの方が堅く動かせます。必要あれば切替可能です。

## パイプライン全体像

1) 本番DBダンプ取得 → ローカル/CI で復元 or 直接接続（読み取り専用）
2) `reviews.csv` を `videos.product_url` で突合 → `interactions(user_id, video_id, label/weight)` 生成
   - 未登録URLは `videos` に最小限で `INSERT`（任意）
3) データ分割: train/valid（例: 8/2、ユーザーで層化）
4) Two‑Tower 学習（PyTorch/TensorFlow いずれも可）
5) 生成物
   - A) モデルファイル（保存）
   - B) `video_embeddings` / `user_embeddings` Upsert（pgvector）
6) （任意）監視用メトリクス出力: Recall@K、Precision@K、MAP@K

## 本番DBダンプ運用（推奨）

- いずれかの方法で本番DBをダンプ（必要テーブルのみでも可）
  - Supabase CLI: `supabase db dump --project-ref <REF> --schema-only=false`（要ドキュメント参照）
  - もしくは `pg_dump`（接続文字列は GitHub Secrets 管理）
- 復元（ローカル検証向け）
  - Docker Postgres をアドホック起動し `pg_restore` で復元、または CI 内で一時DBへ復元
- セキュリティ
  - 接続情報・ダンプは GitHub Encrypted Secrets / OIDC で保護
  - PII・ユーザー識別子の匿名化が必要な場合は、ダンプ後の変換工程でハッシュ化

### 取得対象（例）

- `public.videos`（id, product_url, created_at, …）
- `public.video_embeddings`（初回は空でも可）
- `public.user_embeddings`（初回は空でも可）
- （他）likes/decisions/視聴履歴があれば将来活用

## ローカル検証の流れ（提案）

1) Python venv 構築

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt  # 既存
# 追加で学習用依存（例）
pip install torch torchvision torchaudio pytorch-lightning scikit-learn pandas numpy pgvector psycopg[binary]
```

2) 前処理スクリプト（例: `scripts/prep_two_tower_dataset.py`）

- 入力: `data/dmm_reviews.csv` + Postgres(`videos`)
- 出力: `data/interactions.parquet`（`user_id, video_id, label`）

3) 学習スクリプト（例: `scripts/train_two_tower.py`）

- 引数例

```bash
python scripts/train_two_tower.py \
  --input data/interactions.parquet \
  --embedding-dim 128 \
  --epochs 5 --batch-size 2048 --lr 1e-3 \
  --out-model artifacts/two_tower_latest.pkl \
  --out-video-emb data/video_embeddings.parquet \
  --out-user-emb data/user_embeddings.parquet
```

4) 反映

- DB 書き込み（Upsert）ユーティリティ（例: `scripts/upsert_embeddings.py`）で `video_embeddings` / `user_embeddings` を更新
- pgvector インデックス（初回のみ）

```sql
-- 初回のみ: 類似検索用インデックス（次元は実際の embedding 次元に合わせる）
create index if not exists idx_video_embeddings_cosine on public.video_embeddings using ivfflat (embedding vector_cosine_ops);
create index if not exists idx_user_embeddings_cosine on public.user_embeddings using ivfflat (embedding vector_cosine_ops);
```

## GitHub Actions（本番実行）の雛形

以下は学習 → ストレージへモデル配置 →（任意で）埋め込みをDBに反映、までの一例です。実際にはリポジトリにスクリプトを実装した後、このワークフローを `.github/workflows/train_two_tower.yml` として追加します。

```yaml
name: Train TwoTower
on:
  workflow_dispatch: {}
  schedule:
    - cron: '0 4 * * 0'  # 毎週日曜 04:00 JST（docs/batch_jobs と整合）

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r scripts/requirements.txt
          pip install torch torchvision torchaudio pytorch-lightning scikit-learn pandas numpy requests

      - name: Fetch prod DB dump (pg_dump)
        env:
          DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}
        run: |
          pg_dump --no-owner --no-privileges --format=custom "$DATABASE_URL" \
            --table=public.videos \
            --table=public.video_embeddings \
            --table=public.user_embeddings \
            --file=prod.dump

      - name: Restore dump to local Postgres
        run: |
          docker run -d --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
          sleep 10
          pg_restore --clean --if-exists --no-owner --no-privileges --dbname=postgresql://postgres:postgres@localhost:5432/postgres prod.dump

      - name: Prepare dataset
        run: |
          python scripts/prep_two_tower_dataset.py \
            --reviews data/dmm_reviews.csv \
            --db postgresql://postgres:postgres@localhost:5432/postgres \
            --out data/interactions.parquet

      - name: Train TwoTower
        run: |
          python scripts/train_two_tower.py \
            --input data/interactions.parquet \
            --embedding-dim 128 --epochs 5 --batch-size 2048 --lr 1e-3 \
            --out-model artifacts/two_tower_latest.pkl \
            --out-video-emb data/video_embeddings.parquet \
            --out-user-emb data/user_embeddings.parquet

      - name: Upload model to Supabase Storage
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}
        run: |
          # 例: storage バケット "models" にアップロード
          curl -X POST "$SUPABASE_URL/storage/v1/object/models/two_tower_latest.pkl" \
            -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
            -H "Content-Type: application/octet-stream" \
            --data-binary @artifacts/two_tower_latest.pkl

      - name: Upsert embeddings to DB (optional)
        env:
          DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}
        run: |
          python scripts/upsert_embeddings.py \
            --db "$DATABASE_URL" \
            --video-emb data/video_embeddings.parquet \
            --user-emb data/user_embeddings.parquet
```

### Secrets と権限

- `PROD_DATABASE_URL`: 本番 Postgres（読み書き）接続文字列（RLSやロールに注意）
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`: Storage API へのアップロードに使用
- Storage 側のバケット（例: `models`）は事前に作成しておく

## 週次自動更新（GHA）

初期学習と同一パイプラインを週次で自動実行します。

- スケジュール: `cron: '0 4 * * 0'`（毎週日曜 04:00 JST）
- バージョニング: `two_tower_YYYYMMDDHHMM.pkl` を保存しつつ、`two_tower_latest.pkl` を上書き
- 品質ゲート（推奨）:
  - 評価指標の下限（例: Recall@10 が前回値から大幅悪化しない）を満たさない場合は `latest` の更新を抑止
  - 失敗時フォールバック: 直近安定版の `latest` を維持し、埋め込み Upsert はスキップ
- 通知: Slack/メールで成功・失敗を通知

上記の GitHub Actions 雛形（Train TwoTower）は、手動実行 `workflow_dispatch` と週次 `schedule` の両方をサポートしています。そのまま週次更新として利用できます。

## 推薦API側の切替

- 短期: Edge Function で `video_embeddings` × `user_embeddings` の類似検索を実装（`vector_cosine_ops`）。
- 長期: モデルファイル（Storage）を学習・運用のバージョニングに使用（オンライン学習/AB テスト基盤へ拡張）。

## ローカル検証の最小チェックリスト

- [ ] `reviews.csv` → `interactions.parquet` が妥当（件数、ユーザー/動画数、星分布）
- [ ] 学習 1〜5 epoch で overfit しすぎない・loss 低下
- [ ] Recall@K（例: K=10）がランダムより十分に良い
- [ ] DB 反映後、簡易クエリで類似検索が返る（例: 上位10件）

## よくある論点

- 未登録 `product_url` の扱い: ingest で最小挿入→後続クロールで enrich がベター
- 埋め込み次元: 計算資源とトレードオフ。最終的には 768 へ射影してスキーマ整合
- 評価: ユーザーごとのレビュー濃度差に留意。ヒストリー極小ユーザーは正則化・popularity 混合が有効

---

以上です。了承いただければ、前処理/学習/Upsert 用のスクリプト雛形を `scripts/` 配下に順次追加します（最初はダミー実装→段階的に精緻化）。
