# Two‑Tower 学習ガイド（256 次元・PyTorch・ローカル優先）

本ドキュメントは、LIKE/DISLIKE を主とした明示フィードバックを用いて Two‑Tower（ユーザー塔×アイテム塔）モデルを学習し、生成物をオブジェクトストレージ（Supabase Storage）へ配置するまでをまとめたものです。まずはローカルで学習し、のちに GitHub Actions へ移行できる構成とします。

## 目的とアウトプット

- 目的: ユーザー×動画の嗜好を Two‑Tower で学習し、初期の個人化推薦を可能にする。
- 埋め込み次元: 256 に固定。
- 学習アウトプット（本仕様）
  - モデル（PyTorch state_dict と ONNX）を Storage へ配置（Python/TypeScript から読み込み可能）

## データソース（優先: 明示フィードバック）

- 明示フィードバック（推奨）: `reviewer_id, product_url, label`（LIKE=1, DISLIKE=0）
  - 負例サンプリングは不要。DISLIKE が無い/極端に少ないユーザーのみ、任意で疑似負例を補う。
- 代替（レビュー由来の星評価）: `product_url, reviewer_id, stars`（例: `ml/data/dmm_reviews_videoa_YYYY-MM-DD.csv`）
  - 簡易に「暗黙フィードバック化」して学習可能。正例（stars>=4）に対し、未観測アイテムから K 件の疑似負例をサンプリング。

### データソース統合（今回の前処理スクリプト仕様）

- 想定元ソース:
  - テーブル相当: `user_video_decisions`（LIKE/DISLIKE ベース、初期モデルでは未使用）
- 外部CSV: DMM レビュー `ml/data/dmm_reviews_videoa_*.csv`
- スクリプトは2つのソースを単一形式（`reviewer_id, product_url, label`）に整形・統合できます。
  - 既定では decisions を含めません（初期モデル方針）。
  - 必要時に `--decisions-csv` へ CSV を渡すと、レビュー由来のデータに decisions をマージします（同一ユーザー×アイテムは decisions 側で上書き）。
  - decisions CSV の列は柔軟に解釈します:
    - ユーザー: `reviewer_id` または `user_id`
    - アイテム: `product_url` または `item_id`（本スクリプトでは product_url を推奨）
    - ラベル: `label`（0/1） または `decision`（LIKE/DISLIKE/SKIP）。`decision` は LIKE→1, DISLIKE→0, SKIP→無視。

## ラベル設計

- 明示: LIKE=1 / DISLIKE=0 をそのまま学習（BCE）に使用。
- 暗黙（レビュー）: 正例は `stars>=4`、負例は未観測アイテムからの疑似負例（例: 正例1に対しK=3）。
  - 注: 将来好きになる可能性を考慮し、K を小さく・学習率を保守的に。

## パイプライン全体像

1) 入力CSVの準備（優先: LIKE/DISLIKE の明示データ）
2) 前処理: `interactions(user_id, item_id, label)` を作成（ユーザー層化で train/val 分割）
3) Two‑Tower を PyTorch で学習（埋め込み256次元）
4) 生成物: モデル（state_dict, ONNX）、IDマップ（user/item）、メタ情報
5) （任意）評価メトリクス: Recall@K / MAP@K

## ローカル検証の流れ

1) Python venv 構築

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/train_two_tower/requirements.txt
```

### uv での実行（推奨）

uv を使うと仮想環境の作成と依存導入、スクリプト実行が簡潔になります。

```bash
# 1) uv のインストール（未導入の場合）
# macOS (Homebrew):
brew install uv
# もしくは公式スクリプト:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Python 3.11 の用意（pyenv or brew のいずれか）
# pyenv:  pyenv install 3.11.9 && pyenv local 3.11.9
# brew:   brew install python@3.11; export UV_PYTHON="$(which python3.11)"

# 3) 仮想環境の作成と依存導入
uv venv
uv pip install -r scripts/train_two_tower/requirements.txt

# 4) 前処理（DMMレビューのみ・初期モデル）
uv run scripts/prep_two_tower/prep_two_tower_dataset.py \
  --mode reviews \
  --input ml/data/dmm_reviews_videoa_2025-10-04.csv \
  --min-stars 4 --neg-per-pos 3 \
  --val-ratio 0.2 \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet

# 5) 学習（256次元）
uv run scripts/train_two_tower/train_two_tower.py \
  --embedding-dim 256 --epochs 5 --batch-size 1024 --lr 1e-3

# 参考: レビュー＋decisions を統合する場合
uv run scripts/prep_two_tower/prep_two_tower_dataset.py \
  --mode reviews \
  --input ml/data/dmm_reviews_videoa_2025-10-04.csv \
  --decisions-csv ml/data/user_video_decisions_export.csv \
  --min-stars 4 --neg-per-pos 3 \
  --val-ratio 0.2 \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet
```

トラブルシューティング
- 「No virtual environment found」: `uv venv` を実行してから `uv pip install ...` を実行してください。
- 「pyenv: version '3.11' is not installed」: `pyenv install 3.11.x` か、`brew install python@3.11 && export UV_PYTHON=$(which python3.11)` を設定してください。

2) 前処理（明示ラベルが既にある場合: 既定）

```bash
python scripts/prep_two_tower/prep_two_tower_dataset.py \
  --mode explicit \
  --input ml/data/reactions.csv \
  --val-ratio 0.2 \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet
```

3) 前処理（レビュー星から作る場合: 代替）

```bash
python scripts/prep_two_tower_dataset.py \
  --mode reviews \
  --input ml/data/dmm_reviews_videoa_2025-10-04.csv \
  --min-stars 4 --neg-per-pos 3 \
  --val-ratio 0.2 \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet

3b) 前処理（レビュー＋decisions を統合する場合: 任意）

```bash
python scripts/prep_two_tower_dataset.py \
  --mode reviews \
  --input ml/data/dmm_reviews_videoa_2025-10-04.csv \
  --decisions-csv ml/data/user_video_decisions_export.csv \
  --min-stars 4 --neg-per-pos 3 \
  --val-ratio 0.2 \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet
```

3c) videos をDBから直接取得（cid→external_id 突合／itemキーを video_id に正規化）

```bash
# 事前に videos テーブルをCSVにエクスポートしておく（最低: id, product_url）
# 例: supabase studio からダウンロード、またはSQL→CSV出力

uv run scripts/prep_two_tower/prep_two_tower_dataset.py \
  --mode reviews \
  --input ml/data/dmm_reviews_videoa_2025-10-04.csv \
  --db-url postgresql://postgres:postgres@127.0.0.1:54322/postgres \
  --join-on external_id \
  --min-stars 4 --neg-per-pos 3 \
  --val-ratio 0.2 \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet \
  --out-items ml/data/item_features.parquet

# 出力と挙動:
# - interactions_{train,val}.parquet に 'video_id' 列が付与（join成功分のみ保持）
# - DMMの product_url から `cid` を抽出し、`videos.external_id` と突合（推奨）。一致しないものは除外（ドロップ件数はサマリに出力）
# - item_features.parquet に video_id, product_url, title, maker, label, series, external_id, tags（カンマ区切り）を格納
```
```
```

4) 学習（256 次元）

```bash
python scripts/train_two_tower/train_two_tower.py \
  --embedding-dim 256 --epochs 5 --batch-size 2048 --lr 1e-3

# itemキーに video_id を使う場合（prepで videos を渡したとき）
python scripts/train_two_tower/train_two_tower.py \
  --embedding-dim 256 --epochs 5 --batch-size 2048 --lr 1e-3 \
  --item-key video_id
```

5) 生成物（`ml/artifacts/`）

- `two_tower_latest.pt`（PyTorch state_dict）
- `two_tower_latest.onnx`（ONNX: Python/TypeScript からロード可能）
- `mappings/user_id_map.json`, `mappings/item_id_map.json`
- `model_meta.json`（メタ情報: 次元、件数など）

### 取得対象（例）

- `public.videos`（id, product_url, external_id, title, maker, label, series, …）
- `public.video_embeddings`（初回は空でも可）
- `public.user_embeddings`（初回は空でも可）
- （他）likes/decisions/視聴履歴があれば将来活用

## ローカル検証の流れ（提案）

1) Python venv 構築

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/train_two_tower/requirements.txt
```

2) 前処理スクリプト（例: `scripts/prep_two_tower/prep_two_tower_dataset.py`）

- 入力: `ml/data/dmm_reviews.csv` + Postgres(`videos`)
- 出力: `ml/data/interactions.parquet`（`user_id, video_id, label`）

3) 学習スクリプト（例: `scripts/train_two_tower/train_two_tower.py`）

- 引数例

```bash
python scripts/train_two_tower/train_two_tower.py \
  --input ml/data/interactions.parquet \
  --embedding-dim 128 \
  --epochs 5 --batch-size 2048 --lr 1e-3 \
  --out-model ml/artifacts/two_tower_latest.pkl \
  --out-video-emb ml/data/video_embeddings.parquet \
  --out-user-emb ml/data/user_embeddings.parquet
```

4) 反映

- DB 書き込み（Upsert）ユーティリティ（例: `scripts/upsert_embeddings.py`）で `video_embeddings` / `user_embeddings` を更新
- pgvector インデックス（初回のみ）

```sql
-- 初回のみ: 類似検索用インデックス（次元は実際の embedding 次元に合わせる）
create index if not exists idx_video_embeddings_cosine on public.video_embeddings using ivfflat (embedding vector_cosine_ops);
create index if not exists idx_user_embeddings_cosine on public.user_embeddings using ivfflat (embedding vector_cosine_ops);
```

## GitHub Actions（将来移行の雛形）

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
          pip install -r scripts/train_two_tower/requirements.txt

      - name: Prepare dataset (explicit)
        run: |
          python scripts/prep_two_tower_dataset.py \
            --mode explicit \
            --input ml/data/reactions.csv \
  --out-train ml/data/interactions_train.parquet \
  --out-val ml/data/interactions_val.parquet

      - name: Train TwoTower
        run: |
          python scripts/train_two_tower.py \
            --embedding-dim 256 --epochs 5 --batch-size 2048 --lr 1e-3

      - name: Upload model to Supabase Storage
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}
        run: |
          # 例: storage バケット "models" にアップロード（ONNX推奨）
          curl -X POST "$SUPABASE_URL/storage/v1/object/models/two_tower_latest.onnx" \
            -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
            -H "Content-Type: application/octet-stream" \
  --data-binary @ml/artifacts/two_tower_latest.onnx

      # 埋め込みのDB反映は本仕様では不要のため省略
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
-- [ ] ONNXの推論が期待通り（例: サンプルユーザー/アイテムでスコアが出る）

## よくある論点

- 未登録 `product_url` の扱い: 前処理時に最小挿入→後続クロールで enrich（任意）。
- 埋め込み次元: 本仕様は 256 に固定。将来変更時はDBスキーマ・下流も一括更新。
- 評価: ユーザー履歴が薄い場合、popularity 混合や正則化で安定化。

---

以上です。スクリプトは `scripts/` 配下に追加済みです。必要に応じて Upsert 用スクリプトや GHA ワークフローも追補します。
