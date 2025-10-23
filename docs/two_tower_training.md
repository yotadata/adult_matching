# Two‑Tower 学習ガイド（256 次元・PyTorch・ローカル優先）

本ドキュメントは、LIKE/DISLIKE を主とした明示フィードバックを用いて Two‑Tower（ユーザー塔×アイテム塔）モデルを学習し、生成物をオブジェクトストレージ（Supabase Storage）へ配置するまでをまとめたものです。まずはローカルで学習し、のちに GitHub Actions へ移行できる構成とします。

## 目的とアウトプット

- 目的: ユーザー×動画の嗜好を Two‑Tower で学習し、初期の個人化推薦を可能にする。
- 埋め込み次元: 256 に固定。
- 学習アウトプット（本仕様）
  - モデル（PyTorch state_dict と ONNX）を Storage へ配置（Python/TypeScript から読み込み可能）

## ローカル開発 4 ステップ構成（Docker 実行）

1. **データ生成（prep）**  
   - レビューデータ（CSV）と DB の動画マスタ（CSV/DB 経由）を突合し、`reviewer_id, video_id, label` を含む学習データを `ml/data/` に出力。  
   - 統計（件数／欠損／結合率など）をログ化し、再実行時も安全に上書きできるようにする。

2. **学習（train）**  
   - 上記 Parquet を入力に Two‑Tower を学習。  
   - `ml/artifacts/` に成果物（`.pt`, `.onnx`, `user_embeddings.parquet`, `video_embeddings.parquet`, `model_meta.json`）を保存。ONNX はユーザー/アイテム特徴量を入力して埋め込みベクトルを生成できる推論モデルとする。

3. **評価（eval）**  
   - 学習済みモデル＋検証データから指標（例: AUC, Recall@K, MAP@K）を算出し、閾値を満たすか判定。  
   - PyTorch モデルと ONNX 推論の出力ベクトルを比較し、コサイン距離などで許容誤差内に収まることを確認。  
   - 結果を JSON などに保存し、NG の場合はアップサート処理へ進ませない。

4. **埋め込み反映（upsert / オンライン更新）**  
   - 評価 OK の成果物だけを Supabase Postgres の `public.user_embeddings` / `public.video_embeddings`（vector(256)）へアップサート。  
   - Storage には `models/two_tower_latest.onnx` 等をアップロードし、バージョン管理 (`two_tower_YYYYMMDDHHMM.*`) を行う。

> 各ステップは `scripts/<name>/run.sh` 経由で Docker 上から実行する。  
> 将来的な GitHub Actions 化では、この順番でジョブ／ステップを構成すればそのまま流用可能。

## データソース（優先: 明示フィードバック）

- 明示フィードバック（推奨）: `reviewer_id, product_url, label`（LIKE=1, DISLIKE=0）
  - 負例サンプリングは不要。DISLIKE が無い/極端に少ないユーザーのみ、任意で疑似負例を補う。
- 代替（レビュー由来の星評価）: `product_url, reviewer_id, stars`（例: `ml/data/raw/reviews/dmm_reviews_videoa_YYYY-MM-DD.csv`）
  - 簡易に「暗黙フィードバック化」して学習可能。正例（stars>=4）に対し、未観測アイテムから K 件の疑似負例をサンプリング。

### データソース統合（今回の前処理スクリプト仕様）

- 想定元ソース:
  - テーブル相当: `user_video_decisions`（LIKE/DISLIKE ベース、初期モデルでは未使用）
- 外部CSV: DMM レビュー `ml/data/raw/reviews/dmm_reviews_videoa_*.csv`
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
4) 生成物: モデル（state_dict, ONNX）、埋め込み Parquet、メタ情報  
5) （任意）評価メトリクス: Recall@K / MAP@K

## ONNX 推論入力の設計指針

- ユーザー塔・アイテム塔ともに ID ではなく特徴量テンソルを入力とする。例: 
  - ユーザー: 直近 LIKE のアイテム ID ベクトル、プロフィールカテゴリ、年齢層など
  - アイテム: タイトル/タグのテキスト埋め込み、カテゴリ ID、多値属性など
- 各特徴量は可変長を想定しつつ、ONNX では固定サイズかマスク付きテンソルとして受け取れるよう前処理で整形する。
- 将来の拡張を見据え、未使用フィールドがあってもフォーマットを壊さず追加できるようバージョニングする（`input_schema_version` をメタ情報に保持）。
- 特徴量エンコード（tokenize、カテゴリ ID 付与など）は学習時と推論時で同一ロジックを使う。ONNX に含めるか、前処理コンテナを共通化する。

## 運用サイクル想定

- **週次**: バッチ再学習（Two-Tower 全体を再学習し、新しい ONNX / state_dict / メタ情報を生成）
- **日次**: 新規スクレイピングで取得したアイテムに対し、ONNX を用いて特徴量→ベクトルを生成し `video_embeddings` を更新
- **適時**: ユーザーの行動変化に応じて特徴量を組み立て、ONNX で最新のユーザー埋め込みを生成し `user_embeddings` を更新
- 上記の各ステップは評価スクリプトの指標（Recall@K / AUC / MAP@K など）で前回比を監視し、急激な劣化があれば反映を止める

### データディレクトリ構造（ml/data/）

- `raw/` … 外部ソースそのままの CSV/JSON（例: `raw/reviews/dmm_reviews_videoa_YYYY-MM-DD.csv`）
- `processed/two_tower/latest/` … 直近の学習で使用する Parquet（`interactions_train.parquet` など）
- `processed/two_tower/runs/<run-id>/` … 各前処理実行のスナップショット（入力コピー、出力、`summary.json`）

> `scripts/prep_two_tower/run_with_remote_db.sh` はデフォルトで `processed/two_tower/latest/` を上書きしつつ、`--run-id` を付けると `processed/two_tower/runs/<run-id>/` に成果物を保存します。既存のダンプを使う場合は従来の `run.sh` に `--videos-csv` を渡すことも可能です。

## ローカル検証の流れ（Docker 前提）

> すべて `bash scripts/<name>/run.sh` から専用コンテナを起動して実行する。ホストの Python や venv は利用しない。

### 1. データ生成（prep）

```bash
# Remote DB 経由で動画マスタを取り込みつつデータ生成
bash scripts/prep_two_tower/run_with_remote_db.sh \
  --remote-db-url "$REMOTE_DATABASE_URL" \
  --mode reviews \
  --input ml/data/raw/reviews/dmm_reviews_videoa_YYYY-MM-DD.csv \
  --min-stars 4 --neg-per-pos 3 --val-ratio 0.2 \
  --run-id auto \
  --snapshot-inputs
```

- オプション: `--decisions-csv` で LIKE/決定ログをマージ、ローカルにダンプ済みの場合は `--videos-csv` を直接指定しても良い。
- このヘルパーはリモート DB から必要テーブルをダンプし、Docker 内に立てた Postgres にロードした上で `prep_two_tower_dataset.py` を実行する。
- 出力: 既定では `ml/data/processed/two_tower/latest/` に `interactions_train.parquet` などを上書き。結合失敗件数や CID 欠損はサマリ JSON/標準出力で確認。
- `--run-id auto` を指定すると `ml/data/processed/two_tower/runs/<timestamp>/` に入力/出力/summary をスナップショット保存（`--snapshot-inputs` で入力CSVもコピー）。

### 2. 学習（train）

```bash
bash scripts/train_two_tower/run.sh \
  --train ml/data/processed/two_tower/latest/interactions_train.parquet \
  --val ml/data/processed/two_tower/latest/interactions_val.parquet \
  --item-key video_id \
  --embedding-dim 256 \
  --epochs 5 --batch-size 2048 --lr 1e-3 \
  --out-dir ml/artifacts
```

- 出力: `ml/artifacts/` に `two_tower_latest.pt`, `two_tower_latest.onnx`, `user_embeddings.parquet`, `video_embeddings.parquet`, `model_meta.json` など。乱数シードやハイパーは `model_meta.json` に記録。
- ONNX は「特徴量テンソル → ユーザー／アイテム埋め込み」を返すネットワークとしてエクスポートし、ID マップは生成しない。
- 標準出力に epoch ごとの val loss を JSON で出力するので、ログ収集と比較が容易。

### 3. 評価（eval, 実装予定）

```bash
bash scripts/eval_two_tower/run.sh \
  --artifacts-dir ml/artifacts \
  --val ml/data/processed/two_tower/latest/interactions_val.parquet \
  --metrics-json ml/artifacts/metrics.json \
  --recall-k 20 --auc-threshold 0.6
```

- 期待出力: `metrics.json`（AUC, Recall@K, MAP@K など）。閾値を満たさない場合は終了コード ≠ 0 とし、後続ステップを止める。
- Dockerfile / requirements / run.sh を `scripts/eval_two_tower/` に追加し、共通の評価ロジックを実装予定。
- PyTorch で生成した埋め込みと ONNX で生成した埋め込みを比較し、コサイン距離・L2 誤差が許容範囲かを回帰テストする。

### 4. 埋め込み反映（upsert, 実装予定）

```bash
bash scripts/upsert_two_tower/run.sh \
  --artifacts-dir ml/artifacts \
  --env-file docker/env/dev.env \
  --dry-run
```

- 期待動作: `user_embeddings.parquet` / `video_embeddings.parquet` を pgvector テーブルへ upsert し、`two_tower_latest.onnx` などを Storage `models/` バケットへアップロード。
- `--dry-run=false` で実際に書き込み。成功時は更新件数と新モデル バージョンをログ出力する。

> 評価・アップサートのスクリプトは未実装。今後追加し、GitHub Actions から `prep → train → eval → upsert` の順に実行する構成を前提とする。

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
          python scripts/prep_two_tower/prep_two_tower_dataset.py \
            --mode explicit \
            --input ml/data/raw/reactions.csv \
            --out-train ml/data/processed/two_tower/latest/interactions_train.parquet \
            --out-val ml/data/processed/two_tower/latest/interactions_val.parquet \
            --run-id "$(date -u +%Y%m%dT%H%M%SZ)"

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
- [ ] ONNX で生成した埋め込みと PyTorch で生成した埋め込みが十分に一致（コサイン距離が閾値内）

## よくある論点

- 未登録 `product_url` の扱い: 前処理時に最小挿入→後続クロールで enrich（任意）。
- 埋め込み次元: 本仕様は 256 に固定。将来変更時はDBスキーマ・下流も一括更新。
- 評価: ユーザー履歴が薄い場合、popularity 混合や正則化で安定化。

---

以上です。スクリプトは `scripts/` 配下に追加済みです。必要に応じて Upsert 用スクリプトや GHA ワークフローも追補します。
