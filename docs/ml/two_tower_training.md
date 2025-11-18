# Two‑Tower 学習ガイド（128 次元・PyTorch・ローカル優先）

本ドキュメントは、LIKE/DISLIKE を主とした明示フィードバックを用いて Two‑Tower（ユーザー塔×アイテム塔）モデルを学習し、生成物をオブジェクトストレージ（Supabase Storage）へ配置するまでをまとめたものです。まずはローカルで学習し、のちに GitHub Actions へ移行できる構成とします。

## 目的とアウトプット

- 目的: ユーザー×動画の好みを Two‑Tower で学習し、初期の個人化推薦を可能にする。
- 埋め込み次元: 128 に固定。
- 学習アウトプット（本仕様）
  - モデル（PyTorch state_dict と ONNX）を Storage へ配置（Python/TypeScript から読み込み可能）。成果物は `ml/artifacts/runs/<run-id>/` に保存し、直近の出力を `ml/artifacts/latest/` にミラーする。

## ローカル開発 4 ステップ構成（Docker 実行）

1. **データ生成（prep）**  
   - レビューデータ（CSV）と DB の動画マスタ（CSV/DB 経由）を突合し、`reviewer_id, video_id, label` を含む学習データを `ml/data/` に出力。  
   - 統計（件数／欠損／結合率など）をログ化し、再実行時も安全に上書きできるようにする。

2. **学習（train）**  
   - 上記 Parquet を入力に Two‑Tower を学習。  
   - `ml/artifacts/runs/<run-id>/` に成果物（`.pt`, `.onnx`, `user_embeddings.parquet`, `video_embeddings.parquet`, `model_meta.json`）を保存し、`ml/artifacts/latest/` に同期する。ONNX はユーザー/アイテム特徴量を入力して埋め込みベクトルを生成できる推論モデルとする。

3. **評価（eval）**  
   - 学習済みモデル＋検証データから指標（例: AUC, Recall@K, MAP@K）を算出し、閾値を満たすか判定。  
   - PyTorch モデルと ONNX 推論の出力ベクトルを比較し、コサイン距離などで許容誤差内に収まることを確認。  
   - 結果を JSON などに保存し、NG の場合はアップサート処理へ進ませない。

4. **埋め込み反映（upsert / オンライン更新）**  
   - 評価 OK の成果物だけを Supabase Postgres の `public.user_embeddings` / `public.video_embeddings`（halfvec(128)）へアップサート。  
   - Storage には `models/two_tower_latest.onnx` 等をアップロードし、バージョン管理 (`two_tower_YYYYMMDD_HHMMSS.*`) を行う。

> 各ステップは `scripts/<name>/run.sh` 経由で Docker 上から実行する。  
> 将来的な GitHub Actions 化では、この順番でジョブ／ステップを構成すればそのまま流用可能。

### クイック手順（手動再学習）

1. **レビュー/API 取得（必要に応じて）**
   ```bash
   # 例: FANZA API から取得（env は docker/env/prd.env を想定）
   GTE_RELEASE_DATE=2025-01-01 bash scripts/ingest_fanza/run.sh
   ```
2. **前処理**
   ```bash
   source docker/env/prd.env
   bash scripts/prep_two_tower/run_with_remote_db.sh \
     --project-ref "$SUPABASE_PROJECT_ID" -- \
     --mode reviews \
     --input ml/data/raw/reviews/dmm_reviews_videoa_2025-10-04.csv \
     --min-stars 4 --neg-per-pos 3 \
     --run-id auto --snapshot-inputs
   ```
   出力は `ml/data/processed/two_tower/latest/` と `runs/<timestamp>/` に保存される。
3. **学習**
   ```bash
bash scripts/train_two_tower/run.sh \
  --run-id auto \
  --train ml/data/processed/two_tower/latest/interactions_train.parquet \
  --val ml/data/processed/two_tower/latest/interactions_val.parquet \
  --user-features ml/data/processed/two_tower/latest/user_features.parquet \
  --item-features ml/data/processed/two_tower/latest/item_features.parquet \
  --item-key video_id \
  --embedding-dim 128 --hidden-dim 512 \
  --epochs 8 --batch-size 1024 \
  --max-tag-features 4096 --max-performer-features 1024 \
  --out-dir ml/artifacts \
  # 必要に応じて --disable-user-features を追加
   ```
   成果物は `ml/artifacts/runs/<run-id>/` に保存され、`ml/artifacts/latest/` が上書きされる。
4. **評価**
   ```bash
   bash scripts/eval_two_tower/run.sh \
     --artifacts-root ml/artifacts \
     --run-id latest \
     --val ml/data/processed/two_tower/latest/interactions_val.parquet \
     --recall-k 20
   ```
   `latest/metrics.json` と run ディレクトリ内の `metrics.json` が更新される。
5. **定性確認／アップサート**
   - `bash scripts/streamlit_qual_eval/run.sh`
   - 本番/ステージングのユーザー LIKE 履歴を元に最新のベクトルを生成する場合は、まず `bash scripts/gen_user_embeddings/run.sh --min-interactions 5 --env-file docker/env/prd.env` を実行する。`--min-interactions` に指定した値以上の LIKE（`public.user_video_decisions.decision_type = 'like'`）を持つユーザーのみが対象となり、集計は常に Supabase DB の最新データから取得される。
   - 生成済みの `ml/artifacts/live/*.parquet` をそのまま反映する場合は `bash scripts/upsert_two_tower/run.sh --artifacts-dir ml/artifacts/live --include-users --min-user-interactions 5` のように実行する（こちらも DB 上の LIKE 件数でフィルタされる。フォールバックとして学習用 parquet を参照するロジックは残しているが、基本は DB 集計が利用される）。
   - モデル成果物を Supabase Storage へ配置する際は、検証前に `bash scripts/publish_two_tower/run.sh --env-file docker/env/prd.env upload` で run_id ディレクトリへアップロードし、公開準備が整ったら `bash scripts/publish_two_tower/run.sh --env-file docker/env/prd.env activate` で manifest を切り替える。


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
3) Two‑Tower を PyTorch で学習（埋め込み128次元）  
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
- `processed/two_tower/runs/<run-id>/` … 各前処理実行のアーカイブ（入力コピー、出力、`summary.json`）

> `scripts/prep_two_tower/run_with_remote_db.sh` はデフォルトで `processed/two_tower/latest/` を上書きしつつ、`--run-id` を付けると `processed/two_tower/runs/<run-id>/` に成果物を保存します。ローカルにダンプした場合でも `run.sh` は常に `--db-url` で Postgres 接続を指定するようになり、`--videos-csv` オプションは廃止されました。

## ローカル検証の流れ（Docker 前提）

> すべて `bash scripts/<name>/run.sh` から専用コンテナを起動して実行する。ホストの Python や venv は利用しない。

### 1. データ生成（prep）

```bash
# Remote DB 経由で動画マスタを取り込みつつデータ生成
bash scripts/prep_two_tower/run_with_remote_db.sh \
  --remote-db-url "$REMOTE_DATABASE_URL" \
  --mode reviews \
  --input ml/data/raw/reviews/dmm_reviews_videoa_YYYY-MM-DD.csv \
  --min-stars 4 --max-negative-stars 3 \
  --neg-per-pos 3 --val-ratio 0.2 \
  --run-id auto \
  --snapshot-inputs
```

- オプション: `--decisions-csv` で LIKE/決定ログをマージ。ローカル DB にダンプ済みの動画マスタを使う場合も `--db-url` で接続先を指定する（`--videos-csv` は不要）。
- このヘルパーはリモート DB から必要テーブルをダンプし、Docker 内に立てた Postgres にロードした上で `prep_two_tower_dataset.py` を実行する。
- 出力: 既定では `ml/data/processed/two_tower/latest/` に `interactions_train.parquet`, `interactions_val.parquet`, `item_features.parquet`, `user_features.parquet` を上書き。`user_features.parquet` は `user_video_decisions` / `profiles` / `video_tags` を集約し、`recent_positive_video_ids`, `like_count_30d`, `positive_ratio_30d`, `signup_days`, `preferred_tag_ids` などを保持する。`--mode reviews` の場合は CSV 内のレビュー情報から同等の統計量を擬似生成し、DB なしでもユーザー特徴を確保する。また、`--max-negative-stars` を指定すると指定以下の星評価を明示的な負例 (`label=0`) として取り込み、SOD 等への偏りを抑制できる。結合失敗件数や CID 欠損はサマリ JSON/標準出力で確認。
- `--run-id auto` を指定すると `ml/data/processed/two_tower/runs/<timestamp>/` に入力/出力/summary をアーカイブ保存（`--snapshot-inputs` で入力CSVもコピー）。

### 2. 学習（train）

```bash
bash scripts/train_two_tower/run.sh \
  --run-id auto \
  --train ml/data/processed/two_tower/latest/interactions_train.parquet \
  --val ml/data/processed/two_tower/latest/interactions_val.parquet \
  --user-features ml/data/processed/two_tower/latest/user_features.parquet \
  --item-features ml/data/processed/two_tower/latest/item_features.parquet \
  --item-key video_id \
  --embedding-dim 128 \
  --hidden-dim 512 \
  --epochs 5 \
  --batch-size 1024 \
  --lr 1e-3 \
  --max-tag-features 2048 \
  --max-performer-features 512 \
  # --use-price-feature \
  --out-dir ml/artifacts
```

- `--max-tag-features` / `--max-performer-features` はタグ・出演者の多値カテゴリを頻度上位で打ち切るオプション。ローカル実行でメモリ使用量を抑えたい場合に指定する。0 以下を渡すと全候補を保持する。
- 価格を特徴量に含める場合は `--use-price-feature` を付与する。既定では価格を除外し、期待しないドメインバイアスを避ける。

- 出力: `ml/artifacts/runs/<run-id>/` に `two_tower_latest.pt`, `two_tower_latest.onnx`, `user_embeddings.parquet`, `video_embeddings.parquet`, `model_meta.json` などが保存され、`ml/artifacts/latest/` に同内容がコピーされる。`user_embeddings.parquet` / `video_embeddings.parquet` は `user_features.parquet` / `item_features.parquet` の特徴量を MLP エンコーダに通して得た埋め込みを保存する。
- ONNX 入力は `user_features` / `item_features`（いずれも `float32`）で、ID マップは生成しない。推論時は前処理で同一ベクトルを作成し ONNX に入力する。
- 標準出力に epoch ごとの val loss を JSON で出力するので、ログ収集と比較が容易。

### 3. 評価（eval, 実装予定）

```bash
bash scripts/eval_two_tower/run.sh \
  --artifacts-root ml/artifacts \
  --run-id latest \
  --val ml/data/processed/two_tower/latest/interactions_val.parquet \
  --recall-k 20
```

- 期待出力: `ml/artifacts/runs/<run-id>/metrics.json`（AUC, Recall@K, MAP@K など）。閾値を満たさない場合は終了コード ≠ 0 とし、後続ステップを止める。`--run-id latest` を指定すると `ml/artifacts/latest/metrics.json` へもコピーされる。
- Dockerfile / requirements / run.sh を `scripts/eval_two_tower/` に追加し、共通の評価ロジックを実装予定。
- PyTorch で生成した埋め込みと ONNX で生成した埋め込みを比較し、コサイン距離・L2 誤差が許容範囲かを回帰テストする。

### 4. 定性評価（Streamlit）

```
bash scripts/streamlit_qual_eval/run.sh
```

- ブラウザで `http://localhost:8501/` を開き、以下のタブで確認する。
  - **User-level recommendations**: Model A（既定 `ml/artifacts/latest/`）と任意の Model B を指定すると、同一ユーザーに対する推薦リストを左右で比較できる。既知アイテムを除外するかはサイドバーで切り替え可能。
  - **Dataset bias**: `item_features.parquet` の属性（maker / source / label / series / tags / performer_ids）ごとの出現数と正例数を可視化し、データの偏りを把握する。タグ文字列はカンマ区切りで分割、出演者は `performer_ids`（UUID）のまま集計する（現状は名前マッピング未実装）。
  - **Recommendation distribution**: 推薦結果に登場する属性分布（件数・ユニークアイテム数・到達ユーザー数）を集計し、モデルの偏りを定量的に確認する。タグはカンマ区切りで解析するため、独立したキーワード単位で確認できる。出演者は `performer_ids` の UUID で集計される。
- デフォルトでは Model A のみを読み込む。比較が必要な場合はサイドバーで Model B のアーティファクトパスを指定する（`user_embeddings.parquet` / `video_embeddings.parquet` / `metrics.json` が同ディレクトリにあること）。

### 5. ライブユーザー埋め込み生成（gen_user_embeddings）

訓練時は学習データに含まれる「仮想ユーザー」の埋め込みのみ生成される。実ユーザーの埋め込みを最新の決定履歴から作り直したい場合は、以下のスクリプトを実行して `ml/artifacts/live/user_embeddings.parquet` を作成する。

```bash
# 本番 DB から生成（Supabase プール経由で IPv4 に解決される）
bash scripts/gen_user_embeddings/run.sh \
  --env-file docker/env/prd.env \
  --output-dir ml/artifacts/live \
  --min-interactions 20    # LIKE が一定数あるユーザーのみ出力

# ローカル環境で検証する場合（dry-run で件数だけ確認）
bash scripts/gen_user_embeddings/run.sh \
  --env-file docker/env/dev.env \
  --dry-run \
  --limit-users 5
```

- 既定では最新アーティファクト (`ml/artifacts/latest/two_tower_latest.pt` / `model_meta.json`) と参照用 `user_features.parquet` から vocab を復元する。
- `--output-dir` を変えれば複数世代のライブ埋め込みを共存させられる。
- 実行時に **最小インタラクション数 (`--min-interactions`) を満たすユーザーのみ** を対象とし、学習時と異なるフィーチャー数だった場合は自動的にゼロ埋めで補正した上で推論する。
- `--dry-run` 以外で実行すると、完了後に `scripts/upsert_two_tower/run.sh --include-users` を自動起動して DB を更新する。生成だけ行いたい場合は `--dry-run` もしくは `--skip-upsert` を渡す。
- 自動 upsert 時はモデルメタ情報（`model_meta.json`）を `ml/artifacts/latest/` からコピーし、`--min-user-interactions 0` でフィルタリングせずに反映する。閾値を変えたい場合は `--skip-upsert` の上で `upsert_two_tower/run.sh` を手動実行する。

### 6. 埋め込み反映（upsert）

> 初回に `docker/db/init/001-roles.sql` が適用されていれば、Supabase のローカルスタック起動時に `CREATE EXTENSION vector;` が自動で通ります。古いボリュームを利用している場合は `docker compose --env-file docker/env/dev.env -f docker/compose.yml down -v` でリセットしてから再起動してください。

```bash
# 例: ローカル Supabase を更新
bash scripts/upsert_two_tower/run.sh \
  --env-file docker/env/dev.env \
  --artifacts-dir ml/artifacts/latest \
  --dry-run

# 例: 本番 Supabase を更新（動画埋め込みのみ）
bash scripts/upsert_two_tower/run.sh \
  --env-file docker/env/prd.env \
  --artifacts-dir ml/artifacts/latest

# 例: 本番で動画 + ライブユーザー埋め込みを反映
bash scripts/upsert_two_tower/run.sh \
  --env-file docker/env/prd.env \
  --artifacts-dir ml/artifacts/live \
  --include-users \
  --min-user-interactions 20
```
- `gen_user_embeddings/run.sh` を実行した場合は、既定で本スクリプトが自動的に呼ばれる（`--skip-upsert` を指定した場合を除く）。
- `--dry-run` を付けると件数だけ確認し、テーブルへは書き込まない。
- `--include-users` を指定すると `user_embeddings.parquet` も upsert する（事前に `gen_user_embeddings` で実ユーザーの埋め込みを生成しておく）。
- 埋め込み列は `halfvec(128)` であることを前提とする。スクリプト実行時に型/次元が異なると検知され、最新マイグレーションを適用するようエラーで知らせる。

- 期待動作: `user_embeddings.parquet` / `video_embeddings.parquet` を pgvector テーブルへ upsert し、`two_tower_latest.onnx` などを Storage `models/` バケットへアップロード。
- 実行ログには処理件数・対象 DB URL（IPv4/プールホストに分解済み）・必要に応じてテーブル再構成の情報が出力される。

## 日次動画埋め込み更新（自動）

新作動画が Supabase `public.videos` に追加された際は、Two‑Tower モデルを再学習せずに「既存モデルで推論した埋め込み」を追加入力するだけでよいケースが多い。この日次バッチでは以下を自動化する。

1. **最新モデル成果物の取得**  
   `scripts/publish_two_tower/run.sh fetch` が `models/two_tower/latest/manifest.json` を参照し、現在稼働中の ONNX / `.pt` / `model_meta.json` を `ml/artifacts/latest/` にダウンロードする。
2. **動画メタデータの同期**  
   `scripts/ingest_fanza/run.sh` を所定の期間（既定: 直近 3 日）で実行し、新着の作品を `videos`・`video_tags`・`video_performers` に upsert する。
3. **不足動画の埋め込み生成**  
   `scripts/gen_video_embeddings/run.sh` が Supabase DB から「`videos` に存在するが `video_embeddings` に未登録、または旧 `model_version` のまま」のレコードのみを抽出し、Two‑Tower の item encoder でベクトル化する。出力は `ml/artifacts/live/video_embedding_syncs/<JST-run-id>/video_embeddings.parquet` に保存され、完了後に `upsert_two_tower` が自動起動して差分を反映する。

ハンドオフ用スクリプトとして `scripts/sync_video_embeddings/run.sh` を追加した。基本的な利用例は以下の通り。

```bash
# 本番相当の環境変数を docker/env/prd.env に準備した状態で
SYNC_VIDEO_ENV_FILE=docker/env/prd.env \
SYNC_VIDEO_LOOKBACK_DAYS=3 \
bash scripts/sync_video_embeddings/run.sh

# 直近1日分だけ FANZA から再取得したい場合
bash scripts/sync_video_embeddings/run.sh --lookback-days 1

- `--skip-ingest` や `--skip-fetch` を付けると各工程を飛ばせる。`--` 以降の引数は `gen_video_embeddings/run.sh` にそのまま渡される（例: `--skip-upsert`）。
- `gen_video_embeddings` 単体でも利用可能。`--include-existing` を付けると最新モデルで全動画を再エンコードできる。
- 生成されたディレクトリには `summary.json` と `model_meta.json` が自動で配置されるため、後段のアップロードや検証に再利用しやすい。

### GitHub Actions による日次運用

`.github/workflows/sync-video-embeddings.yml` を追加し、UTC 18:00（JST 03:00）に上記 3 ステップを実行する。必要な Secrets は以下を想定:

| Secret 名 | 説明 |
| --- | --- |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` | プロジェクト URL と anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Storage / DB 書き込み用 |
| `REMOTE_DATABASE_URL` | pgvector を含む Postgres 接続文字列 |
| `SUPABASE_PROJECT_ID` (`SUPABASE_PROJECT_REF`) / `SUPABASE_REGION` | プール接続を解決するために利用（任意） |
| `FANZA_API_ID` / `FANZA_API_AFFILIATE_ID` / `FANZA_LINK_AFFILIATE_ID` | ingest 用 FANZA API 認証 |

Secrets を設定した後は手動 `workflow_dispatch` もしくはスケジュール実行で、日次の動画取り込みと埋め込み反映が走る。手動実行時は以下の入力を指定できる（省略可）:

| Input | 役割 |
| --- | --- |
| `start_date`, `end_date` | `GTE_RELEASE_DATE` / `LTE_RELEASE_DATE` を直接指定。`YYYY-MM-DD`（JST基準） |
| `lookback_days` | 日数で指定したい場合に利用（デフォルト 3 日） |
| `skip_ingest`, `skip_fetch` | FANZA 取得／Storage 取得ステップをパスするブール値（`true` / `false`） |
| `mode` | `full`（取得+埋め込み）/ `embeddings`（埋め込みのみ再実行） |

ローカルで同じ流れを再現したい場合は上記 `scripts/sync_video_embeddings/run.sh` を利用するだけでよい。

> Supabase の pooler へ接続する際は、専用の CA 証明書をローカルにも配置してください。以下のコマンドでダウンロードし、`PGSSLROOTCERT` をそのパスに設定してから実行すると安全です。

```bash
curl -sSLo docker/env/supabase-ca.crt \
  https://download.supabase.com/storage/v1/object/public/supabase-ca-certs/2021/root.crt
export PGSSLROOTCERT=docker/env/supabase-ca.crt
```

GitHub Actions では同証明書を自動で取得し、`.env` の `PGSSLROOTCERT` に設定しています。
リポジトリに登録する Secrets `SUPABASE_CA_CERT` には、上記 `supabase-ca.crt` を Base64 エンコードした文字列を登録してください（例: `base64 -w0 docker/env/supabase-ca.crt`）。

#### 6.1 環境変数一覧

sync パイプラインをローカル／CI で実行する際に必要となる主な環境変数の一覧です。`.env` ファイル（例: `docker/env/prd.env`）や GitHub Secrets へ設定してください。

| 変数名 | 必須 | 用途 |
| --- | :---: | --- |
| `SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_URL` | ○ | Supabase プロジェクトのベース URL（`https://<ref>.supabase.co`） |
| `SUPABASE_ANON_KEY` / `NEXT_PUBLIC_SUPABASE_ANON_KEY` | ○ | フロント／CLI で使用する anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | ○ | Storage/DB 書き込みを行うための Service Role key |
| `SUPABASE_SERVICE_ROLE_JWT_AUDIENCE` | △ | Supabase JS クライアント用の audience（通常 `authenticated`） |
| `REMOTE_DATABASE_URL` | ○ | 本番 Postgres 接続文字列（`postgresql://user:pass@host:port/db?sslmode=require`） |
| `SUPABASE_DB_URL` | △ | `REMOTE_DATABASE_URL` を再利用する場合にセット。未指定時は自動で補完される |
| `SUPABASE_PROJECT_ID` (`SUPABASE_PROJECT_REF`) | △ | Supabase プロジェクト Ref（プール接続補助） |
| `SUPABASE_REGION` | △ | `aws-1-ap-northeast-1` などのリージョン名（プール接続補助） |
| `SUPABASE_POOLER_HOST` / `SUPABASE_POOLER_PORT` | △ | プールホストを固定したい場合に利用（未指定でも自動推測） |
| `FANZA_API_ID` | ○ | DMM/FANZA API のアプリケーション ID |
| `FANZA_API_AFFILIATE_ID` | ○ | API 用アフィリエイト ID（動画データ取得時に使用） |
| `FANZA_LINK_AFFILIATE_ID` | ○ | リンク作成用のアフィリエイト ID（product_url 生成に使用） |
| `PGSSLMODE` / `PGSSLROOTCERT` | △ | GitHub Actions 等で SSL 接続する際の設定（例: `require` / `/etc/ssl/certs/ca-certificates.crt`） |
| `SYNC_VIDEO_LOOKBACK_DAYS` | △ | デフォルトの取得日数（未指定時は 3 日） |

※「△」は任意項目ですが、プロジェクトによっては必須になる場合があります。

> **補足:** 他コンポーネント（フロントエンド、Edge Functions、学習パイプライン等）で利用する環境変数も含めた網羅的な一覧は `docs/env/variables.md` を参照してください。環境変数を追加・更新した場合は同ドキュメントも忘れずに更新します。`SUPABASE_PROJECT_ID` を正として扱い、旧 `SUPABASE_PROJECT_REF` は後方互換用のみに残しています。

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
            --embedding-dim 128 --epochs 5 --batch-size 2048 --lr 1e-3

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
- バージョニング: `two_tower_YYYYMMDD_HHMMSS.pkl.gz` を保存しつつ、`two_tower_latest.pkl` を上書き
- 品質ゲート（推奨）:
  - 評価指標の下限（例: Recall@10 が前回値から大幅悪化しない）を満たさない場合は `latest` の更新を抑止
  - 失敗時フォールバック: 直近安定版の `latest` を維持し、埋め込み Upsert はパス
- 通知: Slack/メールで成功・失敗を通知

上記の GitHub Actions 雛形（Train TwoTower）は、手動実行 `workflow_dispatch` と週次 `schedule` の両方をサポートしています。そのまま週次更新として利用できます。

## 推薦API側の切替

- 短期: Edge Function で `video_embeddings` × `user_embeddings` の類似検索を実装（`halfvec_cosine_ops`）。
- 長期: モデルファイル（Storage）を学習・運用のバージョニングに使用（オンライン学習/AB テスト基盤へ拡張）。

## ローカル検証の最小チェックリスト

- [ ] `reviews.csv` → `interactions.parquet` が妥当（件数、ユーザー/動画数、星分布）
- [ ] 学習 1〜5 epoch で overfit しすぎない・loss 低下
- [ ] Recall@K（例: K=10）がランダムより十分に良い
- [ ] ONNX で生成した埋め込みと PyTorch で生成した埋め込みが十分に一致（コサイン距離が閾値内）

## よくある論点

- 未登録 `product_url` の扱い: 前処理時に最小挿入→後続クロールで enrich（任意）。
- 埋め込み次元: 本仕様は 128 に固定。将来変更時はDBスキーマ・下流も一括更新。
- 評価: ユーザー履歴が薄い場合、popularity 混合や正則化で安定化。

---

以上です。スクリプトは `scripts/` 配下に追加済みです。必要に応じて Upsert 用スクリプトや GHA ワークフローも追補します。
