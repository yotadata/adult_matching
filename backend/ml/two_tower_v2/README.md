# Two-Tower Rebuild (v2)

本ディレクトリは Supabase の `videos` / `user_video_decisions` スキーマと互換を取りながら、Two-Tower レコメンドモデルを再構築するための Python パイプラインを収容しています。目的は次の 4 点です。

## 最新アップデート / 進捗状況 (2025-10-06)
- Supabase RPC (`get_user_embedding_features`, `get_item_embedding_features`, バッチ版) を新設し、Edge Function・CI・学習パイプラインが共通JSONペイロードを参照する構成に移行しました。
- Edge Function の共通処理をRPC対応へ更新済み。`MODEL_ARTIFACT_BASE_URL` は Supabase Storage (`two-tower/v3/current/`) を指す前提です。
- 学習パイプラインはデフォルトでRPC経由の特徴量を利用するよう変更し、ローカルスキーマでも同じ関数を提供します。
- GitHub Actions `train-two-tower.yml` に manifest 生成＋Storage 2系統アップロード（バージョン/カレント）と `MODEL_ARTIFACT_BASE_URL` エクスポートを追加しました。
- `manifest.json` を生成する `scripts/build_manifest.py` を追加し、成果物ハッシュとメタデータの追跡を統一しました。
- 未実施: 本番/ステージングへのマイグレーション適用、RPC動作確認、CIワークフローの乾燥。本番切り替え前に各環境での検証が必要です。

1. Supabase 互換スキーマを持つ PostgreSQL から特徴量を抽出し、Two-Tower モデルを学習する。
2. 学習済み User Tower を ONNX 形式で出力し、Supabase Edge Functions（Deno/TypeScript + onnxruntime-web）から推論できるようにする。
3. 動画埋め込み（Item Tower 出力）を生成し、`video_embeddings` テーブルに一括アップサートできる成果物を出力する。
4. 推論に必要なメタ情報（特徴量スキーマ・正規化パラメータ・語彙辞書など）を JSON で出力し、CDN 経由で Edge Functions が取得できるようにする。

## ディレクトリ構成

```
backend/ml/two_tower_v2/
├── README.md
├── artifacts/                # エクスポート成果物配置先（例: user_tower.onnx, vocab_tag.json など）
├── checkpoints/              # 学習後のチェックポイント
├── config/                   # ハイパーパラメータや入出力設定 (default.yaml など)
├── infra/                    # モデル固有の Docker Compose 定義
│   └── docker-compose.yml
├── requirements.txt          # Python 依存関係
├── run_local_pipeline.py     # ローカル用パイプライン実行スクリプト
├── scripts/
│   ├── bootstrap_local_db.sh # ローカル Postgres にスキーマを適用
│   └── build_manifest.py     # artifacts/ 配下のハッシュとメタ情報を集約
├── sql/
│   └── local_schema.sql      # Two-Tower 用のテーブル定義
├── postgres_pipeline_spec.md # PostgreSQL 化の設計メモ
└── src/
    ├── data_utils.py         # 学習サンプル生成・データ分割ロジック
    ├── db.py                 # Postgres からの読み書きユーティリティ
    ├── features.py           # 特徴量エンジニアリングとスキーマ生成
    ├── model.py              # PyTorch Two-Tower 実装
    ├── train.py              # 学習スクリプト（Postgres を入力とする）
    ├── export.py             # ONNX および JSON 成果物の書き出し
    ├── generate_embeddings.py# Postgres から全動画を取得し埋め込み生成
    ├── pull_remote_into_pg.py# Supabase → Postgres へのデータ同期
    └── utils.py              # 補助関数群
```

## 使い方

### 前提

- Docker Desktop（WSL2 + Docker Desktop など）
- `psql` クライアント
- Supabase 実環境にアクセス可能な `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY`
- Docker Compose で `ankane/pgvector` イメージを利用（`infra/docker-compose.yml` 参照）

### 一括実行（推奨）

```bash
cd backend/ml/two_tower_v2

# 1. Postgres を起動
docker compose -f infra/docker-compose.yml up -d

# 2. スキーマ適用
./scripts/bootstrap_local_db.sh

# 3. （任意）Supabase から最新データを同期
export SUPABASE_URL=...
export SUPABASE_SERVICE_ROLE_KEY=...
python src/pull_remote_into_pg.py --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower

# 4. パイプライン実行
python run_local_pipeline.py \
  --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower \
  --fetch-remote
```

`--fetch-remote` を付けると実行前に Supabase から Postgres へデータ同期します（Supabase の環境変数が必須）。ローカル DB に既にデータがある場合は省略可能です。`--skip-export` や `--skip-embeddings` で後続ステップをスキップできます。

### Supabase RPC と特徴量

- `config/default.yaml` では `training.use_rpc_features=true` が既定。`src/features.py` の FeatureStore は Supabase RPC から取得した JSON をもとに特徴量を組み立てます。
- 利用する RPC
  - `get_user_embedding_features(user_id uuid)` / `get_user_embedding_feature_batch(limit, offset)`
  - `get_item_embedding_features(video_id uuid)` / `get_item_embedding_feature_batch(limit, offset)`
- 学習パイプライン・Edge Function・CI が同じペイロードを参照するため、学習と推論のズレを最小化できます。
- ローカルで RPC を呼べない状況（例: スタンドアロン検証）では、`training.use_rpc_features=false` に設定し、従来の pandas ベース集計にフォールバック可能です。

### 手動実行（詳細）

1. 依存関係をインストール:
   ```bash
   cd backend/ml/two_tower_v2
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   # uv を利用する場合
   uv pip install -r requirements.txt
   ```

2. Postgres を起動（初回のみ）:
   ```bash
   docker compose -f infra/docker-compose.yml up -d
   ```

3. スキーマ適用（初回のみ）:
   ```bash
   ./scripts/bootstrap_local_db.sh
   ```

4. （任意）Supabase からデータを同期:
   ```bash
   export SUPABASE_URL=...
   export SUPABASE_SERVICE_ROLE_KEY=...
   python src/pull_remote_into_pg.py --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower
   ```

5. 学習を実行:
   ```bash
   python src/train.py --config config/default.yaml \
     --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower \
     --output-dir checkpoints
   ```
   `training.use_rpc_features=true` の場合は Supabase RPC を呼び出して特徴量を構築します。
   ローカルの DataFrame から従来通り算出したい場合は設定ファイルで `use_rpc_features=false` に変更してください。

6. 成果物を書き出し（上記コマンドで保存された `checkpoints/latest.pt` を利用）:
   ```bash
   python src/export.py --config config/default.yaml --checkpoint checkpoints/latest.pt --output-dir artifacts
   ```

7. 動画埋め込みを生成（Supabase にアップサートする Parquet を作成）:
   ```bash
   python src/generate_embeddings.py \
     --config config/default.yaml \
     --checkpoint checkpoints/latest.pt \
     --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower \
     --output artifacts/video_embeddings.parquet
   ```

## 出力される成果物

- `user_tower.onnx` — Edge Functions から onnxruntime-web で読み込み可能な User Tower モデル
- `feature_schema.json` — 推論時に使用する特徴量の順序と型定義
- `normalizer.json` — 数値特徴量の平均・標準偏差
- `vocab_tag.json` / `vocab_actress.json` — 多値カテゴリ特徴量の語彙辞書
- `model_meta.json` — 署名、ハイパーパラメータ、学習日時を含むメタ情報
- `video_embeddings.parquet` — `video_id` とベクトル（`float32[embedding_dim]`）のテーブル
- `manifest.json` — `scripts/build_manifest.py` で生成される成果物一覧とハッシュ情報（Storage 配信向け）

成果物一式は CI から Supabase Object Storage にアップロードし、Edge Functions / CI 双方が `MODEL_ARTIFACT_BASE_URL` 経由で参照します。GitHub Actions (`train-two-tower.yml`) では `scripts/build_manifest.py` を用いて `manifest.json` を生成し、`two-tower/v3/<version>/` および `two-tower/v3/current/` プレフィックスへ同期します。手動アップロード時も同スクリプトを利用してください。

### モデル入出力仕様

Two-Tower はユーザ側・アイテム側で同じ 256 次元の埋め込みを出力しますが、入力ベクトルの構築ロジックは `src/features.py` に定義されています。ローカル学習時・CI では `backend/ml/two_tower_v2/infra/docker-compose.yml` で起動した Postgres からデータを取得し、`run_local_pipeline.py` が `FeaturePipeline` を保存します。

- **User Tower**
  - 入力テンソル名: `user_features`（`float32[225]`）。`FeaturePipeline.user_feature_dim` と一致します。
  - 入力生成プロセス:
    1. `UserFeatureStore.build_features(user_id)` が（既定設定では Supabase RPC を経由して）ユーザ統計量を取得し、
       必要な数値・カテゴリ特徴に整形。
       - `numeric` 8 項目: アカウント年齢、選好価格の平均/中央値、LIKE/NOPE カウントや比率、直近 Like からの日数など（`NumericNormalizer` で z-score 正規化）。
       - `hour_of_day`: LIKE の発生時刻の平均を 0–1 に線形スケール。
       - `tag_vector`: LIKE した動画に含まれるタグ ID を語彙（`vocab_tag.json`）に基づき Bag-of-Words 集計 → L2 正規化。
       - `actress_vector`: 同様に女優語彙に基づく Bag-of-Words（現行データでは語彙サイズ 0 のため空配列）。
    2. `assemble_user_feature_vector(...)` がこれらの要素を順番に連結し、ONNX 入力ベクトルを作成。
  - 出力テンソル名: `user_embedding`（`float32[256]`）。`src/model.py` の `UserTower.forward` 内で L2 正規化済み。Edge Function では `computeUserEmbedding` → `inferUserEmbedding` により取得し、`user_embeddings` テーブルへ保存されます。

- **Item Tower**
  - 入力テンソル名: `item_features`（`float32[219]`）。`FeaturePipeline.item_feature_dim` と一致します。
  - 入力生成プロセス（商用環境）:
    1. **データ取得**: `src/pull_remote_into_pg.py` が Supabase REST API から `videos` テーブルと `video_tags` / `video_performers` を取得し、モデル専用の Postgres に upsert。推論時は `get_item_embedding_features` RPC を介して同じ統計情報を取得します。
       - 主キー `videos.id`（UUID）
       - 数値/日時フィールド: `price`, `duration_seconds`, `distribution_started_at`, `product_released_at`, `published_at`, `created_at` など
       - メタ情報: `title`, `description`, `maker`, `label`, `series`, `image_urls` など（必要に応じてフィルタ可能）
       - タグ/出演者: JOIN 結果を `transform_videos()`（`pull_remote_into_pg.py` 内）で整形し、`video_tags` / `video_performers` テーブルとして保存
    2. **特徴量抽出**: `ItemFeatureStore.build_features(video_id)` が Postgres から読み込んだ `pandas.DataFrame` を参照し、以下を生成します。
       - `numeric` 3 項目: `price`、最新リリース日との差分日数（`product_released_at` を基準）、`duration_seconds`。`NumericNormalizer` により z-score 正規化。
       - `tag_vector`: `tags` 配列を `vocab_tag.json` の語彙順に Bag-of-Words 化し L2 正規化。
       - `actress_vector`: `performers` 配列を同様に語彙化（現行は語彙サイズ 0 だが、構造は保持）。
    3. **テンソル化**: `assemble_item_feature_vector(...)` が上記セグメントを結合し、PyTorch/ONNX に渡す 1 次元ベクトル（`float32[219]`）を生成します。
  - 出力テンソル名: `item_embedding`（`float32[256]`）。`src/model.py` の `ItemTower.forward` で L2 正規化された結果。`generate_embeddings.py` が各動画 ID へ割り当て、`video_embeddings.parquet` に書き出します。

`feature_schema.json` には User Tower 入力のセグメント定義、`model_meta.json` には双方の入力次元と語彙サイズ、`normalizer.json` には数値特徴量の平均・標準偏差が記録されているので、ONNX 推論実装（Edge Function やバッチ処理）で同じ順序・スケールを再現する際の参照資料になります。

## Edge Function への導入手順

1. **モデル成果物の配置**
   - `artifacts/` に生成された以下のファイルを、Supabase Storage の公開バケット（推奨: `model-artifacts/two-tower/v3/<version>/`）にアップロードします。
     - `user_tower.onnx`
     - `feature_schema.json`, `normalizer.json`, `vocab_tag.json`, `vocab_actress.json`, `model_meta.json`, `manifest.json`
     - onnxruntime-web の WASM 依存ファイル（`ort-wasm.wasm`, `ort-wasm-simd.wasm`, `ort-wasm-threaded.wasm` など）。
   - `manifest.json` には成果物のハッシュや生成メタ情報が記録されています。`scripts/build_manifest.py` を再利用すれば手動アップロード時も同じ形式を作れます。
   - 最新版を指す `two-tower/v3/current/` プレフィックスを維持し、その URL を `MODEL_ARTIFACT_BASE_URL` として Edge Functions 側で参照してください。

2. **環境変数の設定**
   Edge Functions 実行時に以下の変数を読み込ませてください。
   - `MODEL_ARTIFACT_BASE_URL` : アーティファクトが配置された CDN パス（末尾スラッシュ不要）。
   - `SUPABASE_URL`, `SUPABASE_ANON_KEY` : 通常の Supabase クライアント用。
   - `SUPABASE_SERVICE_ROLE_KEY`（または `SUPABASE_SERVICE_KEY`）: 履歴取得・埋め込み保存のためサービスロール権限が必須。
   例: `.env.production` を `supabase functions serve` の `--env-file` に渡す、またはホスティング環境の環境変数に登録する。

3. **ローカルでの動作確認**
   ```bash
   # 環境変数を読み込み（例: .env.production を利用）
   set -a
   source ../../.env.production
   set +a

   # ユーザ埋め込み生成
   supabase functions serve embed-user

   # 推薦エンドポイント
   supabase functions serve ai-recommend-v5
   ```

   別ターミナルから `curl` で API を叩き、結果が得られることを確認します。
   ```bash
   curl -X POST http://localhost:54321/functions/v1/embed-user \
     -H "Content-Type: application/json" \
     -d '{"user_id": "<UUID>", "force": true}'

   curl -X POST http://localhost:54321/functions/v1/ai-recommend-v5 \
     -H "Content-Type: application/json" \
     -d '{"user_id": "<UUID>", "limit": 20}'
   ```

4. **本番デプロイ**
   Supabase CLI を利用して Edge Functions をデプロイします。
   ```bash
   supabase functions deploy embed-user --project-ref <project-ref>
   supabase functions deploy ai-recommend-v5 --project-ref <project-ref>
   ```
   デプロイ後、`.env.production` と同等の環境変数を Supabase プロジェクト設定に登録してください。
   ベクトルの更新を手動で行う場合は、`upload_embeddings.py` を利用して REST API 経由で `video_embeddings` に upsert できます。

#### ローカルテスト時の注意点

- `supabase functions serve` は内部的に Docker コンテナを利用します。Docker デーモンへ接続できない環境（例: Docker ソケットへのアクセスが禁止されている CI や制限付きサーバ）では `permission denied while trying to connect to the Docker daemon socket` と表示され起動できないため、その場合は Docker が利用可能な端末でテストしてください。
- Edge Functions が参照する Two-Tower 成果物（`user_tower.onnx`, `feature_schema.json`, `ort-wasm*.wasm` など）は `MODEL_ARTIFACT_BASE_URL` から取得します。ローカル検証時は `python -m http.server 8001` などで `backend/ml/two_tower_v2/artifacts` を配信し、`MODEL_ARTIFACT_BASE_URL=http://127.0.0.1:8001` のように設定します。ポートバインドが禁止されている環境では `PermissionError: [Errno 1] Operation not permitted` になるため、別環境で実施するか CDN 等を利用してください。
- `.env` ファイルを `source` する際、値にスペースが含まれると `set -a; source ...` でコマンドエラーになる場合があります。Supabase CLI の `--env-file` オプションを使うか、必要に応じて値を引用符で囲んでください。

### GitHub Actions による自動化

`.github/workflows/train-two-tower.yml` は手動トリガー (`workflow_dispatch`) を想定した 2 ジョブ構成です。

1. **train ジョブ** — Postgres コンテナを起動し、`run_local_pipeline.py --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower --skip-embeddings` を実行します（`use_remote_data` が true の場合は Supabase から同期してから学習）。成果物は `two-tower-training` アーティファクト（`artifacts/*`, `checkpoints/latest.pt`）として保存されます。
2. **embeddings ジョブ** — 同様に Postgres コンテナを起動し、`src/generate_embeddings.py --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower` を実行して `artifacts/video_embeddings.parquet` を生成します。入力 `generate_embeddings` もしくは `publish_artifacts` / `update_embeddings` が true のときにのみ動作し、必要であれば Supabase Storage へのコピーや DB 更新を行います。

主な workflow inputs:

- `generate_embeddings` (default: true) — embeddings ジョブを動かすか制御します。false にすると学習のみ実施されます。
- `publish_artifacts` — Supabase Storage へ `artifacts/` 一式（ONNX + JSON + embeddings）をコピーします。
- `update_embeddings` — REST API 経由で `video_embeddings` に upsert し、HNSW インデックスを再構築します。
- `use_remote_data` — Supabase からデータを取得して学習・埋め込み生成を行います（`SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` が必須）。

実行前に GitHub Secrets として以下を登録してください。

- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_PROJECT_ID`, `SUPABASE_DB_PASSWORD`, `SUPABASE_ACCESS_TOKEN`
- `MODEL_ARTIFACTS_BUCKET`（必要であれば `MODEL_ARTIFACTS_PREFIX`）

※ Supabase Storage を利用しない場合は `publish_artifacts` を false のまま実行し、生成された Workflows の artifact を手動で配布する運用でも問題ありません。

`publish_artifacts=true` の場合は以下が自動で行われます。
- `scripts/build_manifest.py` による `manifest.json` 生成（成果物の SHA256 / サイズ / メタ情報を格納）
- `model-artifacts/<prefix>/<timestamp-commit>/` への成果物コピー
- `model-artifacts/<prefix>/current/` への同期、および `MODEL_ARTIFACT_BASE_URL` 環境変数のエクスポート（Edge Function デプロイや後続ジョブで再利用可能）

ローカルでワークフローの主要ステップを確認する場合は、仮想環境を作成して以下を参考にしてください。

```bash
cd backend/ml/two_tower_v2
python -m venv .venv_test
./.venv_test/bin/pip install -r requirements.txt
./scripts/bootstrap_local_db.sh
./.venv_test/bin/python run_local_pipeline.py --pg-dsn postgresql://two_tower:two_tower@localhost:5433/two_tower --fetch-remote
./.venv_test/bin/python upload_embeddings.py --dry-run --parquet artifacts/video_embeddings.parquet
```

- PyTorch や CUDA 関連ホイールのダウンロードで数百 MB 程度必要になるため、十分なネットワーク帯域とストレージ容量を確保してください。
- 検証後は `.venv_test` などの仮想環境やキャッシュを削除し、空き容量を回収することを推奨します。

## 今後の TODO

- ステージング / 本番 Supabase へのマイグレーション適用と RPC 動作確認
- `MODEL_ARTIFACT_BASE_URL` を更新した Edge Function の再デプロイとエンドツーエンド検証
- GitHub Actions ワークフローのドライラン（Storage アップロード / `MODEL_ARTIFACT_BASE_URL` 伝播の確認）
- A/B テストやメトリクス算出の仕組み作り
- ハイパーパラメータチューニング用のスクリプト整備
