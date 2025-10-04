# Two-Tower Rebuild (v2)

本ディレクトリは Supabase の `videos` / `user_video_decisions` スキーマと互換を取りながら、Two-Tower レコメンドモデルを再構築するための Python パイプラインを収容しています。目的は次の 4 点です。

1. 疑似ユーザーデータ（`backend/data_processing/local_compatible_data`）や本番 DB から特徴量を抽出し、Two-Tower モデルを学習する。
2. 学習済み User Tower を ONNX 形式で出力し、Supabase Edge Functions（Deno/TypeScript + onnxruntime-web）から推論できるようにする。
3. 動画埋め込み（Item Tower 出力）を生成し、`video_embeddings` テーブルに一括アップサートできる成果物を出力する。
4. 推論に必要なメタ情報（特徴量スキーマ・正規化パラメータ・語彙辞書など）を JSON で出力し、CDN 経由で Edge Functions が取得できるようにする。

## ディレクトリ構成

```
backend/ml/two_tower_v2/
├── README.md
├── artifacts/                # エクスポート成果物配置先（例: user_tower.onnx, vocab_tag.json など）
├── config/                   # ハイパーパラメータや入出力設定
│   └── default.yaml
├── requirements.txt          # Python 依存関係
└── src/
    ├── data.py               # Supabase互換データのローディング・前処理
    ├── features.py           # 特徴量エンジニアリングとスキーマ生成
    ├── model.py              # PyTorch Two-Tower 実装
    ├── train.py              # 入口スクリプト（学習）
    ├── export.py             # ONNX および JSON 成果物の書き出し
    ├── generate_embeddings.py# Item Tower で全動画の埋め込みを生成
    └── utils.py              # 補助関数群
```

## 使い方

### 一括実行（推奨）

```
cd backend/ml/two_tower_v2
python run_local_pipeline.py
```

必要に応じて `--config` や入力データパスを上書きできます。エクスポートのみ/埋め込みのみをスキップしたい場合は `--skip-export` や `--skip-embeddings` オプションを指定してください。
デフォルトでは、疑似データセットとして `backend/data_processing/local_compatible_data` 以下の JSON を参照します。

本番 DB からデータを直接取得して学習する場合は `--use-remote` を付け、`SUPABASE_URL` と `SUPABASE_SERVICE_ROLE_KEY` を環境変数で指定してください（必要に応じて `--remote-output` で保存先ディレクトリ、`--remote-page-size` で取得ページサイズを変更可能）。

```bash
export SUPABASE_URL=...       # 例: https://xyz.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=...
python run_local_pipeline.py --use-remote
```

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

2. 学習を実行:
   ```bash
   python src/train.py --config config/default.yaml \
     --profiles backend/data_processing/local_compatible_data/profiles.json \
     --videos backend/data_processing/local_compatible_data/videos_subset.json \
     --decisions backend/data_processing/local_compatible_data/user_video_decisions.json
   ```

3. 成果物を書き出し（上記コマンドで保存された `checkpoints/latest.pt` を利用）:
   ```bash
   python src/export.py --config config/default.yaml --checkpoint checkpoints/latest.pt --output-dir artifacts
   ```

4. 動画埋め込みを生成（Supabase にアップサートする CSV or Parquet を作成）:
   ```bash
   python src/generate_embeddings.py --config config/default.yaml --checkpoint checkpoints/latest.pt --videos backend/data_processing/local_compatible_data/videos_subset.json --output artifacts/video_embeddings.parquet
   ```

   Supabase 上の本番データを利用したい場合は、先に `src/fetch_remote_data.py` を実行して JSON を取得するか、`run_local_pipeline.py --use-remote` を利用してください。

## 出力される成果物

- `user_tower.onnx` — Edge Functions から onnxruntime-web で読み込み可能な User Tower モデル
- `feature_schema.json` — 推論時に使用する特徴量の順序と型定義
- `normalizer.json` — 数値特徴量の平均・標準偏差
- `vocab_tag.json` / `vocab_actress.json` — 多値カテゴリ特徴量の語彙辞書
- `model_meta.json` — 署名、ハイパーパラメータ、学習日時を含むメタ情報
- `video_embeddings.parquet` — `video_id` とベクトル（`float32[embedding_dim]`）のテーブル

成果物一式は CI から CDN（例: Supabase Storage, CloudFront など）へアップロードし、Edge Functions で利用します。アップロード先のベース URL は `MODEL_ARTIFACT_BASE_URL` 環境変数として Edge Functions に設定してください。

### モデル入出力仕様

Two-Tower はユーザ側・アイテム側で同じ 256 次元の埋め込みを出力しますが、入力ベクトルの構築ロジックは `src/features.py` に定義されています。ローカル学習時は `backend/data_processing/local_compatible_data/` 以下の JSON（`profiles.json`, `videos_subset.json`, `user_video_decisions.json`）から特徴量を組み立て、`run_local_pipeline.py` 内で `FeaturePipeline` を保存します。

- **User Tower**
  - 入力テンソル名: `user_features`（`float32[225]`）。`FeaturePipeline.user_feature_dim` と一致します。
  - 入力生成プロセス:
    1. `UserFeatureStore.build_features(user_id)` が `profiles` と `user_video_decisions` からユーザごとの統計値を作成。
       - `numeric` 8 項目: アカウント年齢、選好価格の平均/中央値、LIKE/NOPE カウントや比率、直近 Like からの日数など（`NumericNormalizer` で z-score 正規化）。
       - `hour_of_day`: LIKE の発生時刻の平均を 0–1 に線形スケール。
       - `tag_vector`: LIKE した動画に含まれるタグ ID を語彙（`vocab_tag.json`）に基づき Bag-of-Words 集計 → L2 正規化。
       - `actress_vector`: 同様に女優語彙に基づく Bag-of-Words（現行データでは語彙サイズ 0 のため空配列）。
    2. `assemble_user_feature_vector(...)` がこれらの要素を順番に連結し、ONNX 入力ベクトルを作成。
  - 出力テンソル名: `user_embedding`（`float32[256]`）。`src/model.py` の `UserTower.forward` 内で L2 正規化済み。Edge Function では `computeUserEmbedding` → `inferUserEmbedding` により取得し、`user_embeddings` テーブルへ保存されます。

- **Item Tower**
  - 入力テンソル名: `item_features`（`float32[219]`）。`FeaturePipeline.item_feature_dim` と一致します。
  - 入力生成プロセス（商用環境）:
    1. **データ取得**: `src/fetch_remote_data.py` が Supabase REST API から `videos` テーブルを取得し、関連テーブル `video_tags`・`video_performers` を `select` 句で JOIN して JSON を生成します。
       - 主キー `videos.id`（UUID）
       - 数値/日時フィールド: `price`, `duration_seconds`, `distribution_started_at`, `product_released_at`, `published_at`, `created_at` など
       - メタ情報: `title`, `description`, `maker`, `label`, `series`, `image_urls` など（必要に応じてフィルタ可能）
       - タグ/出演者: join 結果を `video_tags ( tags ( name ) )`, `video_performers ( performers ( name ) )` として受け取り、`transform_videos()`（`fetch_remote_data.py` 内）で `tags` と `performers` の一次元配列へ正規化
       - 取得結果は `tmp/remote_data/videos_subset.json` に保存され、ローカル JSON が商用データのキャッシュとなります。
    2. **特徴量抽出**: `ItemFeatureStore.build_features(video_id)` が上記 JSON をロードした `pandas.DataFrame` からレコードを参照し、以下を生成します。
       - `numeric` 3 項目: `price`、最新リリース日との差分日数（`product_released_at` を基準）、`duration_seconds`。`NumericNormalizer` により z-score 正規化。
       - `tag_vector`: `tags` 配列を `vocab_tag.json` の語彙順に Bag-of-Words 化し L2 正規化。
       - `actress_vector`: `performers` 配列を同様に語彙化（現行は語彙サイズ 0 だが、構造は保持）。
    3. **テンソル化**: `assemble_item_feature_vector(...)` が上記セグメントを結合し、PyTorch/ONNX に渡す 1 次元ベクトル（`float32[219]`）を生成します。
  - 出力テンソル名: `item_embedding`（`float32[256]`）。`src/model.py` の `ItemTower.forward` で L2 正規化された結果。`generate_embeddings.py` が各動画 ID へ割り当て、`video_embeddings.parquet` に書き出します。

`feature_schema.json` には User Tower 入力のセグメント定義、`model_meta.json` には双方の入力次元と語彙サイズ、`normalizer.json` には数値特徴量の平均・標準偏差が記録されているので、ONNX 推論実装（Edge Function やバッチ処理）で同じ順序・スケールを再現する際の参照資料になります。

## Edge Function への導入手順

1. **モデル成果物の配置**
   - `artifacts/` に生成された以下のファイルを、公開アクセス可能な CDN / Supabase Storage バケットにアップロードします。
     - `user_tower.onnx`
     - `feature_schema.json`, `normalizer.json`, `vocab_tag.json`, `vocab_actress.json`, `model_meta.json`
     - onnxruntime-web の WASM 依存ファイル（`ort-wasm.wasm`, `ort-wasm-simd.wasm`, `ort-wasm-threaded.wasm` など）。
   - URL のベースパスを `MODEL_ARTIFACT_BASE_URL` として Edge Functions 側で参照できるようにしておきます。

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

1. **train ジョブ** — `run_local_pipeline.py --skip-embeddings` を実行し、学習・ONNX エクスポートのみを行います。成果物は `two-tower-training` アーティファクト（`artifacts/*`, `checkpoints/latest.pt`）として保存されます。
2. **embeddings ジョブ** — 入力 `generate_embeddings` もしくは `publish_artifacts` / `update_embeddings` が true のときに実行され、前段のアーティファクトを用いて `src/generate_embeddings.py` を再実行します。生成した `artifacts/video_embeddings.parquet` を `two-tower-embeddings` としてアップロードし、必要であれば Supabase Storage へのコピーや DB 更新を行います。

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

ローカルでワークフローの主要ステップを確認する場合は、仮想環境を作成して以下を参考にしてください。

```bash
cd backend/ml/two_tower_v2
python -m venv .venv_test
./.venv_test/bin/pip install -r requirements.txt
./.venv_test/bin/python run_local_pipeline.py
./.venv_test/bin/python upload_embeddings.py --dry-run --parquet artifacts/video_embeddings.parquet
```

- PyTorch や CUDA 関連ホイールのダウンロードで数百 MB 程度必要になるため、十分なネットワーク帯域とストレージ容量を確保してください。
- 検証後は `.venv_test` などの仮想環境やキャッシュを削除し、空き容量を回収することを推奨します。

## 今後の TODO

- 本番 DB から直接データを取得する SQL / Supabase API ラッパー
- A/B テストやメトリクス算出
- ハイパーパラメータチューニング用のスクリプト
