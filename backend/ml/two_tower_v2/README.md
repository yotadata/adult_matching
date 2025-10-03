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

### GitHub Actions による自動化

`.github/workflows/train-two-tower.yml` を用意しており、手動トリガー (`workflow_dispatch`) で以下を実行できます。

1. パイプライン実行 (`run_local_pipeline.py`) と成果物生成。
2. 任意で、Supabase Storage への成果物アップロード (`publish_artifacts` を true)。
3. 任意で、`video_embeddings` テーブルへのベクトル upsert と HNSW インデックス再構築 (`update_embeddings` を true)。
4. 本番データを利用した学習を行う場合は、実行時に `use_remote_data` を true にし、`SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` を Secrets に登録してください。

実行前に GitHub Secrets として以下を登録してください。

- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_PROJECT_ID`, `SUPABASE_DB_PASSWORD`, `SUPABASE_ACCESS_TOKEN`
- `MODEL_ARTIFACTS_BUCKET`（必要であれば `MODEL_ARTIFACTS_PREFIX`）

※ Supabase Storage を利用しない場合は `publish_artifacts` を false のまま実行し、生成された Workflows の artifact を手動で配布する運用でも問題ありません。

## 今後の TODO

- 本番 DB から直接データを取得する SQL / Supabase API ラッパー
- A/B テストやメトリクス算出
- ハイパーパラメータチューニング用のスクリプト
