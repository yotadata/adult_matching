# Two-Tower v2 I/O とアーティファクト概要

## パイプライン全体の入力
- **データソース（Postgres / Supabase）**: `profiles`、`videos`、`user_video_decisions`、`video_tags`、`video_performers` テーブル。Supabase を利用する場合は RPC `get_user_embedding_features` / `get_item_embedding_features` が同内容の JSON を提供する。
- **設定ファイル**: `config/default.yaml`（ハイパーパラメータ、語彙制約、RPC 利用可否、エクスポート設定）。
- **環境変数**:
  - `SUPABASE_URL`、`SUPABASE_SERVICE_ROLE_KEY`（リモート同期や埋め込みアップロード時に使用）。
  - 任意の `GITHUB_SHA`（manifest 生成時のメタデータに埋め込み）。
  - `MODEL_ARTIFACT_BASE_URL`（推論側設定用、パイプライン実行には必須ではない）。
- **コマンドライン引数**: `run_local_pipeline.py` および各スクリプトの `--pg-dsn`、`--config`、`--fetch-remote`、`--skip-export`、`--skip-embeddings`、`--page-size`、`--batch-size`、`--version` など。

## パイプライン全体の出力
- **チェックポイント**: `checkpoints/latest.pt`（PyTorch モデル重み、特徴量パイプライン、設定スナップショット）。
- **推論成果物**: `artifacts/` ディレクトリに生成される ONNX、スキーマ／正規化器／語彙 JSON、`model_meta.json`、`manifest.json` など。
- **アイテム埋め込み**: `artifacts/video_embeddings.parquet`（`video_id` と浮動小数ベクトル）。
- **Supabase 反映結果（任意）**: `upload_embeddings.py` や `supabase storage cp` を実行した場合、Supabase Postgres の `video_embeddings` テーブルおよび Object Storage にアップロードされる。

## パイプライン I/O ステージ別一覧
| ステージ | エントリーポイント | 主な入力 | 出力 | 補足 |
| --- | --- | --- | --- | --- |
| リモート同期（任意） | `src/pull_remote_into_pg.py` | Supabase REST/RPC、`SUPABASE_URL`、`SUPABASE_SERVICE_ROLE_KEY`、`--pg-dsn` | ローカル Postgres に同期したレコード群 | `run_local_pipeline.py --fetch-remote` の下請け。最新データを学習に利用したい場合。 |
| データ抽出 | `src/train.py`（内部で `db.py`） | ローカル Postgres DSN、`config/default.yaml` | Pandas DataFrame（`profiles_df`、`videos_df`、`decisions_df`） | スキーマ互換テーブルから学習用データを取得。 |
| 特徴量生成 | `features.build_feature_pipeline` | DataFrame、語彙制約（設定）、必要に応じて RPC | `FeaturePipeline`、`UserFeatureStore`、`ItemFeatureStore`、正規化器 | RPC 利用時は Supabase 側で集約済みの特徴量を取得。 |
| 学習 | `src/train.py` | 特徴量ストア、PyTorch ハイパーパラメータ | `checkpoints/latest.pt`、学習ログ | ベストモデルの state dict とパイプラインメタデータを保存。 |
| エクスポート | `src/export.py` | チェックポイント、設定 | `artifacts/user_tower.onnx`、スキーマ／正規化器／語彙／メタ情報 | ユーザー塔を ONNX 化、推論依存ファイルを生成。 |
| 埋め込み生成 | `src/generate_embeddings.py` | チェックポイント、Postgres DSN、設定 | `artifacts/video_embeddings.parquet` | 動画ごとのアイテム埋め込みを一括生成。 |
| マニフェスト生成 | `scripts/build_manifest.py` | `artifacts/`、`--version`、任意 `--commit-sha` | `artifacts/manifest.json` | Storage 公開時のハッシュとメタデータを収集。 |
| 埋め込みアップロード（任意） | `upload_embeddings.py` | `artifacts/video_embeddings.parquet`、Supabase REST、サービスロール鍵 | Supabase `video_embeddings` テーブルの upsert | `--dry-run` で事前検証が可能。 |
| Storage 反映（任意） | Supabase CLI | `artifacts/` バンドル、バケット `model-artifacts`、プレフィックス | Supabase Object Storage 上の成果物資産 | `MODEL_ARTIFACT_BASE_URL` を介し Edge Function 等から参照。 |

## アーティファクト一覧
| アーティファクト | 生成元 | 内容 | 利用者 | 備考 |
| --- | --- | --- | --- | --- |
| `checkpoints/latest.pt` | `src/train.py` | PyTorch state dict、特徴量パイプライン、設定 | `src/export.py`、`src/generate_embeddings.py` | 内部利用。Storage には原則公開しない。 |
| `artifacts/user_tower.onnx` | `src/export.py` | L2 正規化済みユーザー塔モデル | Supabase Edge Function、CI 推論テスト | 入力名は設定で変更可能（既定 `user_features`）。 |
| `artifacts/feature_schema.json` | `src/export.py` | 特徴量セグメントと次元情報 | Edge Function、埋め込み生成 | 推論側が特徴量を再構築するための仕様書。 |
| `artifacts/normalizer.json` | `src/export.py` | 数値特徴の平均・分散 | Edge Function、埋め込み生成 | 学習／推論のスケーリング整合性を維持。 |
| `artifacts/vocab_tag.json` | `src/export.py` | タグ語彙一覧 | Edge Function、埋め込み生成 | L2 正規化 Bag-of-Words 用。 |
| `artifacts/vocab_actress.json` | `src/export.py` | 出演者語彙一覧 | Edge Function、埋め込み生成 |  |
| `artifacts/model_meta.json` | `src/export.py` | 埋め込み次元、語彙サイズ、設定など | 監視、マニフェスト | Manifest に取り込まれ、ロールアウト追跡に利用。 |
| `artifacts/video_embeddings.parquet` | `src/generate_embeddings.py` | `video_id` と 256 次元ベクトル | Supabase REST、分析処理 | `upload_embeddings.py` で upsert。 |
| `artifacts/manifest.json` | `scripts/build_manifest.py` | ファイル一覧、SHA256、サイズ、バージョン | Storage 利用者、検証ツール | `current/` プレフィックス更新時に同期。 |
| ONNX Runtime バイナリ（任意） | 手動配置 | `ort-wasm*.wasm`、`ort-wasm*.js` | Edge Function | 推論スタックを Storage と一緒に配布する場合に同梱。 |

## Supabase Storage 推奨構成
- バケット: `model-artifacts`
- バージョン付きパス: `model-artifacts/two-tower/v3/<timestamp-or-tag>/`
  - `artifacts/` 内の成果物と任意の ONNX Runtime 資材を格納。
- 安定プレフィックス: `model-artifacts/two-tower/v3/current/`
  - 最新版の `manifest.json` と主要ファイルを同期。
- 公開 URL 例: `https://<project>.supabase.co/storage/v1/object/public/model-artifacts/two-tower/v3/current`
  - Edge Function から署名なしで利用する想定。

## 公開チェックリスト
1. `run_local_pipeline.py` で学習・エクスポート・埋め込み生成まで完走する。
2. 一意な `--version` を指定して `scripts/build_manifest.py` で `manifest.json` を生成する。
3. Supabase CLI 等で `artifacts/` をバージョン付きプレフィックスにアップロードする。
4. `current/` プレフィックスを最新の成果物に差し替える。
5. （任意）`upload_embeddings.py` で `video_embeddings.parquet` を Supabase に upsert する。
6. Edge Function の `MODEL_ARTIFACT_BASE_URL` を更新し、推論系で取得確認を行う。

## データ契約
- RPC が返却する JSON 構造は常に `feature_schema.json` と互換でなければならない。変更する場合は再学習と成果物再公開が必要。
- 埋め込み次元 (`embedding_dim`) は ONNX、スキーマ、Parquet、ANN インデックスで共通の契約値。
- `manifest.json` には最新の `model_meta.json` を含め、デプロイ時の検証とロールバック判断を容易にする。
