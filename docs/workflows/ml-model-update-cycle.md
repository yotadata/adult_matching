# TwoTower モデル更新ワークフロー

このドキュメントは、ローカルで行っている TwoTower 学習（`docs/ml/two_tower_training.md` 参照）を GitHub Actions に移行した際のワークフロー構成と、学習→評価→リリースの判断ポイントを整理したものです。  
ローカル実行との差分は主に出力先（`ml/artifacts/` → Supabase Storage/`ml/artifacts`）と、GHA 上での `workflow_dispatch` トリガーだけです。

## 1. Training ワークフロー（`.github/workflows/ml-train-model.yml`）

### 実行対象
- `scripts/prep_two_tower/run_with_remote_db.sh`： 直近の `reviews` / `like` データと動画マスタを `ml/data/processed/two_tower/latest/` に整形。  
- `scripts/train_two_tower/run.sh`： TwoTower モデル（128d）を学習し `ml/artifacts/runs/<run-id>/` に保存。  
- `scripts/eval_two_tower/run.sh`：学習済みモデルを評価し `metrics.json` を生成。  

### 入出力
- `workflow_dispatch` の `run_id`（任意）/`mode`/`min_stars`/`neg_per_pos` を受けて `RUN_ID` を決定（指定がない場合は UTC タイムスタンプ）。  
- `ml/artifacts/runs/<RUN_ID>/` にモデル・メタ・metrics を生成し、`ml/artifacts/latest/` を更新。  
- `two_tower_eval` アーティファクトとして `ml/artifacts/runs` をアップロード。

### 判断（GO/NO-GO）
評価指標（Recall@K, AUC など）を `ml/artifacts/runs/<RUN_ID>/metrics.json` で確認し、結果に応じて手動で `ml-release-embeddings` を起動します。ここで `workflow_dispatch` の `approved=yes` を送ることで信号を出します。

## 2. Release ワークフロー（`.github/workflows/ml-release-embeddings.yml`）

### 前提
- `approved=yes` を `workflow_dispatch` で渡すこと。  
- `SUPABASE_*`, `REMOTE_DATABASE_URL`, `FANZA_*`, `SUPABASE_CA_CERT` などの Secrets を事前設定。  

### 実行内容
1. `scripts/sync_video_embeddings/run.sh`：`ml/artifacts/live/video_embedding_syncs/<RUN_ID>` に差分埋め込みを生成して `public.video_embeddings` に upsert。  
2. `scripts/sync_user_embeddings/run.sh`：`user_embeddings` を最新の `FANZA` 評価履歴で再計算。  
3. `scripts/publish_two_tower/run.sh`：`models/two_tower_latest.onnx` や `manifest.json` を Storage にアップロードし、`two_tower_latest` が最新モデルを指すよう manifest を更新。

### 入力
- `run_id`（省略可）、`lookback_days`（動画埋め込みの遡り日数）、`recent_hours`（ユーザー埋め込み対象期間）。  
- `approved` は `yes` でないと実行されない。  

## 3. 評価アーティファクトと GO の管理

- 学習結果 + 評価結果は `ml/artifacts/runs/<run-id>/` に保存されるため、レビュー担当者は `metrics.json` / `summary.json` を参照して GO 判断。  
- `ml-train-model` の `workflow_dispatch` 入力 `run_id` を `ml-release-embeddings` に手動で渡すことで、同じ `RUN_ID` を一貫して参照可能。

## 4. 実装上の注意・差分

- **gen_video_embeddings** は、埋め込み同期フェーズの差分処理なので `ml-train-model` には含めず `ml-release-embeddings` 側で `sync_video_embeddings` が担当します。  
- **Storage への保存先**：ローカルでは `ml/artifacts/` に保存するが、GHA では `scripts/publish_two_tower` が `models/` バケットへアップロードし `manifest.json` を更新する。  
- **CA 証明書**：`SUPABASE_CA_CERT` を Secrets に設定し、ワークフロー内でデコードして `docker/env/prd.ci.env` へ書き込む（2つのワークフローで共通）。

## 5. 必要な Secrets / Env

上記 docs/ml/two_tower_training.md と同じ。学習/リリースそれぞれで `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `REMOTE_DATABASE_URL`, `FANZA_*`, `SUPABASE_CA_CERT` などを揃えてください。

## 6. 今後の拡張例

- Discord への通知: `ml-train-model` と `ml-release-embeddings` で `DISCORD_WEBHOOK_URL` を Secrets に設定すれば、各ジョブ完了時に Discord へステータス（Success/Failure）と `RUN_ID` を送信するようにしてあるので、通知先を追加できます。Secret が空なら通知はスキップされます。

- `ml-train-model` を週次 `schedule: '0 4 * * 0'` で自動化。  
- `ml-release-embeddings` は Slack 通知や issue コメントで「GO」→ `workflow_dispatch` triggers へ追加。  
- TBD: 評価結果を `doc` とは別に `ml/artifacts/latest/metrics-latest.json` などにコピーして参照しやすくする。
