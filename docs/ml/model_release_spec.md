# Two‑Tower モデルリリース仕様（案）

本ドキュメントは、学習済み Two‑Tower モデルを Supabase Storage に配置する際のガイドラインを定めるためのドラフトです。命名規則・リリース手順・付随メタデータの保持方式を整理し、運用設計のたたき台とします。最終確定前に関係者でレビューし、必要に応じて修正してください。

## 1. ストレージ構成

- バケット: `models`
- 階層構造（例）
  ```
  models/
    two_tower/
      20251031T093831Z/          # run_id（UTC タイムスタンプ or 任意ID）
        two_tower_20251031T093831Z.onnx
        two_tower_20251031T093831Z.pt
        model_meta.json
        metrics.json
        summary.md                # 任意（変更点・注意点）
      20251015T112233Z/
        ...
      latest/
        manifest.json             # 現行リリースのポインタ
  ```
- `run_id` ディレクトリは再学習ごとに追加し、履歴を残す（最低 n=3 バージョン保持を想定）。
- `latest/manifest.json` には現在稼働中のバージョン情報を記録し、クライアントやワークフローはこのファイルを起点に参照する。

## 2. ファイル命名規則

| 対象 | 命名例 | 備考 |
| ---- | ------ | ---- |
| ONNX | `two_tower_<run_id>.onnx` | 推論用（ユーザー/アイテム両対応） |
| PyTorch state_dict | `two_tower_<run_id>.pt` | 再学習・デバッグ用の元モデル |
| メタ情報 | `model_meta.json` | 特徴量次元・語彙サイズ等を含む |
| 評価結果 | `metrics.json` | 指標（Recall@K, AUC など） |
| マニフェスト | `latest/manifest.json` | 現在の稼働バージョンと関連情報 |

`run_id` は `YYYYMMDDTHHMMSSZ` 形式（UTC）を基本とし、GitHub Actions 等の CI では `workflow_run.id` などと組み合わせてもよい。手動リリース時は一意性が担保できるようにすること。

## 3. メタデータ・マニフェスト仕様

### 3.1 `model_meta.json`（run_id 配下）

```jsonc
{
  "model_name": "two_tower",
  "format": "two_tower.feature_mlp",
  "run_id": "20251031T093831Z",
  "trained_at": "2025-10-31T09:38:31Z",
  "git_commit": "<commit-hash>",
  "input_schema_version": 1,
  "embedding_dim": 256,
  "hidden_dim": 512,
  "user_feature_dim": 4099,
  "item_feature_dim": 21671,
  "tag_vocab_size": 4096,
  "performer_vocab_size": 1024,
  "preferred_tag_vocab_size": 4096,
  "use_price_feature": false,
  "data_snapshot": {
    "prep_run_id": "20251030T100538Z",
    "source_csv": "ml/data/raw/reviews/dmm_reviews_videoa_2025-10-04.csv"
  }
}
```

- 既存の `model_meta.json` に加え、`trained_at`、`git_commit`、`data_snapshot` など運用上必要な情報を追加する。
- `data_snapshot` は再現性を担保するために、前処理時の run_id や入力 CSV 名を保持する。

### 3.2 `metrics.json`（run_id 配下）

```jsonc
{
  "run_id": "20251031T093831Z",
  "evaluated_at": "2025-10-31T10:02:15Z",
  "metrics": {
    "recall@20": 0.695,
    "map@20": 0.421,
    "auc": 0.913
  },
  "validation_set": "ml/data/processed/two_tower/latest/interactions_val.parquet",
  "notes": "価格特徴量を OFF に切り替え。"
}
```

- 評価スクリプトの出力をそのまま格納する。必要に応じて `baseline_diff`（前回との差分）や `threshold_passed`（bool）を追加。

### 3.3 `latest/manifest.json`

```jsonc
{
  "model_name": "two_tower",
  "current": {
    "run_id": "20251031T093831Z",
    "published_at": "2025-11-01T02:40:10Z",
    "onnx_path": "two_tower/20251031T093831Z/two_tower_20251031T093831Z.onnx",
    "meta_path": "two_tower/20251031T093831Z/model_meta.json",
    "metrics_path": "two_tower/20251031T093831Z/metrics.json"
  },
  "previous": [
    {
      "run_id": "20251015T112233Z",
      "onnx_path": "two_tower/20251015T112233Z/two_tower_20251015T112233Z.onnx"
    }
  ],
  "release_notes": "Recall@20 が 0.68→0.70 に改善。price feature を OFF に固定。"
}
```

- `current` は最新リリースを指し示す。
- `previous` に過去の代表バージョン（直近 1～2 件）を記録しておくとロールバックが容易。
- `release_notes` は要約レベルの文章で OK。詳細はバージョン配下の `summary.md` に記載。

## 4. リリース手順（案）

1. **前処理・学習**  
   `scripts/prep_two_tower` → `scripts/train_two_tower` → `scripts/eval_two_tower` を実行し、`ml/artifacts/latest/` を更新。

2. **評価合格の確認**  
   - `metrics.json` の閾値（例: `recall@20 >= 0.65`, `auc >= 0.88`）を満たすかチェック。  
   - `gen_user_embeddings` と `upsert_two_tower` でローカル/ステージング環境の検証を行う。

3. **ファイルの配置準備**  
   - `ml/artifacts/latest/` から Storage へアップロードするファイルを収集。  
   - `summary.md` を作成（手動でも可）。`summary.md` には学習条件・気づき・既知の制約を箇条書きする。  
   - manifest に記載する予定値（`run_id`, `onnx_path`, `metrics_path` など）をまとめたドラフトを準備しておくが、この段階ではアップロードしない。

4. **成果物アップロード（Upload ジョブ）**  
   - 推奨: スクリプト化（例: `scripts/publish_two_tower/upload.sh`）。  
   - 必要な環境変数: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_PROJECT_REF`。  
   - 手順例（manifest は触らない）:
     ```bash
     RUN_ID=20251031T093831Z
     supabase storage cp ml/artifacts/latest/two_tower_latest.onnx \
       models/two_tower/${RUN_ID}/two_tower_${RUN_ID}.onnx
     supabase storage cp ml/artifacts/latest/two_tower_latest.pt \
       models/two_tower/${RUN_ID}/two_tower_${RUN_ID}.pt
     supabase storage cp ml/artifacts/latest/model_meta.json \
       models/two_tower/${RUN_ID}/model_meta.json
     supabase storage cp ml/artifacts/latest/metrics.json \
       models/two_tower/${RUN_ID}/metrics.json
     supabase storage cp tmp/summary.md \
       models/two_tower/${RUN_ID}/summary.md
     ```
   - アップロード後は検証が終わるまで manifest を据え置き、新バージョンが `latest` から参照されない状態を保つ。

5. **マニフェスト切り替え（Activate ジョブ）**  
   - 検証完了後、`latest/manifest.json` の `current` を新しい `run_id` に差し替える。  
   - Activate ジョブでは以下を行う:
     1. 旧 `current` を `previous` 配列に追加または更新。  
     2. `current.run_id` / `published_at` / 各パスを新バージョンへ置き換え。  
     3. 生成した manifest を `models/two_tower/latest/manifest.json`（もしくは環境別パス）にアップロード。  
   - `production/manifest.json`, `staging/manifest.json` のように環境別のポインタを分ける場合は、Activate ジョブを環境ごとに実行する。

6. **公開後確認・監視**  
   - manifest 更新後、Edge Function / API / バッチ処理が新バージョンを使用していることを確認する。  
   - Streamlit アプリ等で定性チェックを行い、異常があれば Activate ジョブを再実行し、`current` を `previous` の run_id へ戻す。

## 5. 残課題・要検討事項

- ロールバック手順: `latest/manifest.json` のみを指針にする場合、旧バージョンへ戻すには manifest の `current` を過去の `run_id` に差し替え、`published_at` を更新する。万一 Activate ジョブ実行前後で差異が出た場合に備え、`previous` リストに直近の安定バージョンを必ず保持し、pgvector 反映やキャッシュ更新との整合をどう確保するかを詰める。必要に応じて `production/` `staging/` 等の固定パスを分け、環境ごとのポインタを持たせる案も検討する。
- `latest` 参照の是非: 直接 `latest` ディレクトリ配下のファイルを利用する運用は、意図しないバージョン切り替えを招く可能性がある。基本は `latest/manifest.json` を読み、そこに記録された `run_id` を明示的に解決することで、参照整合性とロールバック容易性を両立させる。クライアント実装では manifest の署名検証や `published_at` の監視を追加し、切り替えトランザクションを観測できるようにしておく。
- CI/CD への組み込み: GitHub Actions で自動的にアップロードする際のアクセスキー管理（Service Role 秘密鍵の扱い、OIDC 連携など）。
- モデルの署名・検証: ダウンロード時にハッシュチェックを行うか。
- 保存期間・コスト最適化: 古いバージョンをどこまで残すか、Glacier 相当の層が必要か。
- モデルカテゴリの拡張: `two_tower` 以外のモデル（例: reranker, re-ranker）と共存させるための命名ルール。
- モデル切り替え通知: `manifest.json` 更新をトリガーに、フロントやエッジへ通知する仕組み（Webhook / Slack 等）が必要か。

---

上記はドラフトであり、運用設計の議論に向けた叩き台です。実装時に決定した内容は本ドキュメントを更新し、関連スクリプト（publish/upsert 等）と整合を取ってください。
