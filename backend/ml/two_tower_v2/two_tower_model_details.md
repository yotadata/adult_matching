# Two-Tower v2 モデル詳細解説

## 1. モデル全体像
Two-Tower v2 は、ユーザー塔とアイテム塔の2本のニューラルネットワークで埋め込みを生成し、両者の内積で嗜好スコアを算出する協調フィルタリングモデルです。Supabase 上の `profiles` / `videos` / `user_video_decisions` スキーマと互換性を保ちながら学習し、ユーザー塔は ONNX 形式でエッジ推論（Supabase Edge Functions）へ提供します。アイテム塔で得た動画埋め込みは Supabase Postgres の `video_embeddings` テーブルに格納し、HNSW ベクター検索で利用します。

## 2. ネットワークアーキテクチャ
- 実装ファイル: `backend/ml/two_tower_v2/src/model.py`
- 共通ディメンション:
  - ユーザー特徴次元 `user_input_dim`
  - アイテム特徴次元 `item_input_dim`
  - 埋め込み次元 `embedding_dim`（既定 256）
- 各塔の構造（ユーザー塔・アイテム塔ともに同一）:
  1. `Linear(input_dim → 512)` + `ReLU`
  2. `Dropout(p=0.2)`
  3. `Linear(512 → 256)` + `ReLU`
  4. `Dropout(p=0.1)`
  5. `Linear(256 → embedding_dim)`
  6. 出力を `F.normalize(..., p=2)` で L2 正規化
- 推論スコア: `similarity = (user_embedding * item_embedding).sum(dim=-1)`
  - 埋め込みが L2 正規化されているため、内積はコサイン類似度と等価
- 推定確率: `torch.sigmoid(similarity)` を返却（学習時は BCEWithLogitsLoss）
- エッジ推論ではユーザー塔のみ ONNX 化し、アイテム埋め込みは事前計算済みベクターを利用

## 3. 特徴量パイプライン

### 3.1 利用カラムの詳細
- `public.profiles`
  - `user_id`: ユニークキー。文字列に変換して全体の主キーとして使用。
  - `created_at`: アカウント作成日時。`account_age_days` を算出するための唯一の直接入力。`display_name` は抽出するが特徴量には使用していない。
- `public.user_video_decisions`
  - `user_id`: ユーザー識別子。`profiles` の `user_id` と結合し、学習対象ユーザーを制限。
  - `video_id`: 動画識別子。`videos` と結合し、タグや価格情報にアクセス。
  - `decision_type`: `like` / `nope` の区別。正例/既知負例の判定および like 比率算出に使用。
  - `created_at`: 行動発生日時。`recent_like_days`、`average_like_hour`、`decision_ts` など派生特徴の計算に使用。
- `public.videos`
  - `id`: 動画識別子。`user_video_decisions.video_id` と結合し、タグ・出演者・価格情報を取得。
  - `price`: アイテム数値特徴（価格）およびユーザー側の like 価格統計（平均・中央値）に使用。
  - `duration_seconds`: アイテム数値特徴としてそのまま使用。
  - `product_released_at`: リリース時刻。最新動画との日数差 (`recency_days`) を計算。
  - `distribution_started_at`, `created_at`, `published_at` などは抽出するが現行の特徴量では未使用。
- `public.video_tags`
  - `video_id`: `videos.id` と結合。
  - `tag_name`: タグ語彙ベクトルの Bag-of-Words の元データ。
- `public.video_performers`
  - `video_id`: `videos.id` と結合。
  - `performer_name`: 出演者語彙ベクトルの Bag-of-Words の元データ。

**派生ラベルと特徴量**
- 学習ラベル `label`: `decision_type='like'` を 1、それ以外（`nope`、追加サンプリング負例）を 0 として付与。
- 追加負例: `negative_sampling_ratio` に基づき、同一ユーザーが未評価の動画 ID を `videos.id` からランダム選択し `decision_type='sampled_negative'` として追加。
- ユーザー数値特徴:
  - `account_age_days`: `profiles.created_at` から決定。
  - `mean_price` / `median_price`: like した動画の `price` から算出。
  - `like_count` / `nope_count` / `decision_count`: `user_video_decisions` から集計。
  - `like_ratio`: like_count / decision_count（ゼロ割回避で 0.5 初期値）。
  - `recent_like_days`: 最新 like の `created_at` と基準日時との差。
- アイテム数値特徴:
  - `price`: `videos.price`
  - `recency_days`: `videos.product_released_at` とデータセット内最新リリース日の差。
  - `duration_seconds`: `videos.duration_seconds`
- 語彙ベクトル:
  - タグ語彙は `video_tags.tag_name` の頻度上位を `max_tag_vocab` で制限。
  - 出演者語彙は `video_performers.performer_name` を `max_actress_vocab` で制限。

```yaml
# config/default.yaml から抜粋
artifacts:
  max_tag_vocab: 512
  min_tag_freq: 3
  max_actress_vocab: 256
  min_actress_freq: 2
```

RPC 利用時は、同じカラムを Supabase 側で集約した JSON として取得します（`sql/local_schema.sql` の `get_user_embedding_features` / `get_item_embedding_features` を参照）。それ以外のカラムは現行モデルでは学習に用いていません。

### 3.2 学習入力の流れ
1. **サンプル組み立て (`src/data_utils.py`)**
   - `public.user_video_decisions` から `decision_type` と `created_at` を取得。
   - `label=1` は `decision_type='like'`、`label=0` は `decision_type in ('nope', 'sampled_negative')`。
   - `negative_sampling_ratio` に基づき、未評価動画 ID を追加で `sampled_negative` として付与。
   - 結果 DataFrame 列: `user_id`, `video_id`, `label`, `decision_type`, `decision_ts`。
2. **ユーザー特徴ベクトル (`features.UserFeatureStore`)**
   - 入力カラム: `profiles.created_at`, `user_video_decisions` 各列, `videos.price`, `videos.product_released_at`, `video_tags.tag_name`, `video_performers.performer_name`。
   - 派生数値: `account_age_days`, `mean_price`, `median_price`, `like_ratio`, `like_count`, `nope_count`, `decision_count`, `recent_like_days`。
   - 時間帯: like の `created_at` から `average_like_hour` を算出。
   - タグ／出演者: Vocabulary で L2 正規化 BOW に変換。
   - 正規化: 数値8項目を StandardScaler し、`hour_of_day` を 23 で割って単一要素として追加。
   - 最終ベクトル順序: `[normalized_numeric(8), hour_norm(1), tag_vector(n), actress_vector(m)]`。
3. **アイテム特徴ベクトル (`features.ItemFeatureStore`)**
   - 入力カラム: `videos.price`, `videos.duration_seconds`, `videos.product_released_at`, `video_tags.tag_name`, `video_performers.performer_name`。
   - 派生数値: `recency_days` = 最新 `product_released_at` との差分日数。
   - BOW: タグ／出演者を Vocabulary で変換。
   - 正規化: 数値3項目を StandardScaler で変換。
   - 最終ベクトル順序: `[normalized_numeric(3), tag_vector(n), actress_vector(m)]`。
4. **PyTorch への入力 (`InteractionDataset`)**
   - 上記ユーザー／アイテムベクトルを Tensor 化して `TwoTowerModel` に投入。
   - ラベル `label` を同バッチで保持し、`BCEWithLogitsLoss` に渡す。

### 3.3 埋め込み生成時の入力
1. `src/generate_embeddings.py`
   - `decisions_df` に依存せず、`videos` テーブルの全 `video_id` を対象。
   - `ItemFeatureStore.build_features(video_id)` で数値＋BOW を生成（RPC モード時は `get_item_embedding_features` の JSON が入力）。
   - `assemble_item_feature_vector` で学習時と同じ正規化・連結を行い、`TwoTowerModel.encode_item` に渡して 256 次元埋め込みを得る。
2. 出力フォーマット
   - DataFrame 列: `video_id`, `embedding`（Python list / Arrow array）。
   - `video_embeddings.parquet` に書き出し、必要に応じて `upload_embeddings.py` で Supabase `video_embeddings` に upsert。




- 実装ファイル: `backend/ml/two_tower_v2/src/features.py`
- 調達データ:
  - ユーザー: `profiles`、`user_video_decisions`、任意で RPC `get_user_embedding_features`
  - アイテム: `videos`、`video_tags`、`video_performers`、任意で RPC `get_item_embedding_features`
- 語彙構築:
  - タグ・出演者の出現頻度から上限サイズ／最小頻度フィルタ（`config/default.yaml`）を適用
  - Vocabulary クラスで L2 正規化済み Bag-of-Words ベクターを生成
- 数値特徴:
  - ユーザー側（8項目）: アカウント年齢、価格統計、like/nope 数、like 比率、最近 like までの日数など
  - アイテム側（3項目）: 価格、リリース Recency（日数）、動画秒数
  - `sklearn.StandardScaler` で平均0・分散1に正規化。
- 時間帯特徴:
  - ユーザー like 発生時刻の平均時刻（0〜23）を 23 で除算して [0,1] にスケーリング
- RPC モード:
  - `training.use_rpc_features=true` の場合は Postgres 関数 `get_user_embedding_features` / `get_item_embedding_features` を介して JSON を取得し、ロジックをサーバサイドに集約
  - ローカル DB の `sql/local_schema.sql` に同等の関数が提供されており、開発環境でも再現可能
- 特徴量ベクトル組み立て:
  - ユーザー: `[numeric (8)] + [hour_of_day (1)] + [tag_vector] + [actress_vector]`
  - アイテム: `[numeric (3)] + [tag_vector] + [actress_vector]`
  - ベクトル長は `FeaturePipeline.user_feature_dim` / `item_feature_dim` として保持され、モデル初期化や ONNX 出力に利用

## 4. 学習用データセット
- 実装ファイル: `backend/ml/two_tower_v2/src/data_utils.py`
- 正例: `user_video_decisions` の `label=1`（like）
- 負例: `negative_sampling_ratio`（既定 3）に基づきユーザー毎にランダム動画をサンプリング
- 学習/検証分割: `train_val_split`（既定 0.1）でホールドアウト
- `InteractionDataset`（`src/train.py`）で PyTorch Tensor へ変換し、`DataLoader` でバッチ化

## 5. 学習プロセス
- エントリーポイント: `src/train.py`
- 主要ハイパーパラメータ（`config/default.yaml`）:
  - `epochs`: 25
  - `batch_size`: 256
  - `lr`: 0.001
  - `weight_decay`: 1e-5
  - `gradient_clip_norm`: 5.0（`torch.nn.utils.clip_grad_norm_`）
  - `early_stopping_patience`: 5
- オプティマイザ: `torch.optim.AdamW`
- 損失関数: `nn.BCEWithLogitsLoss`
- 評価指標: 検証セットで loss と accuracy（0.5 閾値）を計算
- 早期終了: 検証 loss 改善が `1e-4` 未満で停止、最良モデルを `checkpoints/latest.pt` に保存
- チェックポイント保存内容:
  - `model_state`: PyTorch state_dict
  - `pipeline`: 語彙・正規化器・次元情報
  - `config`: 使用設定

## 6. エクスポートと推論利用
- エクスポート: `src/export.py`
  - ユーザー塔を ONNX (`artifacts/user_tower.onnx`) として出力（可変バッチ対応）
  - `feature_schema.json`、`normalizer.json`、`vocab_tag.json`、`vocab_actress.json`、`model_meta.json` を生成
  - `embedding_dim` や語彙サイズなどを `model_meta.json` で記録
- 埋め込み生成: `src/generate_embeddings.py`
  - `video_embeddings.parquet` に `video_id` と 256 次元埋め込みを保存
  - Supabase REST (`upload_embeddings.py`) を通じて `video_embeddings` テーブルに upsert
- マニフェスト: `scripts/build_manifest.py`
  - `manifest.json` に SHA256 とメタ情報を記録し、Supabase Storage での整合性確認に利用

## 7. 推論フローとの関係
1. Edge Function は Supabase RPC から特徴量 JSON を取得（ユーザー／アイテム）
2. `feature_schema.json` と `normalizer.json` を用いて JavaScript 側でベクトルを再構築
3. ONNX ランタイム（`onnxruntime-web`）でユーザー塔を推論しユーザー埋め込みを生成
4. アイテム埋め込みは事前計算済みベクトルを Supabase から取得し、コサイン類似度で上位候補を検索
5. 検索結果を API レスポンスとして返却

## 8. 可搬性と再現性
- 乱数シード (`seed`) によりデータ分割と PyTorch 初期化の再現性を担保
- 語彙構築や正規化統計を `pipeline` メタデータとして checkpoint に保存し、再エクスポート時に整合
- `use_rpc_features` を true にすると、学習時と本番推論時が同じ SQL ロジックを共有し、データリークや分岐ミスを防止

## 9. チューニングのヒント
- `embedding_dim` を調整することでベクトルの表現力と容量のトレードオフを最適化
- `max_tag_vocab` / `max_actress_vocab` を増やすと表現力が向上する一方、推論時の入力次元が増加するため ONNX 推論負荷を考慮
- 負例サンプリング比率や `train_val_split` を変更して精度と学習安定性を評価
- Dropout 率や中間層ユニット数を変更する場合は、ユーザー／アイテム塔を同じ構造に保つことが前提

## 10. 今後の拡張案
- ユーザー行動の時間的重み付け（最新行動へのブースト）
- マルチタスク（例: クリック・購入など別タスクの共学習）
- アイテム塔埋め込みを補助する外部特徴（自然言語、画像エンコーディングなど）の導入
- Supabase Storage への成果物公開をトリガーにした自動ロールバック仕組み

---
このドキュメントは `backend/ml/two_tower_v2` ディレクトリのコードベースに基づき、Two-Tower v2 モデルの学習・推論パイプラインを正確に把握するためのリファレンスとして作成しています。運用時は `README.md` および Supabase 連携仕様書と合わせて参照してください。
