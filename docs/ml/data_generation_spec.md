# データ生成仕様

Two-Tower 推薦モデル向けの学習データを生成するための標準手順と、関連するスクリプトの入出力仕様をまとめる。
本ドキュメントの前提はすべて Docker 経由でスクリプトを実行すること（`scripts/<name>/run.sh`）である。

## ディレクトリと命名規約
- 原データ (`raw`): `ml/data/raw/`
  - DMM レビュー CSV: `ml/data/raw/reviews/dmm_reviews_videoa_YYYY-MM-DD.csv`
  - リモート DB ダンプ: `ml/data/raw/db_dumps/<run-id>/prep_subset.sql`
- 加工データ (`processed`): `ml/data/processed/two_tower/`
  - 最新成果物: `latest/`（`interactions_train.parquet`, `interactions_val.parquet`, `item_features.parquet`, `summary.json`）
  - スナップショット: `runs/<run-id>/`（Train/Val/Items/summary + 必要に応じて inputs/）
- 学習成果物: `ml/artifacts/`（モデルや埋め込みを別ドキュメントで定義）

## 生データ取得

### DMM レビュー収集（スクレイピング）
- スクリプト: `scripts/scrape_dmm_reviews/scrape_dmm_reviews.py`
- 実行方法: `bash scripts/scrape_dmm_reviews/run.sh --output ml/data/raw/reviews/dmm_reviews_videoa_2024-06-01.csv`
- 主な引数:
  - `--ranking-period <period>`: ランキング期間（`year`/`month`/`3month`/`halfyear`/`all`）を指定。複数回指定すると結合して重複を除外。
  - `--max-reviewers`（既定100）: ランキング上位レビュアー数
  - `--delay`（既定1.2秒）: リクエスト間隔（レートリミット対策）
  - `--only-reviewers <ID...>`: 指定レビュアーのみ巡回（ランキング取得をスキップ）
  - `--dump-html-dir`, `--dump-on-missing-star` など: HTML ダンプによるデバッグ
- 出力列: `product_url`, `reviewer_id`, `stars`
- 特徴:
  - ランキングページからレビュアー一覧を抽出し、ページ構造変化に備えて複数セレクタと正規表現で冗長に解析。
  - 各レビュアーのレビュー一覧を巡回し、`cid` を含む AV のみ抽出。
  - 星評価欠損時は詳細ページを最大 `--max-detail-fetch-per-reviewer` 回まで再取得し補完を試みる。
  - 解析失敗時は一部 HTML をディスクへダンプし解析しやすくする。

### FANZA API からの動画取り込み
- スクリプト: `scripts/ingest_fanza/index.ts`
- 実行方法: `bash scripts/ingest_fanza/run.sh`
- 必須環境変数（`docker/env/prd.env` を既定利用。別ファイルを使う場合は `INGEST_FANZA_ENV_FILE=<path>` を指定すると、Docker の `--env-file` と Node 側の dotenv 両方でそのファイルを参照する）:
  - `NEXT_PUBLIC_SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`（なければ `NEXT_PUBLIC_SUPABASE_ANON_KEY` で代用するが、RLS が有効なテーブルは失敗する）
  - `FANZA_API_ID`, `FANZA_API_AFFILIATE_ID`, `FANZA_LINK_AFFILIATE_ID`
- 任意指定: `GTE_RELEASE_DATE`, `LTE_RELEASE_DATE`（`YYYY-MM-DD` または `YYYYMMDD` 形式）を環境変数で渡すと対象期間を変更できる（未指定時は「今日から1年前」が基準）。API へは ISO 8601 形式の `gte_date`（`T00:00:00`）/`lte_date`（`T23:59:59`）として伝播する。`bash` 実行時に付けた値はコンテナ側へ自動で引き継がれる。
- デバッグ用途で詳細ログを見たい場合は `INGEST_FANZA_DEBUG=1 bash scripts/ingest_fanza/run.sh` のように実行。
- 仕様:
  - FANZA Affiliate API `ItemList` を 1 年分（`gte_release_date`, `lte_release_date`）でページング取得。
  - `external_id`（FANZA CID）をキーに `videos` を upsert し、既存レコードも最新情報で更新。
  - ジャンルは `tags` / `video_tags`、出演者は `performers` / `video_performers` に upsert。実行中は名前/IDをキャッシュしておき、重複する API 呼び出しを避ける。
  - 動画1件につき `videos`・`video_tags`・`video_performers` への upsert をそれぞれ1リクエストにまとめ、Supabase REST の呼び出し回数を削減。API 取得は発売日の降順（新しい順）で進め、途中で中断してもリスタートしやすい構成にする。
  - 価格・収録時間は文字列を数値化して格納。リンクはアフィリエイト ID 付き URL に書き換え。
  - すべて Supabase REST API（`supabase-js`）経由で実行するため、Docker コンテナから直接 Supabase に書き込む。

### 動画マスタのローカル複製
- スクリプト: `scripts/prep_two_tower/run_with_remote_db.sh`
- 目的: リモート Postgres から `videos`, `video_tags`, `tags` 等をダンプし、前処理用のローカル Postgres を一時的に起動して利用。
- 実行例:
  ```bash
  bash scripts/prep_two_tower/run_with_remote_db.sh \
    --remote-db-url "$REMOTE_DATABASE_URL" \
    --mode reviews \
    --input ml/data/raw/reviews/dmm_reviews_videoa_2024-06-01.csv \
    --run-id auto
  ```
- 処理フロー:
  1. `pg_dump` コンテナで指定テーブルを SQL ダンプ。
  2. `postgres:15-alpine` コンテナを起動し SQL を流し込み。
  3. 同一 Docker ネットワーク上で前処理用コンテナを実行（`--db-url` をローカルに向ける）。
  4. 実行後に Postgres コンテナを破棄。

## 学習データ整形（Two-Tower 前処理）

### スクリプト概要
- 本体: `scripts/prep_two_tower/prep_two_tower_dataset.py`
- 実行方法: `bash scripts/prep_two_tower/run.sh <引数>`（ローカル CSV + 既存動画 CSV）、または上記 run_with_remote_db 経由。
- 入力モード（`--mode`）:
  - `explicit`: `reviewer_id|user_id`, `product_url|item_id`, `label` or `decision`
  - `reviews`: `product_url`, `reviewer_id`, `stars`（星評価から暗黙正例を構築）
- 追加入力:
  - `--decisions-csv`: 明示フィードバック CSV を追加でマージ（同一キーは decisions 側で上書き）
  - `--min-stars`（既定 4）: reviews モードで正例とみなす星数
  - `--neg-per-pos`（既定 3）: 正例 1 件あたりの疑似負例数（同一ユーザー未観測アイテムから抽出）
  - `--val-ratio`（既定 0.2）: ユーザー単位の検証用比率（各ユーザーが train/val 両方に入るよう抽出）

### 特徴量設計（ユーザー／アイテム）

前処理では Two-Tower が特徴量テンソルを直接受け取る前提で、`item_features.parquet`（アイテム特徴量）と `user_features.parquet`（ユーザー特徴量）を出力する。

#### ユーザー特徴量（候補）

| データソース | 項目名 | 型 | 例 | 補足 |
| --- | --- | --- | --- | --- |
| `interactions_train/val.parquet` | `reviewer_id` | string | `user_3f8c9a` | 必須キー。既存学習はこの ID をそのまま埋め込み化。 |
| `public.user_video_decisions`（CSV エクスポート可） | `recent_positive_video_ids` | uuid[]（最大20件） | `["8b2f8a0c-...", "c73d1fe0-..."]` | 直近 LIKE の履歴を時系列降順で格納。パディング後に固定長テンソル化予定。 |
| `public.user_video_decisions` 集計 | `like_count_30d` | int32 | `12` | 直近 30 日間の LIKE 件数。コールドスタート判定・重み付けに利用。 |
| `public.user_video_decisions` 集計 | `positive_ratio_30d` | float32 | `0.67` | LIKE / (LIKE+DISLIKE) の比率。0 除算時は `null`。 |
| `public.profiles` | `signup_days` | int32 | `124` | `now() - created_at` を日数換算。アクティブ度の proxy。 |
| `public.profiles` + 集計 | `preferred_tag_ids` | uuid[] | `["0c8c...", "9f2b..."]` | LIKE した動画タグの上位 N 件。タグ軸のユーザー嗜好を表現。 |
> `user_features.parquet` を追加する際は、上記列を含む長形式（主キー: `reviewer_id`）で保存し、欠損は `null` のまま後段でマスク処理する。数値列は `float32` / `int32` ベースで書き出し、ONNX 入力テンソルのスキーマは `model_meta.json` の `input_schema_version` と同期させる。  
> レビューデータ（星評価 CSV）は初期モデル構築時の暫定データであり、恒常運用の特徴量には利用しない方針とする。ユーザー集計は `user_video_decisions` / `profiles` / `video_tags` を参照するため、`--db-url` を指定して Postgres から直接取得する。ローカル開発で DB 接続が難しい場合は、`interactions_train/val.parquet` を元に LIKE 件数やタグ頻度など最小限の統計量を合成し、`preferred_tag_ids` などの多値列を空配列で埋める暫定措置を取ってもよい（本番前には必ず DB 集計版に差し替える）。

#### アイテム特徴量（現行＋拡張）

| データソース | 項目名 | 型 | 例 | 補足 |
| --- | --- | --- | --- | --- |
| `public.videos` | `video_id` | uuid | `3f9a0d3e-...` | 主キー。`interactions` からの join 結果として必須。 |
| `public.videos` | `product_url` | string | `https://...cid=abcd1234` | Two-Tower 以外の連携向けに保持。 |
| `public.videos` | `source` | string | `fanza`, `mgs`, `fc2` | 取得元サービス。複数ソースのフォーマット差異を識別し、後段で正規化。 |
| `public.videos` | `external_id` | string | `dv-12345`, `FC2-PPV-332211` | 各ソース固有の ID。`source + external_id` でユニーク。 |
| `public.videos` | `title` | string | `サンプル作品 Vol.1` | テキスト埋め込み生成時の原文。 |
| `public.videos` | `maker` | string | `S1 NO.1 STYLE` | カテゴリ分類用。 |
| `public.videos` | `label` | string | `単体作品` | 補助属性。欠損は空文字。 |
| `public.videos` | `series` | string | `超S級` | 作品シリーズ名。 |
| `public.videos` | `price` | float32 | `1980.0` | 価格帯をバケット化して特徴にする想定。 |
| `public.video_tags` join | `tag_ids` | uuid[] | `["4d2c...", "a8be..."]` | 多値カテゴリ。タグ一覧を保持し後段で multi-hot 化。 |
| `public.video_performers` join | `performer_ids` | uuid[] | `["1f20...", "9bb4..."]` | 出演者 ID リスト。出演者の嗜好学習に利用。 |
| `item_features.parquet` 派生 | `tag_multi_hot` | float32[]（vector） | `[0,1,0,1,...]` | タグ集合を固定長ベクトル化したもの。生成時はタグ辞書のバージョンを記録。 |
| `item_features.parquet` 派生 | `title_embedding` | float32[] | `[0.12, -0.03, ...]` | 外部テキストエンコーダの出力。モデル更新時に再計算し、ONNX 入力と揃える。 |

> `item_features.parquet` には必須列（`video_id`, `product_url`, `title`, `maker`, `label`, `series`, `tags`）を最低限含め、ベクトル列を追加する際は JSON 形式ではなく配列カラム（PyArrow List）で保存する。タグ辞書やテキストエンコーダのバージョン情報は `summary.json` と `model_meta.json` 双方に記録し、再学習時の互換性検証を高速化する。

マルチソース対応では、各スクレイパー／API の原データ列を以下のように正規化する。

| 取得元 | 元の ID 列 | 正規化後の `external_id` | タイトル列 | 価格列 | 備考 |
| --- | --- | --- | --- | --- | --- |
| FANZA (DMM) | `cid` | そのまま小文字化 | `title` | `price` | 既存構造。 |
| MGS | `product_code` | `mgs:<code>` | `product_name` | `price` | `mgs:` プレフィックスを付与して重複回避。 |
| FC2 | `content_id` | `fc2:<id>` | `title` | `price` | 一部は個人販売で価格欠損 → `null`。 |

> 新しいソースを追加する場合は、`source` 値と `external_id` の正規化規則を上表に追記し、`scripts/prep_two_tower` 内で共通スキーマ（`video_id`, `external_id`, `product_url`, ...）になるようマッピングを実装する。ロケールや通貨が異なる場合は `price` を円換算した上で格納し、換算レートと更新日時を `summary.json` に記録する。

ユーザーデータも LIKE/NOPE 以外の行動履歴を取り込む前提で拡張可能とする。

| データソース | 項目名 | 型 | 例 | 補足 |
| --- | --- | --- | --- | --- |
| `public.watch_histories`（想定） | `recent_watch_video_ids` | uuid[] | `["2bb1...", "7cd0..."]` | 直近視聴のアイテム。視聴のみで離脱した作品の分析に利用。 |
| `public.search_logs`（想定） | `recent_search_terms` | text[] | `["素人", "人妻"]` | 直近検索ワード。形態素解析後にトークン埋め込みを作成。 |
| `public.watch_histories` 集計 | `watch_duration_avg` | float32 | `842.5` | 視聴完走時間の平均秒数。時間帯や作品尺を特徴化。 |
| `public.watch_histories` 集計 | `last_watch_at` | timestamptz | `2024-06-03T12:34:56Z` | 直近行動の鮮度。ONNX 入力では相対時間（分/時間）に変換。 |

> 上記テーブルは将来的な追加を見越したもので、実装済みではない。スキーマ導入時は `prep_two_tower_dataset.py` の CLI に該当 CSV/テーブルを指定できるようパラメータを追加し、`summary.json` へ収集元と抽出日時を記録する。複数行動ログを統合する際は、ユーザー ID / タイムスタンプをキーに「最新 N 件」「期間集計」の 2 パターンを準備し、ONNX 入力で扱いやすい形（固定長ベクトル＋マスク）へ整形する。

### 動画マスタとの突合
- `--videos-csv` または `--db-url`（優先）で動画マスタを読み込む。未指定の場合はエラーで終了。
- `--join-on` で照合キーを制御（`auto`/`product_url`/`external_id`）。`external_id` を選ぶときは入力 `product_url` から `cid` を抽出し突合。
- 突合結果に video_id 欠損が発生した行は削除し、件数をサマリーに記録。
- 付随情報（`title`, `maker`, `label`, `tags` など）は `item_features.parquet` として保存。

### 出力とスナップショット
- 既定出力（`--skip-legacy-output` 未指定時）:
  - `ml/data/processed/two_tower/latest/interactions_train.parquet`
  - `ml/data/processed/two_tower/latest/interactions_val.parquet`
  - `ml/data/processed/two_tower/latest/item_features.parquet`（動画マスタが利用できた場合）
  - `ml/data/processed/two_tower/latest/summary.json`（統計と出力ファイル情報を JSON で記録）
- `--run-id` を指定すると `runs/<run-id>/` 以下にスナップショットを残す。
  - `--run-id auto` で `UTC YYYYMMDDThhmmssZ` 形式を自動採番。
  - `--snapshot-inputs` で参照した CSV を `runs/<run-id>/inputs/` にコピー。
  - `--skip-legacy-output` を併用するとスナップショットのみを書き出し、`latest/` は更新しない。

### サマリー項目
- `train_rows`, `val_rows`, `users_train`, `users_val`, `items`
- `merged_decisions`, `joined_videos`, `missing_video_id_before_drop`
- `interactions_before_join`, `dropped_no_video_id`, `items_matched`, `join_used`, `cid_missing_in_input`
- `paths`: 各成果物パス、スナップショットパス
- コンソール出力でも同じ JSON を `print` するため、ログからも整形可能。

### 実行例
```bash
bash scripts/prep_two_tower/run.sh \
  --mode reviews \
  --input ml/data/raw/reviews/dmm_reviews_videoa_2024-06-01.csv \
  --videos-csv ml/data/raw/videos_dump.csv \
  --min-stars 4 \
  --neg-per-pos 3 \
  --val-ratio 0.2 \
  --run-id 20240601_reviews
```

## 品質管理と運用メモ
- スクリプト実行ログと `summary.json` を Pull Request やナレッジ共有時の差分確認に利用する。
- `docs/requirements.md` の TODO にある通り、今後は動画 ID 取得やデータ品質検証の自動化、評価・アップサート手順を追加予定。
- 生データ／成果物の配置場所と命名規則は厳守し、手動編集は避ける。
- 外部サイト構造や API 仕様が変わる可能性があるため、スクレイピング・API エラー時は `--dump-html-dir` やログを活用して追跡する。
