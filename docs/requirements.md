# 要件定義（ドラフト）

本プロジェクトで開発・運用する機能の要件を記録します。新規機能や仕様変更時は、ここをまず更新してから実装に着手してください。

- 環境変数の一覧／設定方針は `docs/env/variables.md` に集約しています。GitHub Actions ごとの必要 Secrets や入力パラメータは `docs/env/github_actions.md` を参照してください。
- GitHub Actions の命名規約および現在の命名一覧は `docs/workflows/naming_conventions.md` で管理します。ワークフローを追加・改名した場合は同ドキュメントを更新してください。

## 目的/背景
- Two-Tower ベースの推薦モデルでユーザー嗜好に応じた動画レコメンドを提供し、初期体験の価値を高める。
- Supabase 上に蓄積した LIKE / レビュー履歴を活用し、ランキング品質を継続的に改善する。

## ユースケース
- エンドユーザーがアプリで LIKE / DISLIKE を行い、その履歴を元にパーソナライズされた動画一覧を取得する。
- 管理者が学習済みモデル（ONNX）とマッピング JSON をロードして推論品質を可視検証する（`frontend/src/app/admin/page.tsx`）。
- バッチ処理（例: 週次）でモデル・埋め込みを再生成し、Supabase Storage / pgvector テーブルへ反映する。
- エンドユーザーのスワイプ UI は `frontend/src/app/swipe/page.tsx`（ルート `/swipe`）で提供し、トップページ `/` は `/swipe` へリダイレクトする。

## 機能要件
- [ ] レビュー/LIKE データから `interactions_train/val.parquet` を生成する Docker スクリプトを整備する（現状: `scripts/prep_two_tower/run.sh` でベース実装あり。動画 ID 取得やデータ品質検証を自動化する TODO）。
- [ ] `prep_two_tower` は `--db-url` で Postgres/Supabase を直接参照して動画マスタを取得する仕様とし、CSV ダンプを介さずローカル DB を準備する前提とする。詳細手順は `docs/ml/data_generation_spec.md` に記録する。
- [ ] Two-Tower 学習コンテナを実行し、ONNX / state_dict / 埋め込み Parquet / メタ情報を出力する。ONNX は特徴量→ユーザー/アイテム埋め込みを生成できるよう設計する（評価・ログ整備は要追加）。
- [ ] 学習済み埋め込みを `public.video_embeddings` / `public.user_embeddings`（halfvec(128)）へアップサートするユーティリティを実装する。
- [ ] `user_video_decisions` の最新アクティビティをもとに、直近のユーザーのみを対象にした `user_embeddings` の増分更新ジョブ（1時間毎）を提供する。`scripts/sync_user_embeddings` で Docker 実行し、`[ML] Ops - Update User Embeddings` ワークフローから自動起動できること。
- [ ] モデル成果物（`two_tower_latest.onnx` + メタデータ）を Supabase Storage `models/` バケットへアップロードし、バージョニング/`latest` 更新を管理する。日次のアイテム更新・適時のユーザー更新で ONNX を利用するパイプラインを整備する。
- [ ] Supabase Storage へのモデル配置手順・命名規則・付随メタデータの保持方式を仕様化し、リリース時に参照可能なドキュメントを整備する。
- [ ] Edge Function `supabase/functions/ai-recommend` では、(1) ユーザー嗜好ベース、(2) 全体トレンド、(3) ユーザー入力（今の気分・キーワード）を反映したセットをそれぞれ1セクション以上返却し、モード選択なしのシンプルな UI から呼び出せるようにする。
- [ ] いいね動画ドロワーのフィルター UI（`frontend/src/components/LikedVideosDrawer.tsx`）では、「検索 / 絞り込み」セクション見出しの直上に検索結果件数をまとめて表示し、フィルター変更時の件数変化が即座に把握できるようにする。
- [ ] 嗜好分析ページ（`frontend/src/app/analysis-results/page.tsx`）では、シェアボタンから結果カードをプレビュー表示し、Twitter(X) 等で共有できる画像 1 枚に利用者の「好きなタグ / 出演者ランキング」や推しポイントを中心に訴求する。
- [ ] PyTorch と ONNX の出力ベクトル差分・ランキング指標を自動で計測する評価スクリプト（Recall@K, AUC など）を整備する。
- [ ] `maker`（ブランド）ごとの推薦分布を自動集計し、偏りが一定以上の時にリサンプリング・reranking・重みづけで補正する対策パイプライン（ドキュメント/CSVログ含む）を整備する。初期データは `2025-11-10T09-02_export.csv` を参照して、`recommendations`/`unique_items`/`user_coverage` などを継続的に記録する。

## 非機能要件
- [ ] すべての前処理・学習は専用 Docker コンテナ経由で再現可能にする（ホスト Python 依存を排除）。
- [ ] 埋め込み次元は 128（halfvec）とし、pgvector インデックス (`halfvec_cosine_ops`) を維持する。
- [ ] pgvector 拡張は halfvec 型を提供する 0.5 以降のバージョンを前提とする。
- [ ] 学習ジョブは再実行時に同一成果物パスへ上書きしても安全（idempotent）であること。
- [ ] Storage / DB 転送時に機密キーを `.env` から安全に読み取る。
- [ ] `ml/data` のディレクトリ構成を `raw/`（生データ）・`processed/two_tower/latest/`（最新出力）・`processed/two_tower/runs/<run-id>/`（スナップショット）の形で整理し、前処理スクリプトがこの構造を前提に動作する。
- [ ] ML 関連ドキュメントは `docs/ml/` 配下に配置し、再学習手順や仕様は同ディレクトリで管理する。

## インターフェース/依存
- Supabase Postgres (`public.video_embeddings`, `public.user_embeddings`, `public.video_popularity_daily`) と pgvector 拡張。
- Supabase Storage（`models/` バケット）および Edge Functions（`ai-recommend`）。
- Python 3.11 + PyTorch 2.2 + ONNX Runtime（Docker イメージ `python:3.11-bookworm` ベース）。
- 入力データ: `ml/data/raw/reviews/dmm_reviews_videoa_*.csv`, `ml/data/processed/two_tower/latest/interactions_*.parquet` など。

## 受け入れ基準
- [ ] `bash scripts/prep_two_tower/run.sh ...` で学習データが生成でき、サマリーログに件数・欠損が記録される。
- [ ] `bash scripts/train_two_tower/run.sh ...` 実行で ONNX / parquet / メタ JSON が `ml/artifacts/` に出力され、ONNX と PyTorch のベクトル一致テストが合格する。
- [ ] Embeddings upsert と Storage へのアップロードが手順化され、再学習後・日次アイテム更新・適時ユーザー更新で新モデルを利用できること。
- [ ] `ai-recommend` Edge Function がユーザー埋め込み入力時に Two-Tower 類似検索結果を返却する。

## 運用前提（Supabase 自前スタック）
- DB 初期化: `docker/db/init/*.sql` を idempotent に適用する。
- 互換性: Realtime（Ecto）は `public.schema_migrations(version, inserted_at)` を期待する。
  - 他ツールにより `public.schema_migrations(version)` のみが存在する場合、`003-fix-ecto-schema-migrations.sql` で `inserted_at` を自動追加する。
  - 既存データは保持される（`inserted_at` は `now()` で補完）。
- マイグレーション方針: 既存のマイグレーションファイルは変更しない。修正が必要な場合は新しいマイグレーションを追加して対応する。
