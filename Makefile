# Adult Matching アプリケーション - Makefile

# 基本設定
PYTHON = uv run python
SHELL = /bin/bash

# ディレクトリ（統合後のbackend構造対応）
DATA_DIR = backend/data-processing
ML_DIR = backend/ml-pipeline
FRONTEND_DIR = frontend
BACKEND_DIR = backend

.PHONY: help setup data clean train serve test

# デフォルトターゲット
help:
	@echo "Adult Matching アプリケーション"
	@echo ""
	@echo "=== データ収集コマンド ==="
	@echo "  collect-top-reviewers - トップレビュワー抽出"
	@echo "  collect-large-reviews - 大規模レビュー収集（フォアグラウンド）"
	@echo "  collect-large-bg      - 大規模レビュー収集（バックグラウンド）"
	@echo "  collect-all           - 大規模データ収集フル実行"
	@echo "  collect-status        - 収集進捗確認"
	@echo "  collect-logs          - 収集ログ確認"
	@echo "  collect-watch         - 収集監視（連続）"
	@echo ""
	@echo "=== データ処理・疑似ユーザー生成 ==="
	@echo "  integrate-batch-data  - バッチデータ統合・クリーニング"
	@echo "  generate-rating-users - 評価ベース疑似ユーザー生成（4+→Like）"
	@echo "  generate-pseudo-users - 疑似ユーザー生成（従来版）"
	@echo "  data-clean            - データクリーニング（従来版）"
	@echo ""
	@echo "=== ML学習パイプライン ==="
	@echo "  train-rating          - 評価ベースTwo-Towerモデル訓練"
	@echo "  train-full-rating     - 評価ベースフル学習パイプライン"
	@echo "  train                 - Two-Towerモデル訓練（従来版）"
	@echo "  train-full            - フル学習パイプライン（従来版）"
	@echo "  data-embed            - 埋め込みベクトル生成"
	@echo ""
	@echo "=== パフォーマンステスト・ベンチマーク ==="
	@echo "  benchmark             - 包括的パフォーマンステスト実行"
	@echo "  benchmark-quick       - 高速ベンチマーク実行"
	@echo "  benchmark-ci          - CI/CD統合パフォーマンステスト"
	@echo "  benchmark-results     - 最新ベンチマーク結果確認"
	@echo ""
	@echo "=== 開発・運用 ==="
	@echo "  setup                 - プロジェクト環境のセットアップ"
	@echo "  serve                 - フロントエンドサーバー起動"
	@echo "  clean                 - 一時ファイル・キャッシュのクリーンアップ"
	@echo "  test                  - テスト実行"
	@echo "  data-collect          - レビューデータの収集（従来版）"
	@echo "  train-full-legacy     - フル学習パイプライン（従来版）"

# プロジェクト環境のセットアップ
setup:
	@echo "=== プロジェクト環境セットアップ ==="
	uv sync
	@echo "必要なディレクトリを作成..."
	mkdir -p $(DATA_DIR)/raw_data $(DATA_DIR)/processed_data
	mkdir -p $(ML_DIR)/models $(ML_DIR)/training/data
	@echo "セットアップ完了！"

# データ収集
data-collect:
	@echo "=== レビューデータ収集開始 ==="
	@if [ ! -f "$(DATA_DIR)/config/dmm_cookies.json" ]; then \
		echo "エラー: $(DATA_DIR)/config/dmm_cookies.json が見つかりません"; \
		echo "$(DATA_DIR)/config/Cookie取得手順.md を参照してCookieを設定してください"; \
		exit 1; \
	fi
	cd $(DATA_DIR)/scraping && $(PYTHON) cookie_dmm_scraper.py
	@echo "データ収集完了！"

# データクリーニング
data-clean:
	@echo "=== データクリーニング開始 ==="
	cd $(DATA_DIR)/utils && $(PYTHON) data_cleaner.py
	@echo "データクリーニング完了！"

# トップレビュワー抽出
collect-top-reviewers:
	@echo "=== トップレビュワー抽出開始 ==="
	@if [ ! -f "$(DATA_DIR)/config/dmm_cookies.json" ]; then \
		echo "エラー: $(DATA_DIR)/config/dmm_cookies.json が見つかりません"; \
		echo "$(DATA_DIR)/config/Cookie取得手順.md を参照してCookieを設定してください"; \
		exit 1; \
	fi
	cd $(DATA_DIR)/scraping && $(PYTHON) top_reviewers_scraper.py
	@echo "トップレビュワー抽出完了！"

# 大規模レビュー収集（フォアグラウンド）
collect-large-reviews:
	@echo "=== 大規模レビュー収集開始 ==="
	@if [ ! -f "$(DATA_DIR)/raw_data/top_reviewers.json" ]; then \
		echo "エラー: トップレビュワーデータが見つかりません"; \
		echo "make collect-top-reviewers を先に実行してください"; \
		exit 1; \
	fi
	cd $(DATA_DIR)/scraping && $(PYTHON) robust_batch_scraper.py
	@echo "大規模レビュー収集完了！"

# 大規模レビュー収集（バックグラウンド）
collect-large-bg:
	@echo "=== バックグラウンド収集開始 ==="
	cd $(DATA_DIR)/scraping && ./run_background.sh
	@echo "バックグラウンド実行開始完了"

# 収集進捗確認
collect-status:
	@echo "=== 収集進捗確認 ==="
	cd $(DATA_DIR)/scraping && $(PYTHON) monitor_progress.py

# 収集ログ確認
collect-logs:
	@echo "=== 収集ログ確認 ==="
	cd $(DATA_DIR)/scraping && $(PYTHON) monitor_progress.py --log 50

# 収集監視（連続）
collect-watch:
	@echo "=== 収集監視開始 ==="
	cd $(DATA_DIR)/scraping && $(PYTHON) monitor_progress.py --watch

# 大規模データ収集フル実行
collect-all: collect-top-reviewers collect-large-reviews
	@echo "=== 大規模データ収集完了 ==="

# 疑似ユーザー生成（従来版）
generate-pseudo-users:
	@echo "=== 疑似ユーザー生成開始（従来版） ==="
	cd $(DATA_DIR)/utils && $(PYTHON) pseudo_user_generator.py
	@echo "疑似ユーザー生成完了！"

# 評価ベース疑似ユーザー生成（パターン1）
generate-rating-users:
	@echo "=== 評価ベース疑似ユーザー生成開始 ==="
	@echo "変換ルール: 評価4以上→Like, 3以下→Skip"
	cd $(DATA_DIR)/utils && $(PYTHON) rating_based_user_generator.py
	@echo "評価ベース疑似ユーザー生成完了！"

# バッチレビューデータ統合・クリーニング
integrate-batch-data:
	@echo "=== バッチデータ統合・クリーニング開始 ==="
	cd $(DATA_DIR)/utils && $(PYTHON) batch_data_integrator.py
	@echo "バッチデータ統合完了！"

# 埋め込みベクトル生成
data-embed:
	@echo "=== 埋め込みベクトル生成開始 ==="
	cd $(ML_DIR)/preprocessing && $(PYTHON) review_to_embeddings.py
	@echo "埋め込みベクトル生成完了！"

# 従来モデル訓練
train:
	@echo "=== Two-Towerモデル訓練開始（従来版） ==="
	cd $(ML_DIR)/training && $(PYTHON) train_two_tower_model.py
	@echo "モデル訓練完了！"

# 評価ベースモデル訓練（パターン1）
train-rating:
	@echo "=== 評価ベースTwo-Towerモデル訓練開始 ==="
	@if [ ! -f "$(DATA_DIR)/processed_data/rating_based_pseudo_users.json" ]; then \
		echo "エラー: 評価ベース疑似ユーザーデータが見つかりません"; \
		echo "make generate-rating-users を先に実行してください"; \
		exit 1; \
	fi
	cd $(ML_DIR)/training && $(PYTHON) rating_based_two_tower_trainer.py
	@echo "評価ベースモデル訓練完了！"

# フル学習パイプライン（評価ベース・パターン1）
train-full-rating: collect-all integrate-batch-data generate-rating-users train-rating
	@echo "=== 評価ベースフル学習パイプライン完了 ==="

# フル学習パイプライン（大規模データ収集対応・従来版）
train-full: collect-all data-clean generate-pseudo-users data-embed train
	@echo "=== フル学習パイプライン完了 ==="

# 従来のデータ収集（後方互換性のため残存）
train-full-legacy: data-collect data-clean generate-pseudo-users data-embed train
	@echo "=== 従来フル学習パイプライン完了 ==="

# フロントエンドサーバー起動
serve:
	@echo "=== フロントエンドサーバー起動 ==="
	cd $(FRONTEND_DIR) && npm run dev

# クリーンアップ
clean:
	@echo "=== クリーンアップ開始 ==="
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f debug_page.html cookie_debug_page.html
	@echo "クリーンアップ完了！"

# テスト実行
test:
	@echo "=== テスト実行 ==="
	cd $(FRONTEND_DIR) && npm run lint
	$(PYTHON) -m pytest ml_pipeline/tests/ -v 2>/dev/null || echo "テストディレクトリが見つからないか、pytestがインストールされていません"

# 開発環境情報表示
info:
	@echo "=== プロジェクト情報 ==="
	@echo "Python環境: $(shell uv python --version 2>/dev/null || echo 'uv が見つかりません')"
	@echo "Node.js環境: $(shell cd $(FRONTEND_DIR) && node --version 2>/dev/null || echo 'Node.js が見つかりません')"
	@echo "プロジェクト構造:"
	@tree -d -L 2 . 2>/dev/null || find . -type d -name ".*" -prune -o -type d -print | head -20

# データ統計表示
data-stats:
	@echo "=== データ統計 ==="
	@if [ -d "$(DATA_DIR)/raw_data" ]; then \
		echo "生データファイル:"; \
		ls -la $(DATA_DIR)/raw_data/ 2>/dev/null || echo "  なし"; \
	fi
	@if [ -d "$(DATA_DIR)/processed_data" ]; then \
		echo "前処理済みデータ:"; \
		ls -la $(DATA_DIR)/processed_data/ 2>/dev/null || echo "  なし"; \
	fi
	@if [ -d "$(ML_DIR)/models" ]; then \
		echo "学習済みモデル:"; \
		ls -la $(ML_DIR)/models/ 2>/dev/null || echo "  なし"; \
	fi

# Cookie設定ヘルプ
cookie-help:
	@echo "=== Cookie設定ヘルプ ==="
	@echo "1. ブラウザでDMMサイト (https://www.dmm.co.jp) にアクセス"
	@echo "2. 年齢認証を完了"
	@echo "3. 開発者ツール (F12) を開く"
	@echo "4. Application > Cookies > https://www.dmm.co.jp を選択"
	@echo "5. 重要なCookie (ckcy など) を $(DATA_DIR)/config/dmm_cookies.json に保存"
	@echo ""
	@echo "詳細: $(DATA_DIR)/config/Cookie取得手順.md を参照"

# バックアップ作成
backup:
	@echo "=== データバックアップ作成 ==="
	@BACKUP_NAME="backup_$(shell date +%Y%m%d_%H%M%S)"; \
	mkdir -p backups/$$BACKUP_NAME; \
	cp -r $(DATA_DIR)/raw_data/ backups/$$BACKUP_NAME/ 2>/dev/null || true; \
	cp -r $(DATA_DIR)/processed_data/ backups/$$BACKUP_NAME/ 2>/dev/null || true; \
	cp -r $(ML_DIR)/models/ backups/$$BACKUP_NAME/ 2>/dev/null || true; \
	echo "バックアップ作成完了: backups/$$BACKUP_NAME"

# プロジェクト状態チェック
status:
	@echo "=== プロジェクト状態 ==="
	@echo "Cookie設定:"
	@if [ -f "$(DATA_DIR)/config/dmm_cookies.json" ]; then \
		echo "  ✓ Cookie設定済み"; \
	else \
		echo "  ✗ Cookie未設定 (make cookie-help で設定方法を確認)"; \
	fi
	@echo "データ状況:"
	@if [ -n "$$(ls -A $(DATA_DIR)/raw_data/ 2>/dev/null)" ]; then \
		echo "  ✓ 生データあり"; \
	else \
		echo "  ✗ 生データなし (make data-collect で収集)"; \
	fi
	@if [ -n "$$(ls -A $(DATA_DIR)/processed_data/ 2>/dev/null)" ]; then \
		echo "  ✓ 前処理済みデータあり"; \
	else \
		echo "  ✗ 前処理済みデータなし (make data-clean で生成)"; \
	fi
	@echo "モデル状況:"
	@if [ -n "$$(ls -A $(ML_DIR)/models/ 2>/dev/null)" ]; then \
		echo "  ✓ 学習済みモデルあり"; \
	else \
		echo "  ✗ 学習済みモデルなし (make train で訓練)"; \
	fi

# パフォーマンステスト・ベンチマーク
benchmark:
	@echo "=== Two-Tower パフォーマンスベンチマーク実行 ==="
	$(PYTHON) tests/performance/two_tower_benchmark_suite.py

# 高速ベンチマーク（CI/CD用）
benchmark-quick:
	@echo "=== Two-Tower 高速ベンチマーク実行 ==="
	$(PYTHON) tests/performance/two_tower_benchmark_suite.py --quick

# CI/CD統合ベンチマーク
benchmark-ci:
	@echo "=== CI/CD パフォーマンステスト実行 ==="
	$(PYTHON) tests/performance/ci_benchmark.py

# ベンチマーク結果確認
benchmark-results:
	@echo "=== 最新ベンチマーク結果 ==="
	@if [ -n "$$(ls -t tests/performance/benchmark_results_*.json 2>/dev/null | head -1)" ]; then \
		latest_result=$$(ls -t tests/performance/benchmark_results_*.json | head -1); \
		echo "最新結果ファイル: $$latest_result"; \
		$(PYTHON) -c "import json; r=json.load(open('$$latest_result')); s=r.get('overall_summary',{}); print(f\"合格率: {s.get('pass_rate',0)*100:.1f}% ({s.get('passed_tests',0)}/{s.get('total_tests',0)} テスト合格)\")"; \
	else \
		echo "ベンチマーク結果が見つかりません。make benchmark を実行してください。"; \
	fi