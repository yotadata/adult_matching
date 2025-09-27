#!/bin/bash

# バックグラウンドでスクレイピングを実行するスクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$DATA_DIR/raw_data"

echo "=== バックグラウンドスクレイピング開始 ==="

# ログディレクトリ作成
mkdir -p "$LOG_DIR"

# 実行オプション
RESUME_FLAG=""
if [ "$1" = "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "中断から再開モードで実行します"
fi

# PIDファイル
PID_FILE="$LOG_DIR/scraper.pid"

# 既に実行中かチェック
if [ -f "$PID_FILE" ]; then
    EXISTING_PID=$(cat "$PID_FILE")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "エラー: スクレイパーは既に実行中です (PID: $EXISTING_PID)"
        echo "停止するには: kill $EXISTING_PID"
        exit 1
    else
        echo "古いPIDファイルを削除します"
        rm -f "$PID_FILE"
    fi
fi

# 実行環境チェック
echo "実行環境チェック..."
echo "作業ディレクトリ: $SCRIPT_DIR"
echo "データディレクトリ: $LOG_DIR"

# Cookieファイルチェック
COOKIE_FILE="$DATA_DIR/config/dmm_cookies.json"
if [ ! -f "$COOKIE_FILE" ]; then
    echo "エラー: Cookieファイルが見つかりません: $COOKIE_FILE"
    echo "Cookieを設定してから再実行してください"
    exit 1
fi

# レビュワーファイルチェック
REVIEWER_FILE="$LOG_DIR/top_reviewers.json"
if [ ! -f "$REVIEWER_FILE" ]; then
    echo "エラー: レビュワーファイルが見つかりません: $REVIEWER_FILE"
    echo "make collect-top-reviewers を先に実行してください"
    exit 1
fi

echo "チェック完了"

# バックグラウンド実行
echo "バックグラウンド実行を開始します..."
echo "ログファイル: $LOG_DIR/scraping.log"
echo "PIDファイル: $PID_FILE"

cd "$SCRIPT_DIR"

# nohupでバックグラウンド実行
nohup uv run python robust_batch_scraper.py $RESUME_FLAG > "$LOG_DIR/nohup.log" 2>&1 &

# PIDを保存
echo $! > "$PID_FILE"
SCRAPER_PID=$(cat "$PID_FILE")

echo "バックグラウンド実行開始完了"
echo "PID: $SCRAPER_PID"
echo ""
echo "=== 監視コマンド ==="
echo "進捗確認: python monitor_progress.py"
echo "ログ監視: tail -f $LOG_DIR/scraping.log"
echo "プロセス確認: ps aux | grep $SCRAPER_PID"
echo "停止: kill $SCRAPER_PID"
echo ""
echo "実行が完了すると $PID_FILE が自動削除されます"