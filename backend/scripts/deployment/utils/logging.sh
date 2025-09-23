#!/bin/bash

# =============================================================================
# Logging Utilities for Deployment Scripts
# 
# デプロイメントスクリプト用ロギングユーティリティ
# =============================================================================

# ログレベル定義
LOG_LEVEL_DEBUG=0
LOG_LEVEL_INFO=1
LOG_LEVEL_WARNING=2
LOG_LEVEL_ERROR=3
LOG_LEVEL_SUCCESS=4

# デフォルトログレベル
LOG_LEVEL="${LOG_LEVEL:-$LOG_LEVEL_INFO}"

# ログファイル設定
LOG_FILE="${DEPLOYMENT_LOG:-/tmp/deployment.log}"

# カラー設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# ログ出力関数
# =============================================================================

# 基本ログ関数
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_entry="[$timestamp] [$level] $message"
    
    # ファイルに出力
    echo "$log_entry" >> "$LOG_FILE"
    
    # 標準出力にカラー付きで出力
    case "$level" in
        "DEBUG")
            if [[ $LOG_LEVEL -le $LOG_LEVEL_DEBUG ]]; then
                echo -e "${CYAN}[DEBUG]${NC} $message"
            fi
            ;;
        "INFO")
            if [[ $LOG_LEVEL -le $LOG_LEVEL_INFO ]]; then
                echo -e "${BLUE}[INFO]${NC} $message"
            fi
            ;;
        "WARNING")
            if [[ $LOG_LEVEL -le $LOG_LEVEL_WARNING ]]; then
                echo -e "${YELLOW}[WARNING]${NC} $message"
            fi
            ;;
        "ERROR")
            if [[ $LOG_LEVEL -le $LOG_LEVEL_ERROR ]]; then
                echo -e "${RED}[ERROR]${NC} $message" >&2
            fi
            ;;
        "SUCCESS")
            if [[ $LOG_LEVEL -le $LOG_LEVEL_SUCCESS ]]; then
                echo -e "${GREEN}[SUCCESS]${NC} $message"
            fi
            ;;
    esac
}

# 個別ログレベル関数
log_debug() {
    log "DEBUG" "$1"
}

log_info() {
    log "INFO" "$1"
}

log_warning() {
    log "WARNING" "$1"
}

log_error() {
    log "ERROR" "$1"
}

log_success() {
    log "SUCCESS" "$1"
}

# フェーズログ（大きなセクション用）
log_phase() {
    local phase_message="$1"
    local separator="=================================="
    
    echo ""
    echo -e "${PURPLE}$separator${NC}"
    echo -e "${PURPLE}$phase_message${NC}"
    echo -e "${PURPLE}$separator${NC}"
    echo ""
    
    log "INFO" "PHASE: $phase_message"
}

# 進捗ログ
log_progress() {
    local current="$1"
    local total="$2"
    local description="$3"
    local percent=$((current * 100 / total))
    local progress_bar=""
    
    # プログレスバー作成（20文字）
    local filled=$((percent / 5))
    for ((i=0; i<filled; i++)); do
        progress_bar+="█"
    done
    for ((i=filled; i<20; i++)); do
        progress_bar+="░"
    done
    
    echo -e "\r${CYAN}[$progress_bar] $percent% ($current/$total) $description${NC}"
    log "INFO" "PROGRESS: $percent% ($current/$total) $description"
}

# 実行時間付きログ
log_with_duration() {
    local start_time="$1"
    local message="$2"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "$message (${duration}s)"
}

# ステップログ（チェックリスト形式）
log_step() {
    local step_number="$1"
    local step_description="$2"
    local status="${3:-PENDING}"
    
    case "$status" in
        "PENDING")
            echo -e "${YELLOW}[$step_number]${NC} $step_description"
            ;;
        "RUNNING")
            echo -e "${BLUE}[$step_number]${NC} $step_description ${BLUE}(running...)${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$step_number] ✅ $step_description${NC}"
            ;;
        "FAILED")
            echo -e "${RED}[$step_number] ❌ $step_description${NC}"
            ;;
        "SKIPPED")
            echo -e "${YELLOW}[$step_number] ⏭️  $step_description (skipped)${NC}"
            ;;
    esac
    
    log "INFO" "STEP $step_number: $step_description ($status)"
}

# エラートレースログ
log_error_trace() {
    local error_message="$1"
    local line_number="${2:-unknown}"
    local function_name="${3:-unknown}"
    
    log_error "Error in function '$function_name' at line $line_number: $error_message"
    
    # スタックトレース出力
    local frame=0
    while caller $frame; do
        ((frame++))
    done | while read line func file; do
        log_debug "  at $func ($file:$line)"
    done
}

# メトリクス付きログ
log_metrics() {
    local metric_name="$1"
    local metric_value="$2"
    local metric_unit="${3:-}"
    local threshold="${4:-}"
    
    local message="METRIC: $metric_name = $metric_value$metric_unit"
    
    if [[ -n "$threshold" ]]; then
        if (( $(echo "$metric_value > $threshold" | bc -l) )); then
            log_warning "$message (exceeds threshold: $threshold$metric_unit)"
        else
            log_info "$message (within threshold: $threshold$metric_unit)"
        fi
    else
        log_info "$message"
    fi
}

# システム情報ログ
log_system_info() {
    log_info "System Information:"
    log_info "  OS: $(uname -s)"
    log_info "  Kernel: $(uname -r)"
    log_info "  Architecture: $(uname -m)"
    log_info "  Hostname: $(hostname)"
    log_info "  User: $(whoami)"
    log_info "  Working Directory: $(pwd)"
    log_info "  Shell: $SHELL"
    
    # リソース情報
    if command -v free &> /dev/null; then
        local memory_info=$(free -h | grep "Mem:" | awk '{print $3 "/" $2}')
        log_info "  Memory Usage: $memory_info"
    fi
    
    if command -v df &> /dev/null; then
        local disk_usage=$(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
        log_info "  Disk Usage: $disk_usage"
    fi
    
    # 環境変数
    log_info "Environment Variables:"
    log_info "  ENVIRONMENT: ${ENVIRONMENT:-not set}"
    log_info "  DRY_RUN: ${DRY_RUN:-not set}"
    log_info "  LOG_LEVEL: ${LOG_LEVEL:-not set}"
}

# ログファイル管理
setup_logging() {
    local log_dir="$(dirname "$LOG_FILE")"
    
    # ログディレクトリ作成
    mkdir -p "$log_dir"
    
    # ログファイル初期化
    echo "# Deployment Log - $(date)" > "$LOG_FILE"
    
    # ログローテーション（古いログを保持）
    if [[ -f "${LOG_FILE}.1" ]]; then
        mv "${LOG_FILE}.1" "${LOG_FILE}.2" 2>/dev/null || true
    fi
    
    if [[ -f "$LOG_FILE" ]] && [[ -s "$LOG_FILE" ]]; then
        mv "$LOG_FILE" "${LOG_FILE}.1" 2>/dev/null || true
    fi
    
    log_info "Logging initialized: $LOG_FILE"
    log_system_info
}

# ログファイル圧縮
compress_logs() {
    local log_dir="$(dirname "$LOG_FILE")"
    local date_suffix=$(date +%Y%m%d_%H%M%S)
    
    # 古いログファイルを圧縮
    for old_log in "${LOG_FILE}".{1,2,3}; do
        if [[ -f "$old_log" ]]; then
            gzip -c "$old_log" > "${old_log}_${date_suffix}.gz"
            rm "$old_log"
        fi
    done
    
    log_info "Log files compressed"
}

# ログ分析
analyze_logs() {
    local log_file="${1:-$LOG_FILE}"
    
    if [[ ! -f "$log_file" ]]; then
        log_error "Log file not found: $log_file"
        return 1
    fi
    
    log_info "Log Analysis for: $log_file"
    
    # 基本統計
    local total_lines=$(wc -l < "$log_file")
    local error_count=$(grep -c "\[ERROR\]" "$log_file" || echo "0")
    local warning_count=$(grep -c "\[WARNING\]" "$log_file" || echo "0")
    local success_count=$(grep -c "\[SUCCESS\]" "$log_file" || echo "0")
    
    log_info "  Total lines: $total_lines"
    log_info "  Errors: $error_count"
    log_info "  Warnings: $warning_count"
    log_info "  Successes: $success_count"
    
    # 実行時間分析
    local start_time=$(grep "Deployment Log" "$log_file" | head -1 | awk '{print $4, $5}')
    local end_time=$(tail -1 "$log_file" | awk '{print $1, $2}' | tr -d '[]')
    
    if [[ -n "$start_time" && -n "$end_time" ]]; then
        log_info "  Start time: $start_time"
        log_info "  End time: $end_time"
    fi
    
    # エラー詳細
    if [[ $error_count -gt 0 ]]; then
        log_warning "Error details:"
        grep "\[ERROR\]" "$log_file" | tail -5 | while read -r line; do
            log_warning "  $line"
        done
    fi
}

# ログクリーンアップ
cleanup_logs() {
    local retention_days="${1:-30}"
    
    # 古いログファイルを削除
    find "$(dirname "$LOG_FILE")" -name "*.log*" -type f -mtime +$retention_days -delete 2>/dev/null || true
    find "$(dirname "$LOG_FILE")" -name "*.gz" -type f -mtime +$retention_days -delete 2>/dev/null || true
    
    log_info "Log cleanup completed (retention: ${retention_days} days)"
}

# ログエクスポート
export_logs() {
    local export_format="${1:-json}"
    local output_file="${2:-deployment_logs_$(date +%Y%m%d_%H%M%S).$export_format}"
    
    case "$export_format" in
        "json")
            # JSON形式でエクスポート
            echo "{" > "$output_file"
            echo "  \"deployment_log\": \"$LOG_FILE\"," >> "$output_file"
            echo "  \"export_time\": \"$(date -Iseconds)\"," >> "$output_file"
            echo "  \"entries\": [" >> "$output_file"
            
            local first_entry=true
            while IFS= read -r line; do
                if [[ "$first_entry" == "true" ]]; then
                    first_entry=false
                else
                    echo "," >> "$output_file"
                fi
                echo -n "    \"$(echo "$line" | sed 's/"/\\"/g')\"" >> "$output_file"
            done < "$LOG_FILE"
            
            echo "" >> "$output_file"
            echo "  ]" >> "$output_file"
            echo "}" >> "$output_file"
            ;;
        "csv")
            # CSV形式でエクスポート
            echo "timestamp,level,message" > "$output_file"
            while IFS= read -r line; do
                if [[ "$line" =~ ^\[([0-9-]+\ [0-9:]+)\]\ \[([A-Z]+)\]\ (.*)$ ]]; then
                    echo "\"${BASH_REMATCH[1]}\",\"${BASH_REMATCH[2]}\",\"${BASH_REMATCH[3]}\"" >> "$output_file"
                fi
            done < "$LOG_FILE"
            ;;
        *)
            # プレーンテキストとしてコピー
            cp "$LOG_FILE" "$output_file"
            ;;
    esac
    
    log_info "Logs exported to: $output_file (format: $export_format)"
}

# 初期化（スクリプト読み込み時に実行）
if [[ -z "${LOGGING_INITIALIZED:-}" ]]; then
    setup_logging
    export LOGGING_INITIALIZED=true
fi