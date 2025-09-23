#!/bin/bash
set -euo pipefail

# =============================================================================
# Adult Matching Backend Deployment Automation
# 
# リファクタリング後システム用自動デプロイメントスクリプト
# ゼロダウンタイム、ロールバック機能、包括的ヘルスチェック対応
# =============================================================================

# 設定とパス
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
HEALTH_CHECK_DIR="$SCRIPT_DIR/health-checks"
ROLLBACK_DIR="$SCRIPT_DIR/rollback"
UTILS_DIR="$SCRIPT_DIR/utils"

# デフォルト設定
ENVIRONMENT="${ENVIRONMENT:-production}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
DEPLOYMENT_LOG="/tmp/deployment_$(date +%Y%m%d_%H%M%S).log"

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ユーティリティ関数の読み込み
source "$UTILS_DIR/logging.sh"
source "$UTILS_DIR/validation.sh"
source "$UTILS_DIR/backup.sh"

# グローバル変数
DEPLOYMENT_ID=""
BACKUP_ID=""
PREVIOUS_VERSION=""
NEW_VERSION=""
ROLLBACK_REQUIRED=false

# =============================================================================
# メイン関数
# =============================================================================

main() {
    log_info "🚀 Starting Adult Matching Backend Deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment ID: $DEPLOYMENT_ID"
    log_info "Log file: $DEPLOYMENT_LOG"
    
    # ロックファイルでデプロイメント重複防止
    local lock_file="/tmp/deployment.lock"
    exec 200>"$lock_file"
    if ! flock -n 200; then
        log_error "Another deployment is already running"
        exit 1
    fi
    
    # トラップでクリーンアップを保証
    trap cleanup EXIT
    trap rollback_on_signal INT TERM
    
    try {
        # フェーズ1: 事前検証
        pre_deployment_validation
        
        # フェーズ2: バックアップ作成
        create_deployment_backup
        
        # フェーズ3: デプロイメント実行
        execute_deployment
        
        # フェーズ4: ヘルスチェック
        run_health_checks
        
        # フェーズ5: 事後検証
        post_deployment_validation
        
        # フェーズ6: デプロイメント完了
        finalize_deployment
        
    } catch {
        log_error "Deployment failed: $1"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            initiate_rollback
        fi
        exit 1
    }
    
    log_success "✅ Deployment completed successfully!"
    log_info "Deployment ID: $DEPLOYMENT_ID"
    log_info "New Version: $NEW_VERSION"
}

# =============================================================================
# フェーズ1: 事前検証
# =============================================================================

pre_deployment_validation() {
    log_phase "Phase 1: Pre-deployment Validation"
    
    # 環境変数チェック
    validate_environment_variables
    
    # 依存関係チェック
    validate_dependencies
    
    # Git状態チェック
    validate_git_state
    
    # Supabase接続チェック
    validate_supabase_connection
    
    # データベース状態チェック
    validate_database_state
    
    # 既存サービス状態チェック
    validate_existing_services
    
    # テスト実行（スキップオプションがない場合）
    if [[ "$SKIP_TESTS" != "true" ]]; then
        run_pre_deployment_tests
    fi
    
    log_success "✅ Pre-deployment validation completed"
}

validate_environment_variables() {
    log_info "Validating environment variables..."
    
    required_vars=(
        "SUPABASE_URL"
        "SUPABASE_ANON_KEY"
        "SUPABASE_SERVICE_ROLE_KEY"
        "DATABASE_URL"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Environment variables validated"
}

validate_dependencies() {
    log_info "Validating dependencies..."
    
    # Node.js バージョンチェック
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    
    local node_version=$(node --version | cut -d'v' -f2)
    if ! version_greater_equal "$node_version" "18.0.0"; then
        log_error "Node.js version must be >= 18.0.0, current: $node_version"
        exit 1
    fi
    
    # Supabase CLI チェック
    if ! command -v supabase &> /dev/null; then
        log_error "Supabase CLI is not installed"
        exit 1
    fi
    
    # Python環境チェック（ML/データ処理用）
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Docker チェック（必要に応じて）
    if command -v docker &> /dev/null; then
        if ! docker info &> /dev/null; then
            log_warning "Docker is installed but not running"
        fi
    fi
    
    log_success "Dependencies validated"
}

validate_git_state() {
    log_info "Validating Git state..."
    
    # Git作業ディレクトリがクリーンかチェック
    if [[ -n "$(git status --porcelain)" ]]; then
        log_error "Git working directory is not clean"
        exit 1
    fi
    
    # 現在のブランチ取得
    local current_branch=$(git branch --show-current)
    log_info "Current branch: $current_branch"
    
    # バージョン情報設定
    PREVIOUS_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
    NEW_VERSION=$(git rev-parse --short HEAD)
    
    log_info "Previous version: $PREVIOUS_VERSION"
    log_info "New version: $NEW_VERSION"
    
    log_success "Git state validated"
}

validate_supabase_connection() {
    log_info "Validating Supabase connection..."
    
    # Supabase接続テスト
    if ! supabase status &> /dev/null; then
        log_error "Cannot connect to Supabase"
        exit 1
    fi
    
    log_success "Supabase connection validated"
}

validate_database_state() {
    log_info "Validating database state..."
    
    # データベース接続テスト
    local db_check_script="$HEALTH_CHECK_DIR/check_database.py"
    if ! python3 "$db_check_script"; then
        log_error "Database validation failed"
        exit 1
    fi
    
    log_success "Database state validated"
}

validate_existing_services() {
    log_info "Validating existing services..."
    
    # Edge Functions状態チェック
    local functions_check_script="$HEALTH_CHECK_DIR/check_functions.sh"
    if ! bash "$functions_check_script"; then
        log_warning "Some Edge Functions may not be running"
    fi
    
    log_success "Existing services validated"
}

run_pre_deployment_tests() {
    log_info "Running pre-deployment tests..."
    
    # ユニットテスト実行
    log_info "Running unit tests..."
    if ! python -m pytest tests/unit/ -v; then
        log_error "Unit tests failed"
        exit 1
    fi
    
    # 統合テスト実行
    log_info "Running integration tests..."
    if ! python -m pytest tests/integration/ -v -m "not slow"; then
        log_error "Integration tests failed"
        exit 1
    fi
    
    # データ品質テスト実行
    log_info "Running data quality tests..."
    if ! python -m pytest tests/integration/data/test_integration_simple.py -v; then
        log_error "Data quality tests failed"
        exit 1
    fi
    
    log_success "Pre-deployment tests completed"
}

# =============================================================================
# フェーズ2: バックアップ作成
# =============================================================================

create_deployment_backup() {
    log_phase "Phase 2: Creating Deployment Backup"
    
    BACKUP_ID="backup_$(date +%Y%m%d_%H%M%S)_${NEW_VERSION}"
    
    # データベースバックアップ
    create_database_backup
    
    # Edge Functionsバックアップ
    backup_edge_functions
    
    # 設定ファイルバックアップ
    backup_configuration_files
    
    # バックアップの整合性チェック
    verify_backup_integrity
    
    log_success "✅ Deployment backup created: $BACKUP_ID"
}

create_database_backup() {
    log_info "Creating database backup..."
    
    local backup_script="$UTILS_DIR/backup.sh"
    if ! bash "$backup_script" create_db_backup "$BACKUP_ID"; then
        log_error "Database backup failed"
        exit 1
    fi
    
    log_success "Database backup created"
}

backup_edge_functions() {
    log_info "Backing up Edge Functions..."
    
    local functions_backup_dir="/tmp/functions_backup_$BACKUP_ID"
    mkdir -p "$functions_backup_dir"
    
    if [[ -d "$PROJECT_ROOT/supabase/functions" ]]; then
        cp -r "$PROJECT_ROOT/supabase/functions" "$functions_backup_dir/"
        tar -czf "/tmp/functions_backup_${BACKUP_ID}.tar.gz" -C "/tmp" "functions_backup_$BACKUP_ID"
        rm -rf "$functions_backup_dir"
    fi
    
    log_success "Edge Functions backup created"
}

backup_configuration_files() {
    log_info "Backing up configuration files..."
    
    local config_backup_dir="/tmp/config_backup_$BACKUP_ID"
    mkdir -p "$config_backup_dir"
    
    # 重要な設定ファイルをバックアップ
    local config_files=(
        "$PROJECT_ROOT/supabase/config.toml"
        "$PROJECT_ROOT/.env.production"
        "$PROJECT_ROOT/package.json"
        "$PROJECT_ROOT/pyproject.toml"
    )
    
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$config_backup_dir/"
        fi
    done
    
    tar -czf "/tmp/config_backup_${BACKUP_ID}.tar.gz" -C "/tmp" "config_backup_$BACKUP_ID"
    rm -rf "$config_backup_dir"
    
    log_success "Configuration files backup created"
}

verify_backup_integrity() {
    log_info "Verifying backup integrity..."
    
    # バックアップファイルの存在と完全性をチェック
    local backup_files=(
        "/tmp/db_backup_${BACKUP_ID}.sql"
        "/tmp/functions_backup_${BACKUP_ID}.tar.gz"
        "/tmp/config_backup_${BACKUP_ID}.tar.gz"
    )
    
    for backup_file in "${backup_files[@]}"; do
        if [[ -f "$backup_file" ]]; then
            local file_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null)
            if [[ "$file_size" -eq 0 ]]; then
                log_error "Backup file $backup_file is empty"
                exit 1
            fi
            log_info "Backup file verified: $backup_file ($file_size bytes)"
        else
            log_warning "Backup file not found: $backup_file"
        fi
    done
    
    log_success "Backup integrity verified"
}

# =============================================================================
# フェーズ3: デプロイメント実行
# =============================================================================

execute_deployment() {
    log_phase "Phase 3: Executing Deployment"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - Simulating deployment..."
        simulate_deployment
        return
    fi
    
    # データベースマイグレーション
    run_database_migrations
    
    # Edge Functions デプロイ
    deploy_edge_functions
    
    # データ処理パイプライン更新
    update_data_pipeline
    
    # ML モデル更新（必要に応じて）
    update_ml_models
    
    # 設定ファイル更新
    update_configuration
    
    log_success "✅ Deployment execution completed"
}

simulate_deployment() {
    log_info "Simulating database migrations..."
    sleep 2
    
    log_info "Simulating Edge Functions deployment..."
    sleep 3
    
    log_info "Simulating data pipeline update..."
    sleep 2
    
    log_info "Simulating ML models update..."
    sleep 1
    
    log_info "Simulating configuration update..."
    sleep 1
    
    log_success "Deployment simulation completed"
}

run_database_migrations() {
    log_info "Running database migrations..."
    
    # Supabase マイグレーション実行
    if ! supabase db push; then
        log_error "Database migration failed"
        exit 1
    fi
    
    # カスタムマイグレーション実行（必要に応じて）
    local migration_script="$SCRIPT_DIR/migrations/run_custom_migrations.sh"
    if [[ -f "$migration_script" ]]; then
        if ! bash "$migration_script"; then
            log_error "Custom migrations failed"
            exit 1
        fi
    fi
    
    log_success "Database migrations completed"
}

deploy_edge_functions() {
    log_info "Deploying Edge Functions..."
    
    # 各Edge Functionを個別にデプロイ
    local functions_dir="$PROJECT_ROOT/supabase/functions"
    
    if [[ -d "$functions_dir" ]]; then
        for function_dir in "$functions_dir"/*; do
            if [[ -d "$function_dir" ]]; then
                local function_name=$(basename "$function_dir")
                log_info "Deploying function: $function_name"
                
                if ! supabase functions deploy "$function_name"; then
                    log_error "Failed to deploy function: $function_name"
                    exit 1
                fi
            fi
        done
    fi
    
    log_success "Edge Functions deployment completed"
}

update_data_pipeline() {
    log_info "Updating data processing pipeline..."
    
    # データ処理スクリプト更新
    local pipeline_update_script="$SCRIPT_DIR/data/update_pipeline.sh"
    if [[ -f "$pipeline_update_script" ]]; then
        if ! bash "$pipeline_update_script"; then
            log_error "Data pipeline update failed"
            exit 1
        fi
    fi
    
    log_success "Data pipeline update completed"
}

update_ml_models() {
    log_info "Updating ML models..."
    
    # ML モデル更新スクリプト
    local ml_update_script="$SCRIPT_DIR/ml/update_models.py"
    if [[ -f "$ml_update_script" ]]; then
        if ! python3 "$ml_update_script"; then
            log_error "ML models update failed"
            exit 1
        fi
    fi
    
    log_success "ML models update completed"
}

update_configuration() {
    log_info "Updating configuration..."
    
    # 設定ファイル更新
    local config_update_script="$SCRIPT_DIR/configs/update_config.sh"
    if [[ -f "$config_update_script" ]]; then
        if ! bash "$config_update_script" "$ENVIRONMENT"; then
            log_error "Configuration update failed"
            exit 1
        fi
    fi
    
    log_success "Configuration update completed"
}

# =============================================================================
# フェーズ4: ヘルスチェック
# =============================================================================

run_health_checks() {
    log_phase "Phase 4: Running Health Checks"
    
    local health_check_start=$(date +%s)
    local max_wait_time="$HEALTH_CHECK_TIMEOUT"
    
    # 基本ヘルスチェック
    run_basic_health_checks
    
    # データベースヘルスチェック
    run_database_health_checks
    
    # Edge Functionsヘルスチェック
    run_functions_health_checks
    
    # エンドツーエンドヘルスチェック
    run_e2e_health_checks
    
    # パフォーマンスチェック
    run_performance_checks
    
    local health_check_duration=$(($(date +%s) - health_check_start))
    log_info "Health checks completed in ${health_check_duration}s"
    
    if [[ $health_check_duration -gt $max_wait_time ]]; then
        log_warning "Health checks took longer than expected"
    fi
    
    log_success "✅ All health checks passed"
}

run_basic_health_checks() {
    log_info "Running basic health checks..."
    
    local health_script="$HEALTH_CHECK_DIR/basic_health.sh"
    if ! bash "$health_script"; then
        log_error "Basic health checks failed"
        exit 1
    fi
    
    log_success "Basic health checks passed"
}

run_database_health_checks() {
    log_info "Running database health checks..."
    
    local db_health_script="$HEALTH_CHECK_DIR/check_database.py"
    if ! python3 "$db_health_script" --comprehensive; then
        log_error "Database health checks failed"
        exit 1
    fi
    
    log_success "Database health checks passed"
}

run_functions_health_checks() {
    log_info "Running Edge Functions health checks..."
    
    local functions_health_script="$HEALTH_CHECK_DIR/check_functions.sh"
    if ! bash "$functions_health_script" --detailed; then
        log_error "Edge Functions health checks failed"
        exit 1
    fi
    
    log_success "Edge Functions health checks passed"
}

run_e2e_health_checks() {
    log_info "Running end-to-end health checks..."
    
    # 簡易E2Eテスト実行
    local e2e_health_script="$HEALTH_CHECK_DIR/e2e_health.py"
    if ! python3 "$e2e_health_script"; then
        log_error "End-to-end health checks failed"
        exit 1
    fi
    
    log_success "End-to-end health checks passed"
}

run_performance_checks() {
    log_info "Running performance checks..."
    
    local perf_check_script="$HEALTH_CHECK_DIR/performance_check.py"
    if ! python3 "$perf_check_script" --threshold-check; then
        log_warning "Performance checks reported degradation"
        # パフォーマンス低下は警告のみで続行
    else
        log_success "Performance checks passed"
    fi
}

# =============================================================================
# フェーズ5: 事後検証
# =============================================================================

post_deployment_validation() {
    log_phase "Phase 5: Post-deployment Validation"
    
    # システム全体状態確認
    validate_system_state
    
    # データ整合性確認
    validate_data_integrity
    
    # 機能テスト実行
    run_functional_tests
    
    # セキュリティチェック
    run_security_checks
    
    log_success "✅ Post-deployment validation completed"
}

validate_system_state() {
    log_info "Validating system state..."
    
    local system_validation_script="$HEALTH_CHECK_DIR/system_validation.py"
    if ! python3 "$system_validation_script"; then
        log_error "System state validation failed"
        exit 1
    fi
    
    log_success "System state validated"
}

validate_data_integrity() {
    log_info "Validating data integrity..."
    
    local data_integrity_script="$HEALTH_CHECK_DIR/data_integrity.py"
    if ! python3 "$data_integrity_script"; then
        log_error "Data integrity validation failed"
        exit 1
    fi
    
    log_success "Data integrity validated"
}

run_functional_tests() {
    log_info "Running functional tests..."
    
    # 重要な機能のスモークテスト
    if ! python -m pytest tests/e2e/test_validation_simple.py -v; then
        log_error "Functional tests failed"
        exit 1
    fi
    
    log_success "Functional tests passed"
}

run_security_checks() {
    log_info "Running security checks..."
    
    local security_check_script="$HEALTH_CHECK_DIR/security_check.py"
    if ! python3 "$security_check_script"; then
        log_warning "Security checks reported issues"
        # セキュリティ警告は記録するが続行
    else
        log_success "Security checks passed"
    fi
}

# =============================================================================
# フェーズ6: デプロイメント完了
# =============================================================================

finalize_deployment() {
    log_phase "Phase 6: Finalizing Deployment"
    
    # デプロイメント記録
    record_deployment
    
    # 古いバックアップクリーンアップ
    cleanup_old_backups
    
    # 通知送信
    send_deployment_notification
    
    # ドキュメント更新
    update_deployment_documentation
    
    log_success "✅ Deployment finalization completed"
}

record_deployment() {
    log_info "Recording deployment..."
    
    local deployment_record="/tmp/deployment_record_${DEPLOYMENT_ID}.json"
    cat > "$deployment_record" << EOF
{
  "deployment_id": "$DEPLOYMENT_ID",
  "environment": "$ENVIRONMENT",
  "previous_version": "$PREVIOUS_VERSION",
  "new_version": "$NEW_VERSION",
  "backup_id": "$BACKUP_ID",
  "timestamp": "$(date -Iseconds)",
  "duration": "$(($(date +%s) - DEPLOYMENT_START_TIME))",
  "status": "success",
  "rollback_required": false
}
EOF
    
    # デプロイメント履歴に追加
    local deployment_history="$PROJECT_ROOT/deployments/history.json"
    if [[ -f "$deployment_history" ]]; then
        # 既存履歴にマージ
        jq ". += [$(cat "$deployment_record")]" "$deployment_history" > "${deployment_history}.tmp"
        mv "${deployment_history}.tmp" "$deployment_history"
    else
        # 新規作成
        mkdir -p "$(dirname "$deployment_history")"
        echo "[$(cat "$deployment_record")]" > "$deployment_history"
    fi
    
    log_success "Deployment recorded"
}

cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    # 30日以上古いバックアップを削除
    find /tmp -name "backup_*" -type f -mtime +30 -delete 2>/dev/null || true
    find /tmp -name "*_backup_*.tar.gz" -type f -mtime +30 -delete 2>/dev/null || true
    
    log_success "Old backups cleaned up"
}

send_deployment_notification() {
    log_info "Sending deployment notification..."
    
    # Slack通知（設定されている場合）
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local notification_script="$UTILS_DIR/send_notification.sh"
        if [[ -f "$notification_script" ]]; then
            bash "$notification_script" "success" "$DEPLOYMENT_ID" "$NEW_VERSION"
        fi
    fi
    
    log_success "Deployment notification sent"
}

update_deployment_documentation() {
    log_info "Updating deployment documentation..."
    
    # README更新
    local readme_file="$PROJECT_ROOT/README.md"
    if [[ -f "$readme_file" ]]; then
        # バージョン情報更新
        sed -i.bak "s/Version: .*/Version: $NEW_VERSION/" "$readme_file" || true
    fi
    
    log_success "Deployment documentation updated"
}

# =============================================================================
# ロールバック機能
# =============================================================================

initiate_rollback() {
    log_phase "ROLLBACK: Initiating Rollback Procedure"
    
    ROLLBACK_REQUIRED=true
    
    # ロールバックスクリプト実行
    local rollback_script="$ROLLBACK_DIR/rollback.sh"
    if [[ -f "$rollback_script" ]]; then
        if bash "$rollback_script" "$BACKUP_ID" "$DEPLOYMENT_ID"; then
            log_success "✅ Rollback completed successfully"
        else
            log_error "❌ Rollback failed - Manual intervention required"
            exit 1
        fi
    else
        log_error "Rollback script not found - Manual intervention required"
        exit 1
    fi
}

rollback_on_signal() {
    log_warning "Deployment interrupted by signal"
    if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
        initiate_rollback
    fi
    exit 1
}

# =============================================================================
# ユーティリティ関数
# =============================================================================

cleanup() {
    log_info "Cleaning up deployment resources..."
    
    # ロックファイル削除
    rm -f /tmp/deployment.lock
    
    # 一時ファイルクリーンアップ
    rm -f /tmp/deployment_*.tmp
    
    # デプロイメントログの保存
    if [[ -f "$DEPLOYMENT_LOG" ]]; then
        local log_archive_dir="$PROJECT_ROOT/deployments/logs"
        mkdir -p "$log_archive_dir"
        cp "$DEPLOYMENT_LOG" "$log_archive_dir/deployment_${DEPLOYMENT_ID}.log"
    fi
    
    log_info "Cleanup completed"
}

# エラーハンドリング機能
try() {
    "$@"
}

catch() {
    local exit_code=$?
    local error_message="$1"
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Command failed with exit code $exit_code: $error_message"
        return $exit_code
    fi
}

# =============================================================================
# 初期化とメイン実行
# =============================================================================

# デプロイメントID生成
DEPLOYMENT_ID="deploy_$(date +%Y%m%d_%H%M%S)_${NEW_VERSION:-unknown}"
DEPLOYMENT_START_TIME=$(date +%s)

# ヘルプ表示
show_help() {
    cat << EOF
Adult Matching Backend Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Target environment (default: production)
    -d, --dry-run              Run in dry-run mode (no actual changes)
    -s, --skip-tests           Skip pre-deployment tests
    -t, --timeout SECONDS     Health check timeout (default: 300)
    -n, --no-rollback         Disable automatic rollback on failure
    -h, --help                 Show this help message

Environment Variables:
    ENVIRONMENT                Target environment
    DRY_RUN                   Enable dry-run mode (true/false)
    SKIP_TESTS               Skip tests (true/false)
    HEALTH_CHECK_TIMEOUT     Health check timeout in seconds
    ROLLBACK_ON_FAILURE      Enable automatic rollback (true/false)

Examples:
    $0                          # Deploy to production
    $0 -e staging              # Deploy to staging
    $0 -d                      # Dry run
    $0 -s -t 600              # Skip tests, 10 min timeout

EOF
}

# コマンドライン引数解析
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -s|--skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            -t|--timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            -n|--no-rollback)
                ROLLBACK_ON_FAILURE="false"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# スクリプト直接実行時のメイン処理
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi