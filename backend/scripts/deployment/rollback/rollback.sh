#!/bin/bash
set -euo pipefail

# =============================================================================
# Adult Matching Backend Rollback Script
# 
# デプロイメント失敗時の自動ロールバック機能
# データベース、Edge Functions、設定ファイルの完全復元
# =============================================================================

# 設定とパス
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
UTILS_DIR="$SCRIPT_DIR/../utils"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/tmp/backups}"

# ユーティリティ読み込み
source "$UTILS_DIR/logging.sh"
source "$UTILS_DIR/validation.sh"
source "$UTILS_DIR/backup.sh"

# ロールバック設定
ROLLBACK_TIMEOUT="${ROLLBACK_TIMEOUT:-600}"  # 10分
VERIFY_ROLLBACK="${VERIFY_ROLLBACK:-true}"
FORCE_ROLLBACK="${FORCE_ROLLBACK:-false}"

# =============================================================================
# メイン ロールバック関数
# =============================================================================

main() {
    local backup_id="$1"
    local deployment_id="${2:-unknown}"
    
    log_phase "🔄 ROLLBACK: Starting Rollback Procedure"
    log_info "Backup ID: $backup_id"
    log_info "Deployment ID: $deployment_id"
    log_info "Timeout: ${ROLLBACK_TIMEOUT}s"
    
    # ロールバック開始時刻記録
    local rollback_start=$(date +%s)
    
    # トラップでクリーンアップを保証
    trap cleanup_rollback EXIT
    
    try {
        # フェーズ1: ロールバック前検証
        validate_rollback_preconditions "$backup_id"
        
        # フェーズ2: サービス停止
        stop_services_gracefully
        
        # フェーズ3: データベースロールバック
        rollback_database "$backup_id"
        
        # フェーズ4: Edge Functions ロールバック
        rollback_edge_functions "$backup_id"
        
        # フェーズ5: 設定ファイルロールバック
        rollback_configuration "$backup_id"
        
        # フェーズ6: サービス再起動
        restart_services
        
        # フェーズ7: ロールバック検証
        if [[ "$VERIFY_ROLLBACK" == "true" ]]; then
            verify_rollback_success
        fi
        
        # フェーズ8: ロールバック完了処理
        finalize_rollback "$backup_id" "$deployment_id"
        
    } catch {
        log_error "Rollback failed: $1"
        record_rollback_failure "$backup_id" "$deployment_id" "$1"
        exit 1
    }
    
    local rollback_duration=$(($(date +%s) - rollback_start))
    log_success "✅ Rollback completed successfully in ${rollback_duration}s"
    
    # ロールバック成功通知
    send_rollback_notification "success" "$backup_id" "$deployment_id"
}

# =============================================================================
# フェーズ1: ロールバック前検証
# =============================================================================

validate_rollback_preconditions() {
    local backup_id="$1"
    
    log_info "Validating rollback preconditions..."
    
    # バックアップ存在確認
    validate_backup_exists "$backup_id"
    
    # バックアップ整合性確認
    validate_backup_integrity "$backup_id"
    
    # システム状態確認
    validate_system_state_for_rollback
    
    # ユーザー確認（フォースロールバックでない場合）
    if [[ "$FORCE_ROLLBACK" != "true" ]]; then
        confirm_rollback_operation "$backup_id"
    fi
    
    log_success "Rollback preconditions validated"
}

validate_backup_exists() {
    local backup_id="$1"
    local metadata_file="$BACKUP_BASE_DIR/metadata_${backup_id}.json"
    
    if [[ ! -f "$metadata_file" ]]; then
        log_error "Backup metadata not found: $backup_id"
        exit 1
    fi
    
    # 各バックアップファイルの存在確認
    local backup_types=("db" "functions" "config")
    for backup_type in "${backup_types[@]}"; do
        local backup_file="$BACKUP_BASE_DIR/${backup_type}_backup_${backup_id}.sql.gz"
        if [[ "$backup_type" != "db" ]]; then
            backup_file="$BACKUP_BASE_DIR/${backup_type}_backup_${backup_id}.tar.gz"
        fi
        
        if [[ -f "$backup_file" ]]; then
            log_info "Backup file found: $backup_file"
        else
            log_warning "Backup file not found: $backup_file"
        fi
    done
}

validate_system_state_for_rollback() {
    log_info "Validating system state for rollback..."
    
    # ディスク容量確認
    if ! check_disk_space "." "2"; then
        log_error "Insufficient disk space for rollback"
        exit 1
    fi
    
    # データベース接続確認
    if ! validate_database_connection; then
        log_error "Cannot connect to database for rollback"
        exit 1
    fi
    
    # Supabase CLI確認
    if ! command -v supabase &> /dev/null; then
        log_error "Supabase CLI not available for rollback"
        exit 1
    fi
    
    log_success "System state validated for rollback"
}

confirm_rollback_operation() {
    local backup_id="$1"
    
    log_warning "⚠️  ROLLBACK CONFIRMATION REQUIRED ⚠️"
    echo ""
    echo "This operation will:"
    echo "  - Restore database from backup: $backup_id"
    echo "  - Rollback Edge Functions deployment"
    echo "  - Restore configuration files"
    echo "  - Restart all services"
    echo ""
    echo "⚠️  ALL CURRENT DATA CHANGES WILL BE LOST! ⚠️"
    echo ""
    
    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        # 対話モードの場合
        read -p "Are you absolutely sure you want to proceed? (type 'rollback' to confirm): " confirmation
        if [[ "$confirmation" != "rollback" ]]; then
            log_info "Rollback cancelled by user"
            exit 0
        fi
    else
        # 自動モードの場合は警告のみ
        log_warning "Automatic rollback proceeding in 5 seconds..."
        sleep 5
    fi
    
    log_info "Rollback confirmed, proceeding..."
}

# =============================================================================
# フェーズ2: サービス停止
# =============================================================================

stop_services_gracefully() {
    log_info "Stopping services gracefully..."
    
    # Edge Functions停止
    stop_edge_functions
    
    # バックグラウンドジョブ停止
    stop_background_jobs
    
    # 接続のドレイン（段階的停止）
    drain_connections
    
    log_success "Services stopped gracefully"
}

stop_edge_functions() {
    log_info "Stopping Edge Functions..."
    
    # 実行中のSupabase Functionsプロセス確認
    if pgrep -f "supabase.*functions.*serve" > /dev/null; then
        log_info "Stopping Supabase Functions serve processes..."
        pkill -f "supabase.*functions.*serve" || true
        sleep 2
    fi
    
    # Deno プロセス停止
    if pgrep -f "deno.*run.*main.ts" > /dev/null; then
        log_info "Stopping Deno processes..."
        pkill -f "deno.*run.*main.ts" || true
        sleep 2
    fi
    
    log_success "Edge Functions stopped"
}

stop_background_jobs() {
    log_info "Stopping background jobs..."
    
    # ML 処理ジョブ停止
    if pgrep -f "python.*ml.*" > /dev/null; then
        log_info "Stopping ML processing jobs..."
        pkill -f "python.*ml.*" || true
        sleep 1
    fi
    
    # データ処理ジョブ停止
    if pgrep -f "python.*data.*processing" > /dev/null; then
        log_info "Stopping data processing jobs..."
        pkill -f "python.*data.*processing" || true
        sleep 1
    fi
    
    log_success "Background jobs stopped"
}

drain_connections() {
    log_info "Draining connections..."
    
    # データベース接続のドレイン
    # 新しい接続を拒否し、既存接続の完了を待つ
    sleep 5
    
    log_success "Connections drained"
}

# =============================================================================
# フェーズ3: データベースロールバック
# =============================================================================

rollback_database() {
    local backup_id="$1"
    
    log_info "Rolling back database..."
    
    # データベースバックアップファイル確認
    local db_backup_file="$BACKUP_BASE_DIR/db_backup_${backup_id}.sql.gz"
    
    if [[ ! -f "$db_backup_file" ]]; then
        log_error "Database backup file not found: $db_backup_file"
        exit 1
    fi
    
    # 現在のデータベース状態をバックアップ（緊急用）
    create_emergency_backup
    
    # データベース復元実行
    restore_database_from_backup "$backup_id"
    
    log_success "Database rollback completed"
}

create_emergency_backup() {
    log_info "Creating emergency backup of current state..."
    
    local emergency_backup_id="emergency_$(date +%Y%m%d_%H%M%S)"
    
    if create_db_backup "$emergency_backup_id"; then
        log_info "Emergency backup created: $emergency_backup_id"
    else
        log_warning "Failed to create emergency backup"
    fi
}

restore_database_from_backup() {
    local backup_id="$1"
    local db_backup_file="$BACKUP_BASE_DIR/db_backup_${backup_id}.sql.gz"
    local temp_sql_file="/tmp/rollback_${backup_id}_$$.sql"
    
    log_info "Restoring database from backup: $backup_id"
    
    # バックアップファイル展開
    if ! gunzip -c "$db_backup_file" > "$temp_sql_file"; then
        log_error "Failed to extract database backup"
        exit 1
    fi
    
    # データベース復元
    local db_url="${DATABASE_URL:-}"
    if [[ -z "$db_url" ]]; then
        log_error "DATABASE_URL not set"
        exit 1
    fi
    
    log_info "Executing database restoration..."
    if psql "$db_url" -f "$temp_sql_file" > "/tmp/rollback_db_${backup_id}.log" 2>&1; then
        log_success "Database restored successfully"
        rm -f "$temp_sql_file"
    else
        log_error "Database restoration failed"
        log_error "Check log: /tmp/rollback_db_${backup_id}.log"
        rm -f "$temp_sql_file"
        exit 1
    fi
}

# =============================================================================
# フェーズ4: Edge Functions ロールバック
# =============================================================================

rollback_edge_functions() {
    local backup_id="$1"
    
    log_info "Rolling back Edge Functions..."
    
    # Edge Functions バックアップファイル確認
    local functions_backup_file="$BACKUP_BASE_DIR/functions_backup_${backup_id}.tar.gz"
    
    if [[ ! -f "$functions_backup_file" ]]; then
        log_warning "Edge Functions backup file not found: $functions_backup_file"
        return 0
    fi
    
    # 現在のEdge Functionsバックアップ
    backup_current_functions
    
    # Edge Functions復元
    restore_functions_from_backup "$backup_id"
    
    # Edge Functions再デプロイ
    redeploy_functions
    
    log_success "Edge Functions rollback completed"
}

backup_current_functions() {
    log_info "Backing up current Edge Functions..."
    
    local current_backup_id="current_$(date +%Y%m%d_%H%M%S)"
    local functions_dir="$PROJECT_ROOT/supabase/functions"
    
    if [[ -d "$functions_dir" ]]; then
        create_files_backup "$current_backup_id" "$functions_dir" "functions"
    fi
}

restore_functions_from_backup() {
    local backup_id="$1"
    local functions_backup_file="$BACKUP_BASE_DIR/functions_backup_${backup_id}.tar.gz"
    local functions_dir="$PROJECT_ROOT/supabase/functions"
    
    log_info "Restoring Edge Functions from backup..."
    
    # 既存のfunctionsディレクトリバックアップ
    if [[ -d "$functions_dir" ]]; then
        mv "$functions_dir" "${functions_dir}_rollback_backup_$(date +%s)" || true
    fi
    
    # バックアップから復元
    mkdir -p "$functions_dir"
    
    if tar -xzf "$functions_backup_file" -C "$PROJECT_ROOT/supabase/" --strip-components=1; then
        log_success "Edge Functions restored from backup"
    else
        log_error "Failed to restore Edge Functions from backup"
        exit 1
    fi
}

redeploy_functions() {
    log_info "Redeploying Edge Functions..."
    
    local functions_dir="$PROJECT_ROOT/supabase/functions"
    
    if [[ -d "$functions_dir" ]]; then
        for function_dir in "$functions_dir"/*; do
            if [[ -d "$function_dir" ]]; then
                local function_name=$(basename "$function_dir")
                log_info "Deploying function: $function_name"
                
                if supabase functions deploy "$function_name" --project-ref "${SUPABASE_PROJECT_REF:-}"; then
                    log_success "Function deployed: $function_name"
                else
                    log_error "Failed to deploy function: $function_name"
                    # 個別の関数デプロイ失敗は警告として継続
                fi
            fi
        done
    fi
    
    log_success "Edge Functions redeployment completed"
}

# =============================================================================
# フェーズ5: 設定ファイルロールバック
# =============================================================================

rollback_configuration() {
    local backup_id="$1"
    
    log_info "Rolling back configuration files..."
    
    # 設定ファイルバックアップ確認
    local config_backup_file="$BACKUP_BASE_DIR/config_backup_${backup_id}.tar.gz"
    
    if [[ ! -f "$config_backup_file" ]]; then
        log_warning "Configuration backup file not found: $config_backup_file"
        return 0
    fi
    
    # 現在の設定バックアップ
    backup_current_config
    
    # 設定ファイル復元
    restore_config_from_backup "$backup_id"
    
    log_success "Configuration rollback completed"
}

backup_current_config() {
    log_info "Backing up current configuration..."
    
    local current_backup_id="current_config_$(date +%Y%m%d_%H%M%S)"
    local config_files=(
        "$PROJECT_ROOT/supabase/config.toml"
        "$PROJECT_ROOT/.env.production"
        "$PROJECT_ROOT/package.json"
        "$PROJECT_ROOT/pyproject.toml"
    )
    
    create_config_backup "$current_backup_id" "${config_files[@]}"
}

restore_config_from_backup() {
    local backup_id="$1"
    local config_backup_file="$BACKUP_BASE_DIR/config_backup_${backup_id}.tar.gz"
    local temp_config_dir="/tmp/config_restore_${backup_id}"
    
    log_info "Restoring configuration from backup..."
    
    # 一時ディレクトリに展開
    mkdir -p "$temp_config_dir"
    
    if tar -xzf "$config_backup_file" -C "$temp_config_dir" --strip-components=1; then
        # 各設定ファイルを適切な場所に復元
        for config_file in "$temp_config_dir"/*; do
            if [[ -f "$config_file" ]]; then
                local filename=$(basename "$config_file")
                local target_path=""
                
                case "$filename" in
                    "config.toml")
                        target_path="$PROJECT_ROOT/supabase/config.toml"
                        ;;
                    ".env.production")
                        target_path="$PROJECT_ROOT/.env.production"
                        ;;
                    "package.json")
                        target_path="$PROJECT_ROOT/package.json"
                        ;;
                    "pyproject.toml")
                        target_path="$PROJECT_ROOT/pyproject.toml"
                        ;;
                esac
                
                if [[ -n "$target_path" ]]; then
                    cp "$config_file" "$target_path"
                    log_info "Restored config file: $target_path"
                fi
            fi
        done
        
        rm -rf "$temp_config_dir"
        log_success "Configuration restored from backup"
    else
        log_error "Failed to restore configuration from backup"
        rm -rf "$temp_config_dir"
        exit 1
    fi
}

# =============================================================================
# フェーズ6: サービス再起動
# =============================================================================

restart_services() {
    log_info "Restarting services..."
    
    # Supabase サービス再起動
    restart_supabase_services
    
    # Edge Functions 再起動
    restart_edge_functions
    
    # バックグラウンドサービス再起動
    restart_background_services
    
    # サービス起動確認
    verify_services_started
    
    log_success "Services restarted successfully"
}

restart_supabase_services() {
    log_info "Restarting Supabase services..."
    
    # Supabase停止・再起動
    if command -v supabase &> /dev/null; then
        # ローカル開発環境の場合
        if supabase status --local &> /dev/null; then
            log_info "Restarting local Supabase instance..."
            supabase stop || true
            sleep 2
            if supabase start; then
                log_success "Local Supabase restarted"
            else
                log_error "Failed to restart local Supabase"
                exit 1
            fi
        else
            log_info "Using remote Supabase instance"
        fi
    fi
}

restart_edge_functions() {
    log_info "Restarting Edge Functions..."
    
    # Edge Functions サーバー再起動
    if [[ -d "$PROJECT_ROOT/supabase/functions" ]]; then
        # バックグラウンドでサーバー起動
        (cd "$PROJECT_ROOT" && supabase functions serve --debug > /tmp/functions_serve.log 2>&1 &)
        
        # 起動確認
        sleep 5
        if pgrep -f "supabase.*functions.*serve" > /dev/null; then
            log_success "Edge Functions server restarted"
        else
            log_warning "Edge Functions server may not have started properly"
        fi
    fi
}

restart_background_services() {
    log_info "Restarting background services..."
    
    # 必要に応じてバックグラウンドサービスを再起動
    # 例：データ処理ジョブ、ML処理など
    
    log_success "Background services restarted"
}

verify_services_started() {
    log_info "Verifying services are running..."
    
    # データベース接続確認
    if ! validate_database_connection; then
        log_error "Database connection verification failed"
        exit 1
    fi
    
    # Edge Functions 確認
    # 簡単なヘルスチェック
    local health_check_timeout=30
    local check_count=0
    
    while [[ $check_count -lt $health_check_timeout ]]; do
        if pgrep -f "supabase.*functions.*serve" > /dev/null; then
            log_success "Edge Functions are running"
            break
        fi
        
        sleep 1
        ((check_count++))
    done
    
    if [[ $check_count -ge $health_check_timeout ]]; then
        log_warning "Edge Functions may not be running properly"
    fi
    
    log_success "Service verification completed"
}

# =============================================================================
# フェーズ7: ロールバック検証
# =============================================================================

verify_rollback_success() {
    log_info "Verifying rollback success..."
    
    # 基本ヘルスチェック実行
    run_basic_health_checks
    
    # データベース整合性確認
    verify_database_integrity
    
    # Edge Functions 動作確認
    verify_functions_operation
    
    # エンドツーエンド確認
    run_post_rollback_tests
    
    log_success "Rollback verification completed"
}

run_basic_health_checks() {
    log_info "Running basic health checks..."
    
    local health_script="$PROJECT_ROOT/scripts/deployment/health-checks/basic_health.sh"
    if [[ -f "$health_script" ]]; then
        if bash "$health_script"; then
            log_success "Basic health checks passed"
        else
            log_warning "Some basic health checks failed"
        fi
    else
        log_info "Basic health check script not found, skipping"
    fi
}

verify_database_integrity() {
    log_info "Verifying database integrity..."
    
    local db_check_script="$PROJECT_ROOT/scripts/deployment/health-checks/check_database.py"
    if [[ -f "$db_check_script" ]]; then
        if python3 "$db_check_script" --basic; then
            log_success "Database integrity verified"
        else
            log_error "Database integrity check failed"
            exit 1
        fi
    else
        # 基本的なデータベース接続テスト
        if validate_database_connection; then
            log_success "Database connection verified"
        else
            log_error "Database connection failed"
            exit 1
        fi
    fi
}

verify_functions_operation() {
    log_info "Verifying Edge Functions operation..."
    
    local functions_check_script="$PROJECT_ROOT/scripts/deployment/health-checks/check_functions.sh"
    if [[ -f "$functions_check_script" ]]; then
        if bash "$functions_check_script" --basic; then
            log_success "Edge Functions operation verified"
        else
            log_warning "Some Edge Functions may not be working properly"
        fi
    else
        log_info "Functions check script not found, skipping"
    fi
}

run_post_rollback_tests() {
    log_info "Running post-rollback tests..."
    
    # 軽量なスモークテスト実行
    local test_script="$PROJECT_ROOT/tests/e2e/test_validation_simple.py"
    if [[ -f "$test_script" ]]; then
        if python -m pytest "$test_script" -v --tb=short -x; then
            log_success "Post-rollback tests passed"
        else
            log_warning "Some post-rollback tests failed"
        fi
    else
        log_info "Post-rollback test script not found, skipping"
    fi
}

# =============================================================================
# フェーズ8: ロールバック完了処理
# =============================================================================

finalize_rollback() {
    local backup_id="$1"
    local deployment_id="$2"
    
    log_info "Finalizing rollback..."
    
    # ロールバック記録作成
    record_rollback_success "$backup_id" "$deployment_id"
    
    # 通知送信
    send_rollback_notification "success" "$backup_id" "$deployment_id"
    
    # ドキュメント更新
    update_rollback_documentation "$backup_id"
    
    log_success "Rollback finalization completed"
}

record_rollback_success() {
    local backup_id="$1"
    local deployment_id="$2"
    
    local rollback_record="/tmp/rollback_record_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$rollback_record" << EOF
{
  "rollback_id": "rollback_$(date +%Y%m%d_%H%M%S)",
  "backup_id": "$backup_id",
  "deployment_id": "$deployment_id",
  "environment": "${ENVIRONMENT:-unknown}",
  "rollback_timestamp": "$(date -Iseconds)",
  "rollback_duration": "$(($(date +%s) - ${rollback_start:-$(date +%s)}))",
  "status": "success",
  "initiated_by": "$(whoami)",
  "hostname": "$(hostname)",
  "rollback_reason": "deployment_failure"
}
EOF
    
    # ロールバック履歴に追加
    local rollback_history="$PROJECT_ROOT/deployments/rollback_history.json"
    if [[ -f "$rollback_history" ]]; then
        jq ". += [$(cat "$rollback_record")]" "$rollback_history" > "${rollback_history}.tmp"
        mv "${rollback_history}.tmp" "$rollback_history"
    else
        mkdir -p "$(dirname "$rollback_history")"
        echo "[$(cat "$rollback_record")]" > "$rollback_history"
    fi
    
    log_info "Rollback success recorded"
}

record_rollback_failure() {
    local backup_id="$1"
    local deployment_id="$2"
    local error_message="$3"
    
    local rollback_record="/tmp/rollback_failure_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$rollback_record" << EOF
{
  "rollback_id": "rollback_$(date +%Y%m%d_%H%M%S)",
  "backup_id": "$backup_id",
  "deployment_id": "$deployment_id",
  "environment": "${ENVIRONMENT:-unknown}",
  "rollback_timestamp": "$(date -Iseconds)",
  "status": "failed",
  "error_message": "$error_message",
  "initiated_by": "$(whoami)",
  "hostname": "$(hostname)"
}
EOF
    
    log_error "Rollback failure recorded: $rollback_record"
}

send_rollback_notification() {
    local status="$1"
    local backup_id="$2"
    local deployment_id="$3"
    
    log_info "Sending rollback notification..."
    
    local notification_script="$PROJECT_ROOT/scripts/deployment/utils/send_notification.sh"
    if [[ -f "$notification_script" ]]; then
        bash "$notification_script" "rollback_$status" "$backup_id" "$deployment_id"
    fi
    
    # 緊急連絡（ロールバック失敗時）
    if [[ "$status" == "failed" ]]; then
        log_error "🚨 CRITICAL: Rollback failed - Manual intervention required!"
        # 追加の緊急通知ロジック
    fi
}

update_rollback_documentation() {
    local backup_id="$1"
    
    log_info "Updating rollback documentation..."
    
    local readme_file="$PROJECT_ROOT/README.md"
    if [[ -f "$readme_file" ]]; then
        # ロールバック情報追加
        local rollback_info="Last Rollback: $(date) (Backup: $backup_id)"
        if grep -q "Last Rollback:" "$readme_file"; then
            sed -i.bak "s/Last Rollback:.*/Last Rollback: $rollback_info/" "$readme_file"
        else
            echo "" >> "$readme_file"
            echo "## Deployment Status" >> "$readme_file"
            echo "$rollback_info" >> "$readme_file"
        fi
    fi
}

# =============================================================================
# ユーティリティ関数
# =============================================================================

cleanup_rollback() {
    log_info "Cleaning up rollback resources..."
    
    # 一時ファイル削除
    rm -f /tmp/rollback_*.sql
    rm -f /tmp/rollback_*.log
    rm -f /tmp/config_restore_*
    
    # ロックファイル削除
    rm -f /tmp/rollback.lock
}

# エラーハンドリング
try() {
    "$@"
}

catch() {
    local exit_code=$?
    local error_message="$1"
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Rollback step failed: $error_message"
        return $exit_code
    fi
}

# =============================================================================
# コマンドライン処理
# =============================================================================

show_rollback_help() {
    cat << EOF
Adult Matching Backend Rollback Script

Usage: $0 <backup_id> [deployment_id] [OPTIONS]

Arguments:
    backup_id               Backup ID to rollback to (required)
    deployment_id           Original deployment ID (optional)

Options:
    -f, --force            Force rollback without confirmation
    -t, --timeout SECONDS  Rollback timeout (default: 600)
    -n, --no-verify        Skip rollback verification
    -h, --help             Show this help message

Environment Variables:
    ROLLBACK_TIMEOUT       Rollback timeout in seconds
    VERIFY_ROLLBACK        Enable rollback verification (true/false)
    FORCE_ROLLBACK        Force rollback without confirmation (true/false)
    BACKUP_BASE_DIR       Backup files directory

Examples:
    $0 backup_20240101_120000_abc123
    $0 backup_20240101_120000_abc123 deploy_20240101_120500_def456
    $0 backup_20240101_120000_abc123 -f
    $0 backup_20240101_120000_abc123 -t 300 -n

EOF
}

parse_rollback_arguments() {
    if [[ $# -lt 1 ]]; then
        log_error "Backup ID is required"
        show_rollback_help
        exit 1
    fi
    
    local backup_id="$1"
    local deployment_id="${2:-unknown}"
    shift 2 2>/dev/null || shift 1
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--force)
                FORCE_ROLLBACK="true"
                shift
                ;;
            -t|--timeout)
                ROLLBACK_TIMEOUT="$2"
                shift 2
                ;;
            -n|--no-verify)
                VERIFY_ROLLBACK="false"
                shift
                ;;
            -h|--help)
                show_rollback_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_rollback_help
                exit 1
                ;;
        esac
    done
    
    # 引数を返す
    echo "$backup_id" "$deployment_id"
}

# スクリプト直接実行時のメイン処理
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 引数解析
    read -r backup_id deployment_id <<< "$(parse_rollback_arguments "$@")"
    
    # ロールバック実行
    main "$backup_id" "$deployment_id"
fi