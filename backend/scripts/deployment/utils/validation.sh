#!/bin/bash

# =============================================================================
# Validation Utilities for Deployment Scripts
# 
# デプロイメントスクリプト用検証ユーティリティ
# =============================================================================

# バージョン比較関数
version_greater_equal() {
    local version1="$1"
    local version2="$2"
    
    # セマンティックバージョニング対応
    printf '%s\n%s\n' "$version2" "$version1" | sort -V -C
}

# ポート可用性チェック
check_port_available() {
    local port="$1"
    local host="${2:-localhost}"
    
    if command -v nc &> /dev/null; then
        ! nc -z "$host" "$port" 2>/dev/null
    elif command -v telnet &> /dev/null; then
        ! timeout 1 telnet "$host" "$port" &>/dev/null
    else
        # netcatやtelnetがない場合、/dev/tcpを使用
        ! timeout 1 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null
    fi
}

# URL可用性チェック
check_url_available() {
    local url="$1"
    local timeout="${2:-10}"
    local expected_status="${3:-200}"
    
    if command -v curl &> /dev/null; then
        local status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url")
        [[ "$status_code" == "$expected_status" ]]
    elif command -v wget &> /dev/null; then
        wget --quiet --timeout="$timeout" --spider "$url"
    else
        log_error "Neither curl nor wget available for URL checking"
        return 1
    fi
}

# ディスク容量チェック
check_disk_space() {
    local path="${1:-.}"
    local required_space_gb="$2"
    
    local available_space_kb=$(df "$path" | tail -1 | awk '{print $4}')
    local available_space_gb=$((available_space_kb / 1024 / 1024))
    
    [[ $available_space_gb -ge $required_space_gb ]]
}

# メモリ容量チェック
check_memory_available() {
    local required_memory_gb="$1"
    
    if command -v free &> /dev/null; then
        local available_memory_kb=$(free | grep "Mem:" | awk '{print $7}')
        local available_memory_gb=$((available_memory_kb / 1024 / 1024))
        
        [[ $available_memory_gb -ge $required_memory_gb ]]
    else
        log_warning "Cannot check memory availability (free command not found)"
        return 0  # スキップ
    fi
}

# プロセス実行チェック
check_process_running() {
    local process_name="$1"
    
    pgrep -f "$process_name" > /dev/null
}

# ファイル存在・権限チェック
check_file_permissions() {
    local file_path="$1"
    local required_permissions="$2"  # 例: "r", "w", "x", "rw", "rwx"
    
    if [[ ! -e "$file_path" ]]; then
        log_error "File does not exist: $file_path"
        return 1
    fi
    
    case "$required_permissions" in
        *r*)
            if [[ ! -r "$file_path" ]]; then
                log_error "File is not readable: $file_path"
                return 1
            fi
            ;;
    esac
    
    case "$required_permissions" in
        *w*)
            if [[ ! -w "$file_path" ]]; then
                log_error "File is not writable: $file_path"
                return 1
            fi
            ;;
    esac
    
    case "$required_permissions" in
        *x*)
            if [[ ! -x "$file_path" ]]; then
                log_error "File is not executable: $file_path"
                return 1
            fi
            ;;
    esac
    
    return 0
}

# ディレクトリ構造検証
validate_directory_structure() {
    local base_dir="$1"
    shift
    local required_dirs=("$@")
    
    for dir in "${required_dirs[@]}"; do
        local full_path="$base_dir/$dir"
        if [[ ! -d "$full_path" ]]; then
            log_error "Required directory missing: $full_path"
            return 1
        fi
    done
    
    return 0
}

# JSON設定ファイル検証
validate_json_config() {
    local config_file="$1"
    local required_keys=("${@:2}")
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        return 1
    fi
    
    # JSON形式チェック
    if ! jq empty "$config_file" 2>/dev/null; then
        log_error "Invalid JSON format: $config_file"
        return 1
    fi
    
    # 必須キーチェック
    for key in "${required_keys[@]}"; do
        if ! jq -e ".$key" "$config_file" >/dev/null 2>&1; then
            log_error "Missing required key '$key' in $config_file"
            return 1
        fi
    done
    
    return 0
}

# 環境変数検証
validate_environment_variable() {
    local var_name="$1"
    local var_pattern="${2:-.*}"  # デフォルトは任意の値
    local var_description="${3:-$var_name}"
    
    local var_value="${!var_name:-}"
    
    if [[ -z "$var_value" ]]; then
        log_error "Environment variable $var_name is not set ($var_description)"
        return 1
    fi
    
    if [[ ! "$var_value" =~ $var_pattern ]]; then
        log_error "Environment variable $var_name does not match required pattern: $var_pattern"
        return 1
    fi
    
    return 0
}

# データベース接続検証
validate_database_connection() {
    local db_url="${1:-$DATABASE_URL}"
    local timeout="${2:-10}"
    
    if [[ -z "$db_url" ]]; then
        log_error "Database URL not provided"
        return 1
    fi
    
    # PostgreSQL接続テスト
    if command -v psql &> /dev/null; then
        if timeout "$timeout" psql "$db_url" -c "SELECT 1;" &>/dev/null; then
            log_success "Database connection validated"
            return 0
        else
            log_error "Database connection failed"
            return 1
        fi
    else
        log_warning "psql not available, skipping database connection test"
        return 0
    fi
}

# API エンドポイント検証
validate_api_endpoint() {
    local endpoint_url="$1"
    local expected_response="${2:-.*}"
    local timeout="${3:-10}"
    local auth_header="${4:-}"
    
    local curl_args=(-s --max-time "$timeout")
    
    if [[ -n "$auth_header" ]]; then
        curl_args+=(-H "Authorization: $auth_header")
    fi
    
    local response=$(curl "${curl_args[@]}" "$endpoint_url")
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "API endpoint unreachable: $endpoint_url"
        return 1
    fi
    
    if [[ ! "$response" =~ $expected_response ]]; then
        log_error "API endpoint response validation failed: $endpoint_url"
        log_debug "Expected pattern: $expected_response"
        log_debug "Actual response: $response"
        return 1
    fi
    
    log_success "API endpoint validated: $endpoint_url"
    return 0
}

# SSL証明書検証
validate_ssl_certificate() {
    local hostname="$1"
    local port="${2:-443}"
    local days_warning="${3:-30}"
    
    if ! command -v openssl &> /dev/null; then
        log_warning "openssl not available, skipping SSL certificate validation"
        return 0
    fi
    
    local cert_info=$(echo | openssl s_client -servername "$hostname" -connect "$hostname:$port" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
    
    if [[ -z "$cert_info" ]]; then
        log_error "Could not retrieve SSL certificate for $hostname:$port"
        return 1
    fi
    
    local expiry_date=$(echo "$cert_info" | grep "notAfter=" | cut -d= -f2)
    local expiry_epoch=$(date -d "$expiry_date" +%s 2>/dev/null || date -j -f "%b %d %H:%M:%S %Y %Z" "$expiry_date" +%s 2>/dev/null)
    local current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    if [[ $days_until_expiry -lt 0 ]]; then
        log_error "SSL certificate for $hostname has expired"
        return 1
    elif [[ $days_until_expiry -lt $days_warning ]]; then
        log_warning "SSL certificate for $hostname expires in $days_until_expiry days"
    else
        log_success "SSL certificate for $hostname is valid (expires in $days_until_expiry days)"
    fi
    
    return 0
}

# Docker コンテナ状態検証
validate_docker_container() {
    local container_name="$1"
    local expected_status="${2:-running}"
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available, skipping container validation"
        return 0
    fi
    
    local container_status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null)
    
    if [[ -z "$container_status" ]]; then
        log_error "Container not found: $container_name"
        return 1
    fi
    
    if [[ "$container_status" != "$expected_status" ]]; then
        log_error "Container $container_name status is '$container_status', expected '$expected_status'"
        return 1
    fi
    
    log_success "Container $container_name is $container_status"
    return 0
}

# Git リポジトリ状態検証
validate_git_repository() {
    local repo_path="${1:-.}"
    local required_branch="${2:-}"
    local require_clean="${3:-true}"
    
    if [[ ! -d "$repo_path/.git" ]]; then
        log_error "Not a Git repository: $repo_path"
        return 1
    fi
    
    cd "$repo_path" || return 1
    
    # クリーンな作業ディレクトリチェック
    if [[ "$require_clean" == "true" ]]; then
        if [[ -n "$(git status --porcelain)" ]]; then
            log_error "Git working directory is not clean"
            return 1
        fi
    fi
    
    # ブランチチェック
    if [[ -n "$required_branch" ]]; then
        local current_branch=$(git branch --show-current)
        if [[ "$current_branch" != "$required_branch" ]]; then
            log_error "Current branch '$current_branch' is not the required branch '$required_branch'"
            return 1
        fi
    fi
    
    # リモート同期チェック
    if git remote | grep -q "origin"; then
        local local_commit=$(git rev-parse HEAD)
        local remote_commit=$(git rev-parse origin/$(git branch --show-current) 2>/dev/null || echo "")
        
        if [[ -n "$remote_commit" && "$local_commit" != "$remote_commit" ]]; then
            log_warning "Local branch is not in sync with remote"
        fi
    fi
    
    log_success "Git repository validation passed"
    return 0
}

# ネットワーク接続性検証
validate_network_connectivity() {
    local test_hosts=("google.com" "github.com" "api.supabase.io")
    local timeout="${1:-5}"
    local failed_hosts=()
    
    for host in "${test_hosts[@]}"; do
        if ! ping -c 1 -W "$timeout" "$host" &>/dev/null; then
            failed_hosts+=("$host")
        fi
    done
    
    if [[ ${#failed_hosts[@]} -gt 0 ]]; then
        log_error "Network connectivity issues detected. Failed hosts: ${failed_hosts[*]}"
        return 1
    fi
    
    log_success "Network connectivity validated"
    return 0
}

# システムリソース検証
validate_system_resources() {
    local min_memory_gb="${1:-1}"
    local min_disk_gb="${2:-5}"
    local max_cpu_percent="${3:-90}"
    
    # メモリチェック
    if ! check_memory_available "$min_memory_gb"; then
        log_error "Insufficient memory available (required: ${min_memory_gb}GB)"
        return 1
    fi
    
    # ディスク容量チェック
    if ! check_disk_space "." "$min_disk_gb"; then
        log_error "Insufficient disk space available (required: ${min_disk_gb}GB)"
        return 1
    fi
    
    # CPU使用率チェック
    if command -v top &> /dev/null; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1)
        if (( $(echo "$cpu_usage > $max_cpu_percent" | bc -l) )); then
            log_warning "High CPU usage detected: ${cpu_usage}%"
        fi
    fi
    
    log_success "System resources validated"
    return 0
}

# 総合検証レポート
generate_validation_report() {
    local report_file="${1:-validation_report_$(date +%Y%m%d_%H%M%S).txt}"
    
    {
        echo "Deployment Validation Report"
        echo "Generated: $(date)"
        echo "Environment: ${ENVIRONMENT:-unknown}"
        echo "=================================="
        echo ""
        
        echo "System Information:"
        echo "  OS: $(uname -s -r)"
        echo "  Architecture: $(uname -m)"
        echo "  Hostname: $(hostname)"
        echo "  User: $(whoami)"
        echo ""
        
        echo "Resource Status:"
        if command -v free &> /dev/null; then
            echo "  Memory: $(free -h | grep "Mem:" | awk '{print $3 "/" $2 " (" $5 ")"}')"
        fi
        
        if command -v df &> /dev/null; then
            echo "  Disk: $(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"
        fi
        echo ""
        
        echo "Validation Results:"
        # ここに個別の検証結果を追加
        
    } > "$report_file"
    
    log_info "Validation report generated: $report_file"
}

# クリティカル検証（失敗時即座に終了）
critical_validation() {
    local validation_name="$1"
    shift
    local validation_command=("$@")
    
    log_info "Running critical validation: $validation_name"
    
    if ! "${validation_command[@]}"; then
        log_error "Critical validation failed: $validation_name"
        log_error "Deployment cannot continue"
        exit 1
    fi
    
    log_success "Critical validation passed: $validation_name"
}

# 警告レベル検証（失敗時警告のみ）
warning_validation() {
    local validation_name="$1"
    shift
    local validation_command=("$@")
    
    log_info "Running warning validation: $validation_name"
    
    if ! "${validation_command[@]}"; then
        log_warning "Validation warning: $validation_name"
        return 1
    fi
    
    log_success "Validation passed: $validation_name"
    return 0
}