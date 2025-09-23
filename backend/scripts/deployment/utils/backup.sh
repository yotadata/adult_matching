#!/bin/bash

# =============================================================================
# Backup Utilities for Deployment Scripts
# 
# デプロイメントスクリプト用バックアップユーティリティ
# =============================================================================

# バックアップ設定
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/tmp/backups}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"

# =============================================================================
# データベースバックアップ機能
# =============================================================================

create_db_backup() {
    local backup_id="$1"
    local backup_file="$BACKUP_BASE_DIR/db_backup_${backup_id}.sql"
    local compressed_file="${backup_file}.gz"
    
    log_info "Creating database backup: $backup_id"
    
    # バックアップディレクトリ作成
    mkdir -p "$BACKUP_BASE_DIR"
    
    # データベースURL取得
    local db_url="${DATABASE_URL:-}"
    if [[ -z "$db_url" ]]; then
        log_error "DATABASE_URL not set"
        return 1
    fi
    
    # pg_dump実行
    if command -v pg_dump &> /dev/null; then
        log_info "Running pg_dump..."
        
        if pg_dump "$db_url" \
            --verbose \
            --clean \
            --if-exists \
            --create \
            --format=plain \
            --no-owner \
            --no-privileges \
            > "$backup_file" 2> "${backup_file}.log"; then
            
            # バックアップファイル圧縮
            log_info "Compressing backup file..."
            if gzip -$COMPRESSION_LEVEL "$backup_file"; then
                log_success "Database backup created: $compressed_file"
                
                # バックアップメタデータ作成
                create_backup_metadata "$backup_id" "database" "$compressed_file"
                
                return 0
            else
                log_error "Failed to compress backup file"
                return 1
            fi
        else
            log_error "pg_dump failed"
            if [[ -f "${backup_file}.log" ]]; then
                log_error "pg_dump error log:"
                cat "${backup_file}.log" | while read -r line; do
                    log_error "  $line"
                done
            fi
            return 1
        fi
    else
        log_error "pg_dump not available"
        return 1
    fi
}

# スキーマのみバックアップ
create_schema_backup() {
    local backup_id="$1"
    local backup_file="$BACKUP_BASE_DIR/schema_backup_${backup_id}.sql"
    
    log_info "Creating schema backup: $backup_id"
    
    mkdir -p "$BACKUP_BASE_DIR"
    
    local db_url="${DATABASE_URL:-}"
    if [[ -z "$db_url" ]]; then
        log_error "DATABASE_URL not set"
        return 1
    fi
    
    if pg_dump "$db_url" \
        --schema-only \
        --verbose \
        --clean \
        --if-exists \
        --create \
        --no-owner \
        --no-privileges \
        > "$backup_file" 2> "${backup_file}.log"; then
        
        gzip -$COMPRESSION_LEVEL "$backup_file"
        log_success "Schema backup created: ${backup_file}.gz"
        
        create_backup_metadata "$backup_id" "schema" "${backup_file}.gz"
        return 0
    else
        log_error "Schema backup failed"
        return 1
    fi
}

# データのみバックアップ
create_data_backup() {
    local backup_id="$1"
    local tables="${2:-all}"
    local backup_file="$BACKUP_BASE_DIR/data_backup_${backup_id}.sql"
    
    log_info "Creating data backup: $backup_id"
    
    mkdir -p "$BACKUP_BASE_DIR"
    
    local db_url="${DATABASE_URL:-}"
    if [[ -z "$db_url" ]]; then
        log_error "DATABASE_URL not set"
        return 1
    fi
    
    local pg_dump_args=(
        "$db_url"
        --data-only
        --verbose
        --no-owner
        --no-privileges
        --insert
    )
    
    # 特定テーブルのみバックアップの場合
    if [[ "$tables" != "all" ]]; then
        IFS=',' read -ra table_array <<< "$tables"
        for table in "${table_array[@]}"; do
            pg_dump_args+=(--table="$table")
        done
    fi
    
    if "${pg_dump_args[@]}" > "$backup_file" 2> "${backup_file}.log"; then
        gzip -$COMPRESSION_LEVEL "$backup_file"
        log_success "Data backup created: ${backup_file}.gz"
        
        create_backup_metadata "$backup_id" "data" "${backup_file}.gz"
        return 0
    else
        log_error "Data backup failed"
        return 1
    fi
}

# =============================================================================
# ファイルシステムバックアップ機能
# =============================================================================

create_files_backup() {
    local backup_id="$1"
    local source_path="$2"
    local backup_name="${3:-files}"
    local backup_file="$BACKUP_BASE_DIR/${backup_name}_backup_${backup_id}.tar.gz"
    
    log_info "Creating files backup: $backup_id ($source_path)"
    
    mkdir -p "$BACKUP_BASE_DIR"
    
    if [[ ! -e "$source_path" ]]; then
        log_error "Source path does not exist: $source_path"
        return 1
    fi
    
    # tar実行
    local tar_args=(
        -czf "$backup_file"
        -C "$(dirname "$source_path")"
        --exclude='*.log'
        --exclude='*.tmp'
        --exclude='.git'
        --exclude='node_modules'
        --exclude='__pycache__'
        --exclude='.pytest_cache'
        "$(basename "$source_path")"
    )
    
    if tar "${tar_args[@]}" 2> "${backup_file}.log"; then
        log_success "Files backup created: $backup_file"
        
        create_backup_metadata "$backup_id" "$backup_name" "$backup_file"
        return 0
    else
        log_error "Files backup failed"
        if [[ -f "${backup_file}.log" ]]; then
            log_error "tar error log:"
            cat "${backup_file}.log" | while read -r line; do
                log_error "  $line"
            done
        fi
        return 1
    fi
}

# 設定ファイルバックアップ
create_config_backup() {
    local backup_id="$1"
    local config_files=("$@")
    local backup_dir="$BACKUP_BASE_DIR/config_backup_${backup_id}"
    local backup_file="${backup_dir}.tar.gz"
    
    log_info "Creating configuration backup: $backup_id"
    
    mkdir -p "$backup_dir"
    
    # 設定ファイルコピー
    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            local filename=$(basename "$config_file")
            cp "$config_file" "$backup_dir/$filename"
            log_debug "Backed up config: $config_file"
        else
            log_warning "Config file not found: $config_file"
        fi
    done
    
    # アーカイブ作成
    if tar -czf "$backup_file" -C "$BACKUP_BASE_DIR" "$(basename "$backup_dir")"; then
        rm -rf "$backup_dir"
        log_success "Configuration backup created: $backup_file"
        
        create_backup_metadata "$backup_id" "config" "$backup_file"
        return 0
    else
        log_error "Configuration backup failed"
        return 1
    fi
}

# =============================================================================
# バックアップメタデータ管理
# =============================================================================

create_backup_metadata() {
    local backup_id="$1"
    local backup_type="$2"
    local backup_file="$3"
    local metadata_file="$BACKUP_BASE_DIR/metadata_${backup_id}.json"
    
    local file_size="0"
    local file_hash=""
    
    if [[ -f "$backup_file" ]]; then
        file_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null || echo "0")
        if command -v sha256sum &> /dev/null; then
            file_hash=$(sha256sum "$backup_file" | cut -d' ' -f1)
        elif command -v shasum &> /dev/null; then
            file_hash=$(shasum -a 256 "$backup_file" | cut -d' ' -f1)
        fi
    fi
    
    cat > "$metadata_file" << EOF
{
  "backup_id": "$backup_id",
  "backup_type": "$backup_type",
  "created_at": "$(date -Iseconds)",
  "environment": "${ENVIRONMENT:-unknown}",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo "unknown")",
  "backup_file": "$backup_file",
  "file_size_bytes": $file_size,
  "file_hash_sha256": "$file_hash",
  "creator": "$(whoami)",
  "hostname": "$(hostname)",
  "deployment_id": "${DEPLOYMENT_ID:-unknown}"
}
EOF
    
    log_debug "Backup metadata created: $metadata_file"
}

# バックアップ一覧表示
list_backups() {
    local backup_type="${1:-all}"
    local limit="${2:-10}"
    
    log_info "Listing backups (type: $backup_type, limit: $limit)"
    
    if [[ ! -d "$BACKUP_BASE_DIR" ]]; then
        log_warning "Backup directory does not exist: $BACKUP_BASE_DIR"
        return 0
    fi
    
    local metadata_files=($(find "$BACKUP_BASE_DIR" -name "metadata_*.json" -type f | sort -r | head -n "$limit"))
    
    if [[ ${#metadata_files[@]} -eq 0 ]]; then
        log_info "No backups found"
        return 0
    fi
    
    printf "%-20s %-12s %-20s %-10s %-15s\n" "BACKUP_ID" "TYPE" "CREATED" "SIZE" "STATUS"
    printf "%-20s %-12s %-20s %-10s %-15s\n" "--------" "----" "-------" "----" "------"
    
    for metadata_file in "${metadata_files[@]}"; do
        if [[ -f "$metadata_file" ]]; then
            local backup_id=$(jq -r '.backup_id' "$metadata_file" 2>/dev/null || echo "unknown")
            local type=$(jq -r '.backup_type' "$metadata_file" 2>/dev/null || echo "unknown")
            local created=$(jq -r '.created_at' "$metadata_file" 2>/dev/null | cut -d'T' -f1)
            local size_bytes=$(jq -r '.file_size_bytes' "$metadata_file" 2>/dev/null || echo "0")
            local backup_file=$(jq -r '.backup_file' "$metadata_file" 2>/dev/null || echo "")
            
            # サイズを人間可読形式に変換
            local size_human=""
            if command -v numfmt &> /dev/null; then
                size_human=$(numfmt --to=iec-i --suffix=B --padding=7 "$size_bytes")
            else
                size_human="${size_bytes}B"
            fi
            
            # ファイル存在チェック
            local status="MISSING"
            if [[ -f "$backup_file" ]]; then
                status="OK"
            fi
            
            # タイプフィルタ
            if [[ "$backup_type" == "all" || "$type" == "$backup_type" ]]; then
                printf "%-20s %-12s %-20s %-10s %-15s\n" \
                    "${backup_id:0:20}" \
                    "$type" \
                    "$created" \
                    "$size_human" \
                    "$status"
            fi
        fi
    done
}

# バックアップ詳細表示
show_backup_details() {
    local backup_id="$1"
    local metadata_file="$BACKUP_BASE_DIR/metadata_${backup_id}.json"
    
    if [[ ! -f "$metadata_file" ]]; then
        log_error "Backup metadata not found: $backup_id"
        return 1
    fi
    
    log_info "Backup Details: $backup_id"
    echo "=================================="
    
    # JSON内容を整形して表示
    if command -v jq &> /dev/null; then
        jq . "$metadata_file"
    else
        cat "$metadata_file"
    fi
    
    # バックアップファイル存在確認
    local backup_file=$(jq -r '.backup_file' "$metadata_file" 2>/dev/null)
    if [[ -f "$backup_file" ]]; then
        echo ""
        echo "File Status: EXISTS"
        echo "File Path: $backup_file"
        
        # ファイルハッシュ検証
        local stored_hash=$(jq -r '.file_hash_sha256' "$metadata_file" 2>/dev/null)
        if [[ -n "$stored_hash" && "$stored_hash" != "null" ]]; then
            local current_hash=""
            if command -v sha256sum &> /dev/null; then
                current_hash=$(sha256sum "$backup_file" | cut -d' ' -f1)
            elif command -v shasum &> /dev/null; then
                current_hash=$(shasum -a 256 "$backup_file" | cut -d' ' -f1)
            fi
            
            if [[ "$stored_hash" == "$current_hash" ]]; then
                echo "Integrity: VERIFIED"
            else
                echo "Integrity: CORRUPTED"
                log_warning "Backup file integrity check failed"
            fi
        fi
    else
        echo ""
        echo "File Status: MISSING"
        log_warning "Backup file not found: $backup_file"
    fi
}

# =============================================================================
# バックアップ復元機能
# =============================================================================

restore_database_backup() {
    local backup_id="$1"
    local target_db_url="${2:-$DATABASE_URL}"
    local backup_file="$BACKUP_BASE_DIR/db_backup_${backup_id}.sql.gz"
    
    log_info "Restoring database backup: $backup_id"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [[ -z "$target_db_url" ]]; then
        log_error "Target database URL not provided"
        return 1
    fi
    
    # 確認プロンプト
    echo "WARNING: This will overwrite the target database!"
    echo "Target: $target_db_url"
    echo "Backup: $backup_file"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        log_info "Database restoration cancelled"
        return 1
    fi
    
    # バックアップファイル展開・復元
    local temp_file="/tmp/restore_${backup_id}_$$.sql"
    
    if gunzip -c "$backup_file" > "$temp_file"; then
        log_info "Restoring database from backup..."
        
        if psql "$target_db_url" -f "$temp_file" > "/tmp/restore_${backup_id}.log" 2>&1; then
            rm -f "$temp_file"
            log_success "Database restoration completed"
            return 0
        else
            log_error "Database restoration failed"
            log_error "Check log file: /tmp/restore_${backup_id}.log"
            rm -f "$temp_file"
            return 1
        fi
    else
        log_error "Failed to extract backup file"
        return 1
    fi
}

restore_files_backup() {
    local backup_id="$1"
    local backup_name="${2:-files}"
    local target_path="$3"
    local backup_file="$BACKUP_BASE_DIR/${backup_name}_backup_${backup_id}.tar.gz"
    
    log_info "Restoring files backup: $backup_id"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [[ -z "$target_path" ]]; then
        log_error "Target path not provided"
        return 1
    fi
    
    # 確認プロンプト
    echo "WARNING: This will overwrite files in the target directory!"
    echo "Target: $target_path"
    echo "Backup: $backup_file"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        log_info "Files restoration cancelled"
        return 1
    fi
    
    # ディレクトリ作成
    mkdir -p "$target_path"
    
    # バックアップ展開
    if tar -xzf "$backup_file" -C "$target_path" --strip-components=1; then
        log_success "Files restoration completed"
        return 0
    else
        log_error "Files restoration failed"
        return 1
    fi
}

# =============================================================================
# バックアップメンテナンス
# =============================================================================

cleanup_old_backups() {
    local retention_days="${1:-$BACKUP_RETENTION_DAYS}"
    
    log_info "Cleaning up backups older than $retention_days days"
    
    if [[ ! -d "$BACKUP_BASE_DIR" ]]; then
        log_warning "Backup directory does not exist: $BACKUP_BASE_DIR"
        return 0
    fi
    
    local deleted_count=0
    
    # 古いバックアップファイル削除
    while IFS= read -r -d '' file; do
        rm -f "$file"
        ((deleted_count++))
        log_debug "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_BASE_DIR" -type f -mtime +$retention_days -print0 2>/dev/null)
    
    log_info "Cleanup completed: $deleted_count files deleted"
}

verify_backup_integrity() {
    local backup_id="$1"
    local metadata_file="$BACKUP_BASE_DIR/metadata_${backup_id}.json"
    
    log_info "Verifying backup integrity: $backup_id"
    
    if [[ ! -f "$metadata_file" ]]; then
        log_error "Backup metadata not found: $backup_id"
        return 1
    fi
    
    local backup_file=$(jq -r '.backup_file' "$metadata_file" 2>/dev/null)
    local stored_hash=$(jq -r '.file_hash_sha256' "$metadata_file" 2>/dev/null)
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [[ -z "$stored_hash" || "$stored_hash" == "null" ]]; then
        log_warning "No hash stored for backup verification"
        return 1
    fi
    
    # ハッシュ計算
    local current_hash=""
    if command -v sha256sum &> /dev/null; then
        current_hash=$(sha256sum "$backup_file" | cut -d' ' -f1)
    elif command -v shasum &> /dev/null; then
        current_hash=$(shasum -a 256 "$backup_file" | cut -d' ' -f1)
    else
        log_warning "No hash command available for verification"
        return 1
    fi
    
    if [[ "$stored_hash" == "$current_hash" ]]; then
        log_success "Backup integrity verified: $backup_id"
        return 0
    else
        log_error "Backup integrity check failed: $backup_id"
        log_error "Stored hash:  $stored_hash"
        log_error "Current hash: $current_hash"
        return 1
    fi
}

# バックアップサイズ統計
show_backup_statistics() {
    log_info "Backup Statistics"
    echo "=================================="
    
    if [[ ! -d "$BACKUP_BASE_DIR" ]]; then
        echo "No backup directory found"
        return 0
    fi
    
    # 基本統計
    local total_files=$(find "$BACKUP_BASE_DIR" -type f -name "*.gz" -o -name "*.sql" | wc -l)
    local total_size=0
    
    if command -v du &> /dev/null; then
        total_size=$(du -sb "$BACKUP_BASE_DIR" 2>/dev/null | cut -f1)
    fi
    
    echo "Total backup files: $total_files"
    
    if command -v numfmt &> /dev/null; then
        echo "Total size: $(numfmt --to=iec-i --suffix=B "$total_size")"
    else
        echo "Total size: ${total_size} bytes"
    fi
    
    # タイプ別統計
    echo ""
    echo "Backup types:"
    find "$BACKUP_BASE_DIR" -name "metadata_*.json" -type f -exec jq -r '.backup_type' {} \; 2>/dev/null | sort | uniq -c | while read -r count type; do
        echo "  $type: $count backups"
    done
    
    # 最新・最古バックアップ
    local newest=$(find "$BACKUP_BASE_DIR" -name "metadata_*.json" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    local oldest=$(find "$BACKUP_BASE_DIR" -name "metadata_*.json" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)
    
    if [[ -n "$newest" ]]; then
        local newest_id=$(basename "$newest" | sed 's/metadata_\(.*\)\.json/\1/')
        local newest_date=$(jq -r '.created_at' "$newest" 2>/dev/null | cut -d'T' -f1)
        echo ""
        echo "Newest backup: $newest_id ($newest_date)"
    fi
    
    if [[ -n "$oldest" ]]; then
        local oldest_id=$(basename "$oldest" | sed 's/metadata_\(.*\)\.json/\1/')
        local oldest_date=$(jq -r '.created_at' "$oldest" 2>/dev/null | cut -d'T' -f1)
        echo "Oldest backup: $oldest_id ($oldest_date)"
    fi
}