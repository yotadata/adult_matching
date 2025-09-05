#!/bin/bash

# Batch Embedding Update Runner Script
# This script is designed to be run as a cron job

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_DIR}/logs/batch_update_$(date +%Y%m%d_%H%M%S).log"
VENV_PATH="${PROJECT_DIR}/.venv"
MODELS_PATH="${PROJECT_DIR}/models"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Database connection
DB_URL="${DATABASE_URL:-postgresql://postgres:postgres@127.0.0.1:54322/postgres}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    error_exit "Virtual environment not found at $VENV_PATH"
fi

# Check if models exist
if [ ! -d "$MODELS_PATH" ]; then
    error_exit "Models directory not found at $MODELS_PATH"
fi

log "Starting batch embedding update..."
log "Models path: $MODELS_PATH"
log "Log file: $LOG_FILE"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check if required Python packages are installed
python3 -c "import tensorflow, psycopg2" 2>/dev/null || {
    log "Installing required packages..."
    pip install -r "$SCRIPT_DIR/requirements.txt" >> "$LOG_FILE" 2>&1
}

# Run batch update
log "Running batch update..."
python3 "$SCRIPT_DIR/batch_embedding_update.py" \
    --db-url "$DB_URL" \
    --model-path "$MODELS_PATH" \
    --batch-size 500 \
    >> "$LOG_FILE" 2>&1

BATCH_EXIT_CODE=$?

if [ $BATCH_EXIT_CODE -eq 0 ]; then
    log "Batch update completed successfully"
else
    error_exit "Batch update failed with exit code $BATCH_EXIT_CODE"
fi

# Cleanup old log files (keep last 7 days)
find "$(dirname "$LOG_FILE")" -name "batch_update_*.log" -mtime +7 -delete 2>/dev/null || true

log "Batch update process finished"