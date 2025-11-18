#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

ENV_FILE=${FANZA_BACKFILL_ENV_FILE:-docker/env/dev.env}
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[fanza_backfill] Environment file not found: $ENV_FILE" >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[fanza_backfill] Node.js is required to run this script." >&2
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/node_modules" ]]; then
  echo "[fanza_backfill] Installing dependencies..."
  npm install --prefix "$SCRIPT_DIR" >/dev/null
fi

FANZA_BACKFILL_ENV_FILE="$ENV_FILE" \
DOTENV_CONFIG_OVERRIDE=${DOTENV_CONFIG_OVERRIDE:-true} \
node "$SCRIPT_DIR/index.js" "$@"
