#!/usr/bin/env bash
set -euo pipefail

LOCAL_ENV=${LOCAL_BACKFILL_ENV:-docker/env/dev.env}
REMOTE_ENV=${REMOTE_BACKFILL_ENV:-docker/env/prd.env}
RUN_SCRIPT="$(cd "$(dirname "$0")" && pwd)/run.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[backfill_all] Missing run.sh helper at $RUN_SCRIPT" >&2
  exit 1
fi

function run_backfill() {
  local label="$1"
  local env_file="$2"

  if [[ ! -f "$env_file" ]]; then
    echo "[backfill_all] Skip ${label}: env file not found (${env_file})" >&2
    return 1
  fi

  echo "[backfill_all] >>> ${label} (${env_file})"
  FANZA_BACKFILL_ENV_FILE="$env_file" \
    bash "$RUN_SCRIPT" "$@"
  echo "[backfill_all] <<< ${label} done"
}

ARGS=("$@")

run_backfill "local" "$LOCAL_ENV" "${ARGS[@]}"
run_backfill "remote" "$REMOTE_ENV" "${ARGS[@]}"
