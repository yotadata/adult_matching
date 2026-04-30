#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

ENV_FILE=${AUDIT_MAKER_ENV_FILE:-docker/env/prd.env}
WINDOW_DAYS=${AUDIT_MAKER_WINDOW_DAYS:-30}
THRESHOLD=${AUDIT_MAKER_THRESHOLD:-0.20}
LOG_CSV=${AUDIT_MAKER_LOG_CSV:-docs/ml/maker_bias_log.csv}
SUMMARY_JSON=${AUDIT_MAKER_SUMMARY_JSON:-}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[audit-maker] env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

DB_URL="${REMOTE_DATABASE_URL:-}"
if [[ -z "$DB_URL" ]]; then
  echo "[audit-maker] REMOTE_DATABASE_URL が未設定" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$SUMMARY_JSON" ]]; then
  EXTRA_ARGS+=(--summary-json "$SUMMARY_JSON")
fi

cd "$REPO_ROOT"

docker build -q -t audit-maker-bias "$SCRIPT_DIR"

docker run --rm \
  -e REMOTE_DATABASE_URL="$DB_URL" \
  -v "$REPO_ROOT/docs/ml:/app/docs/ml" \
  audit-maker-bias \
  --db-url "$DB_URL" \
  --window-days "$WINDOW_DAYS" \
  --threshold "$THRESHOLD" \
  --log-csv "docs/ml/$LOG_CSV" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
  || EXIT_CODE=$?

echo "[audit-maker] 完了 (exit=${EXIT_CODE:-0})"
# exit 2 = 偏り検出（警告のみ、CI は失敗させない）
exit 0
