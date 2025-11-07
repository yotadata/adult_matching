#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

ENV_FILE=${SYNC_USER_ENV_FILE:-docker/env/prd.env}
RECENT_HOURS=${SYNC_USER_RECENT_HOURS:-1}
OUTPUT_ROOT=${SYNC_USER_OUTPUT_ROOT:-ml/artifacts/live/user_embedding_syncs}
RUN_ID=${SYNC_USER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
FETCH_LATEST=false
DRY_RUN=false
GEN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --recent-hours)
      RECENT_HOURS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --fetch-latest)
      FETCH_LATEST=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --)
      shift
      GEN_ARGS+=("$@")
      break
      ;;
    *)
      GEN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  exit 1
fi

OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$OUTPUT_DIR"

echo "[sync-user] env: $ENV_FILE"
if [[ "$FETCH_LATEST" == "true" ]]; then
  echo "[sync-user] fetching latest model artifacts from Storage"
  bash "$REPO_ROOT/scripts/publish_two_tower/run.sh" --env-file "$ENV_FILE" fetch --dest ml/artifacts/latest
else
  echo "[sync-user] using existing ml/artifacts contents (no fetch)"
fi

echo "[sync-user] generating user embeddings (recent_hours=$RECENT_HOURS)"
GEN_CMD=(bash "$REPO_ROOT/scripts/gen_user_embeddings/run.sh" --include-existing --recent-hours "$RECENT_HOURS" --output-dir "$OUTPUT_DIR" --output-name user_embeddings.parquet --skip-upsert)
if [[ "$DRY_RUN" == "true" ]]; then
  GEN_CMD+=(--dry-run)
fi
if [[ ${#GEN_ARGS[@]} -gt 0 ]]; then
  GEN_CMD+=("${GEN_ARGS[@]}")
fi
GEN_VIDEO_ENV_FILE="$ENV_FILE" GEN_VIDEO_OUTPUT_DIR="$OUTPUT_DIR" "${GEN_CMD[@]}"

echo "[sync-user] generator finished"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[sync-user] dry-run requested; skipping upsert"
  exit 0
fi

echo "[sync-user] upserting user embeddings"
UPDATE_USER_ENV_FILE="$ENV_FILE" bash "$REPO_ROOT/scripts/update_user_embeddings/run.sh" --env-file "$ENV_FILE" --artifacts-dir "$OUTPUT_DIR"

echo "[sync-user] completed. artifacts: $OUTPUT_DIR"
