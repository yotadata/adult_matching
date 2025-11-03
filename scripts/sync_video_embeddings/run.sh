#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

ENV_FILE=${SYNC_VIDEO_ENV_FILE:-docker/env/prd.env}
LOOKBACK_DAYS=${SYNC_VIDEO_LOOKBACK_DAYS:-3}
SKIP_INGEST=false
SKIP_FETCH=false
FORWARDED=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --lookback-days)
      LOOKBACK_DAYS="$2"
      shift 2
      ;;
    --skip-ingest)
      SKIP_INGEST=true
      shift
      ;;
    --skip-fetch)
      SKIP_FETCH=true
      shift
      ;;
    --)
      shift
      FORWARDED+=("$@")
      break
      ;;
    *)
      FORWARDED+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  exit 1
fi

RUN_ID=${GEN_VIDEO_RUN_ID:-$(
  python - <<'PY'
from datetime import datetime, timezone, timedelta
JST = timezone(timedelta(hours=9))
print(datetime.now(JST).strftime("%Y%m%d_%H%M%S"))
PY
)}
OUTPUT_ROOT=${SYNC_VIDEO_OUTPUT_ROOT:-ml/artifacts/live/video_embedding_syncs}
GEN_VIDEO_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"

if [[ -n "${GTE_RELEASE_DATE:-}" ]]; then
  GTE_DATE="$GTE_RELEASE_DATE"
else
  GTE_DATE=$(python - <<PY
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=9))
days = int("$LOOKBACK_DAYS")
today = datetime.now(JST).date()
start = today - timedelta(days=days)
print(start.isoformat())
PY
)
fi
if [[ -n "${LTE_RELEASE_DATE:-}" ]]; then
  LTE_DATE="$LTE_RELEASE_DATE"
else
  LTE_DATE=$(python - <<PY
from datetime import datetime, timezone, timedelta
JST = timezone(timedelta(hours=9))
today = datetime.now(JST).date()
print(today.isoformat())
PY
)
fi

echo "[sync-video] using env file: $ENV_FILE"
echo "[sync-video] ingest window: ${GTE_DATE} → ${LTE_DATE} (JST)"

if [[ "$SKIP_FETCH" == "false" ]]; then
  bash "$REPO_ROOT/scripts/publish_two_tower/run.sh" --env-file "$ENV_FILE" fetch --dest ml/artifacts/latest
fi

if [[ "$SKIP_INGEST" == "false" ]]; then
  INGEST_FANZA_ENV_FILE="$ENV_FILE" \
  GTE_RELEASE_DATE="$GTE_DATE" \
  LTE_RELEASE_DATE="$LTE_DATE" \
  bash "$REPO_ROOT/scripts/ingest_fanza/run.sh"
fi

GEN_VIDEO_ENV_FILE="$ENV_FILE" \
GEN_VIDEO_OUTPUT_DIR="$GEN_VIDEO_OUTPUT_DIR" \
bash "$REPO_ROOT/scripts/gen_video_embeddings/run.sh" "${FORWARDED[@]}"

echo "[sync-video] completed. embeddings written to $GEN_VIDEO_OUTPUT_DIR"
