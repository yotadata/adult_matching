#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-gen-video-embeddings:latest

docker build -f scripts/gen_video_embeddings/Dockerfile -t "$IMAGE" .

DOCKER_FLAGS=(--rm)
if [ -t 1 ]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi

ENV_FILE=${GEN_VIDEO_ENV_FILE:-docker/env/dev.env}
NETWORK=${GEN_VIDEO_NETWORK:-}
RUN_ID=${GEN_VIDEO_RUN_ID:-$(
  python - <<'PY'
from datetime import datetime, timezone, timedelta
JST = timezone(timedelta(hours=9))
print(datetime.now(JST).strftime("%Y%m%d_%H%M%S"))
PY
)}
DEFAULT_OUTPUT_DIR="ml/artifacts/live/video_embeddings/${RUN_ID}"
OUTPUT_DIR=${GEN_VIDEO_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
OUTPUT_NAME="video_embeddings.parquet"
DRY_RUN=false
SKIP_UPSERT=false

PY_ARGS=(--output-dir "$OUTPUT_DIR" --output-name "$OUTPUT_NAME")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      PY_ARGS+=(--output-dir "$2")
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      PY_ARGS+=(--output-name "$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      PY_ARGS+=(--dry-run)
      shift
      ;;
    --skip-upsert)
      SKIP_UPSERT=true
      shift
      ;;
    --network)
      NETWORK="$2"
      shift 2
      ;;
    --)
      shift
      PY_ARGS+=("$@")
      break
      ;;
    *)
      PY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  echo "Set GEN_VIDEO_ENV_FILE or pass --env-file <path> to a valid env file." >&2
  exit 1
fi

# Source env file to resolve nested references
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

if [[ -n "${REMOTE_DATABASE_URL:-}" ]]; then
  ipv4_url=$(python - <<'PY2'
import os, socket, urllib.parse
url = os.environ.get('REMOTE_DATABASE_URL')
if not url:
    raise SystemExit
parsed = urllib.parse.urlsplit(url)
host = parsed.hostname
if not host:
    raise SystemExit
ipv4_addr = None
try:
    for family, _, _, _, sockaddr in socket.getaddrinfo(host, parsed.port or 5432):
        if family == socket.AF_INET:
            ipv4_addr = sockaddr[0]
            break
except socket.gaierror:
    ipv4_addr = None
if not ipv4_addr:
    raise SystemExit
query = parsed.query
if query:
    query += f'&hostaddr={ipv4_addr}'
else:
    query = f'hostaddr={ipv4_addr}'
replacement = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))
print(replacement)
PY2
  ) || true
  if [[ -n "${ipv4_url:-}" ]]; then
    export REMOTE_DATABASE_URL="$ipv4_url"
    export SUPABASE_DB_URL="${SUPABASE_DB_URL:-$ipv4_url}"
  fi
fi

TMP_ENV_FILE=$(mktemp)
trap 'rm -f "$TMP_ENV_FILE"' EXIT

env_keys=$(grep -E "^[A-Za-z_][A-Za-z0-9_]*=" "$ENV_FILE" | cut -d= -f1)
printed=""
while IFS= read -r key; do
  [[ -z "$key" ]] && continue
  case " $printed " in
    *" $key "*) continue ;;
  esac
  printed+=" $key"
  if [[ -n ${!key+x} ]]; then
    printf "%s=%s\n" "$key" "${!key}" >>"$TMP_ENV_FILE"
  fi
done <<<"$env_keys"

mkdir -p "$OUTPUT_DIR"

DOCKER_CMD=(docker run "${DOCKER_FLAGS[@]}")
if [[ -n "$NETWORK" ]]; then
  DOCKER_CMD+=(--network "$NETWORK")
fi
DOCKER_CMD+=(--add-host host.docker.internal:host-gateway --env-file "$TMP_ENV_FILE" -v "$(pwd)":/workspace -w /workspace "$IMAGE" python scripts/gen_video_embeddings/gen_video_embeddings.py)
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  DOCKER_CMD+=("${PY_ARGS[@]}")
fi

"${DOCKER_CMD[@]}"
status=$?
if [[ $status -ne 0 ]]; then
  exit $status
fi

if [[ "$DRY_RUN" == "true" || "$SKIP_UPSERT" == "true" ]]; then
  exit 0
fi

# Ensure model_meta is available in output dir for upsert script
META_SRC="ml/artifacts/latest/model_meta.json"
if [[ -f "$META_SRC" ]]; then
  cp "$META_SRC" "$OUTPUT_DIR/model_meta.json"
fi

UPSERT_ARGS=(--env-file "$ENV_FILE" --artifacts-dir "$OUTPUT_DIR")
if [[ -n "$NETWORK" ]]; then
  UPSERT_TT_NETWORK="$NETWORK" bash scripts/upsert_two_tower/run.sh "${UPSERT_ARGS[@]}"
else
  bash scripts/upsert_two_tower/run.sh "${UPSERT_ARGS[@]}"
fi
