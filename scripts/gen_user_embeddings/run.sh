#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-gen-user-embeddings:latest

docker build -f scripts/gen_user_embeddings/Dockerfile -t "$IMAGE" .

DOCKER_FLAGS=(--rm)
if [ -t 1 ]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi

ENV_FILE=${GEN_USER_ENV_FILE:-docker/env/dev.env}
PY_ARGS=()
OUTPUT_DIR="ml/artifacts/live"
OUTPUT_NAME="user_embeddings.parquet"
MIN_INTERACTIONS=0
DRY_RUN=false
SKIP_UPSERT=false
NETWORK=${GEN_USER_NETWORK:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      PY_ARGS+=("$1" "$2")
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      PY_ARGS+=("$1" "$2")
      shift 2
      ;;
    --min-interactions)
      MIN_INTERACTIONS="$2"
      PY_ARGS+=("$1" "$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      PY_ARGS+=("$1")
      shift
      ;;
    --no-copy-video-embeddings)
      PY_ARGS+=("$1")
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
  echo "Set GEN_USER_ENV_FILE or pass --env-file <path> to a valid env file." >&2
  exit 1
fi

# Source env file to resolve nested references
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

# Best effort: append hostaddr to remote URLs to avoid IPv6-only DNS
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

# Persist resolved variables to a temporary env file for docker
TMP_ENV_FILE=$(mktemp)
trap 'rm -f "$TMP_ENV_FILE"' EXIT

env_keys=$(grep -E "^[A-Za-z_][A-Za-z0-9_]*=" "$ENV_FILE" | cut -d= -f1)
filtered_keys=""
while IFS= read -r key; do
  [[ -z "$key" ]] && continue
  case " $filtered_keys " in
    *" $key "*) continue;;
  esac
  filtered_keys+=" $key"
  if [[ -n ${!key+x} ]]; then
    printf "%s=%s\n" "$key" "${!key}" >> "$TMP_ENV_FILE"
  fi
done <<< "$env_keys"

DOCKER_CMD=(docker run "${DOCKER_FLAGS[@]}")
if [[ -n "$NETWORK" ]]; then
  DOCKER_CMD+=(--network "$NETWORK")
fi
DOCKER_CMD+=(--add-host host.docker.internal:host-gateway --env-file "$TMP_ENV_FILE" -v "$(pwd)":/workspace -w /workspace "$IMAGE" python scripts/gen_user_embeddings/gen_user_embeddings.py)
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

UPSERT_ARGS=(--env-file "$ENV_FILE" --artifacts-dir "$OUTPUT_DIR" --include-users)

# Ensure model_meta.json is available alongside generated embeddings (required by upsert)
META_SRC="ml/artifacts/latest/model_meta.json"
META_DST="$OUTPUT_DIR/model_meta.json"
if [[ -f "$META_SRC" && ! -f "$META_DST" ]]; then
  mkdir -p "$OUTPUT_DIR"
  cp "$META_SRC" "$META_DST"
fi

# When反映済みのユーザー埋め込みを upsert するときは、ここでフィルタリングせずに全件反映する
UPSERT_ARGS+=(--min-user-interactions "$MIN_INTERACTIONS")

if [[ -n "$NETWORK" ]]; then
  UPSERT_TT_NETWORK="$NETWORK" bash scripts/upsert_two_tower/run.sh "${UPSERT_ARGS[@]}"
else
  bash scripts/upsert_two_tower/run.sh "${UPSERT_ARGS[@]}"
fi
