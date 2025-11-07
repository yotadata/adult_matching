#!/usr/bin/env bash
set -euo pipefail

IMAGE=${UPDATE_USER_IMAGE:-adult-matching-update-user-embeddings:latest}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

DOCKERFILE="$SCRIPT_DIR/Dockerfile"
if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Dockerfile not found: $DOCKERFILE" >&2
  exit 1
fi

docker build -f "$DOCKERFILE" -t "$IMAGE" "$REPO_ROOT"

DOCKER_FLAGS=(--rm)
if [[ -t 1 ]]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi

ENV_FILE=${UPDATE_USER_ENV_FILE:-docker/env/prd.env}
NETWORK=${UPDATE_USER_NETWORK:-}
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
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
  echo "Set UPDATE_USER_ENV_FILE or pass --env-file <path>." >&2
  exit 1
fi

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

CMD=(docker run "${DOCKER_FLAGS[@]}")
if [[ -n "$NETWORK" ]]; then
  CMD+=(--network "$NETWORK")
fi
CMD+=(--add-host host.docker.internal:host-gateway --env-file "$TMP_ENV_FILE" -v "$REPO_ROOT":/workspace -w /workspace "$IMAGE" python scripts/update_user_embeddings/main.py)
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  CMD+=("${PY_ARGS[@]}")
fi

exec "${CMD[@]}"
