#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-upsert:latest

docker build -f scripts/upsert_two_tower/Dockerfile -t "$IMAGE" .

DOCKER_FLAGS=(--rm)
NETWORK=${UPSERT_TT_NETWORK:-}
if [ -t 1 ]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi

ENV_FILE=${UPSERT_TT_ENV_FILE:-docker/env/prd.env}
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
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
  echo "Set UPSERT_TT_ENV_FILE or pass --env-file <path> to a valid env file." >&2
  exit 1
fi

# Source env file to resolve nested references (e.g., ${PGPASSWORD})
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

# If REMOTE_DATABASE_URL is present, append hostaddr=<ipv4> to avoid IPv6-only DNS results
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
for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
    if family == socket.AF_INET:
        ipv4_addr = sockaddr[0]
        break
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
  )
  if [[ -n "$ipv4_url" ]]; then
    export REMOTE_DATABASE_URL="$ipv4_url"
    export SUPABASE_DB_URL="$ipv4_url"
  fi
fi


# Persist resolved variables to a temporary env file for docker
TMP_ENV_FILE=$(mktemp)
trap 'rm -f "$TMP_ENV_FILE"' EXIT

# Collect variable names declared in the env file (ignores comments / blank lines)
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
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  exec docker run "${DOCKER_FLAGS[@]}" \
    ${NETWORK:+--network "$NETWORK"} \
    --add-host host.docker.internal:host-gateway \
    --env-file "$TMP_ENV_FILE" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE" \
    python scripts/upsert_two_tower/upsert_two_tower.py \
    "${PY_ARGS[@]}"
else
  exec docker run "${DOCKER_FLAGS[@]}" \
    ${NETWORK:+--network "$NETWORK"} \
    --add-host host.docker.internal:host-gateway \
    --env-file "$TMP_ENV_FILE" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE" \
    python scripts/upsert_two_tower/upsert_two_tower.py
fi
