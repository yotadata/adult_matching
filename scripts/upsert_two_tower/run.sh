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

exec docker run "${DOCKER_FLAGS[@]}" \
  ${NETWORK:+--network "$NETWORK"} \
  --add-host host.docker.internal:host-gateway \
  --env-file "$ENV_FILE" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" \
  python scripts/upsert_two_tower/upsert_two_tower.py \
  "${PY_ARGS[@]}"
