#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-ingest-fanza:latest

docker build -f scripts/ingest_fanza/Dockerfile -t "$IMAGE" .

ENV_FILE=${INGEST_FANZA_ENV_FILE:-docker/env/prd.env}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  echo "Set INGEST_FANZA_ENV_FILE to an existing env file (e.g. docker/env/dev.env) before running." >&2
  exit 1
fi

DOCKER_FLAGS=(--rm)
if [ -t 1 ]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi

DOCKER_ENV_ARGS=()
if [[ "${GTE_RELEASE_DATE:-}" != "" ]]; then
  DOCKER_ENV_ARGS+=(--env "GTE_RELEASE_DATE=${GTE_RELEASE_DATE}")
fi
if [[ "${LTE_RELEASE_DATE:-}" != "" ]]; then
  DOCKER_ENV_ARGS+=(--env "LTE_RELEASE_DATE=${LTE_RELEASE_DATE}")
fi
if [[ "${INGEST_FANZA_DEBUG:-}" != "" ]]; then
  DOCKER_ENV_ARGS+=(--env "INGEST_FANZA_DEBUG=${INGEST_FANZA_DEBUG}")
fi

exec docker run "${DOCKER_FLAGS[@]}" \
  --env-file "$ENV_FILE" \
  --env INGEST_FANZA_ENV_FILE="$ENV_FILE" \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"
