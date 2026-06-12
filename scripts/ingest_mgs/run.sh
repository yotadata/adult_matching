#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-ingest-mgs:latest

docker build -f scripts/ingest_mgs/Dockerfile -t "$IMAGE" .

ENV_FILE=${INGEST_MGS_ENV_FILE:-docker/env/prd.env}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  echo "Set INGEST_MGS_ENV_FILE to an existing env file (e.g. docker/env/dev.env) before running." >&2
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
if [[ "${PAGE_LIMIT:-}" != "" ]]; then
  DOCKER_ENV_ARGS+=(--env "PAGE_LIMIT=${PAGE_LIMIT}")
fi
if [[ "${SKIP_EXISTING:-}" != "" ]]; then
  DOCKER_ENV_ARGS+=(--env "SKIP_EXISTING=${SKIP_EXISTING}")
fi
if [[ "${INGEST_MGS_DEBUG:-}" != "" ]]; then
  DOCKER_ENV_ARGS+=(--env "INGEST_MGS_DEBUG=${INGEST_MGS_DEBUG}")
fi
if [[ "${REQUEST_DELAY_MS:-}" != "" ]]; then
  DOCKER_ENV_ARGS+=(--env "REQUEST_DELAY_MS=${REQUEST_DELAY_MS}")
fi

exec docker run "${DOCKER_FLAGS[@]}" \
  --env-file "$ENV_FILE" \
  --env INGEST_MGS_ENV_FILE="$ENV_FILE" \
  ${DOCKER_ENV_ARGS[@]+"${DOCKER_ENV_ARGS[@]}"} \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"
