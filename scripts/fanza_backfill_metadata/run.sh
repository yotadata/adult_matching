#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-ingest-fanza:latest

docker build -f scripts/ingest_fanza/Dockerfile -t "$IMAGE" .

ENV_FILE=${FANZA_BACKFILL_ENV_FILE:-docker/env/prd.env}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  echo "Set FANZA_BACKFILL_ENV_FILE (or INGEST_FANZA_ENV_FILE) to an existing env file before running." >&2
  exit 1
fi

DOCKER_FLAGS=(--rm)
if [ -t 1 ]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi
DOCKER_FLAGS+=(--add-host host.docker.internal:host-gateway)

exec docker run "${DOCKER_FLAGS[@]}" \
  --env-file "$ENV_FILE" \
  --env INGEST_FANZA_ENV_FILE="$ENV_FILE" \
  ${DOTENV_CONFIG_OVERRIDE:+--env DOTENV_CONFIG_OVERRIDE="$DOTENV_CONFIG_OVERRIDE"} \
  ${NEXT_PUBLIC_SUPABASE_URL:+--env NEXT_PUBLIC_SUPABASE_URL="$NEXT_PUBLIC_SUPABASE_URL"} \
  ${SUPABASE_URL:+--env SUPABASE_URL="$SUPABASE_URL"} \
  ${SUPABASE_DB_URL:+--env SUPABASE_DB_URL="$SUPABASE_DB_URL"} \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" tsx scripts/fanza_backfill_metadata/index.ts "$@"
