#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-docker/env/dev.env}
COMPOSE_FILE=${COMPOSE_FILE:-docker/compose.yml}
RELOAD_REST=${RELOAD_REST:-true}

echo "Using env:    $ENV_FILE"
echo "Using compose: $COMPOSE_FILE"

# Load Postgres password from env file
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi
if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
  echo "Error: POSTGRES_PASSWORD not found in $ENV_FILE" >&2
  exit 1
fi

# Determine compose project name to join the right network
PROJECT_NAME=$(awk 'BEGIN{FS=":"} $1 ~ /^name$/ {gsub(/ /,"",$2); print $2; exit}' "$COMPOSE_FILE")
if [[ -z "$PROJECT_NAME" ]]; then
  PROJECT_NAME=$(basename "$(pwd)")
fi
NETWORK="${PROJECT_NAME}_default"

echo "Running one-off migration container on network: $NETWORK"
docker run --rm \
  --network "$NETWORK" \
  -e PGPASSWORD="$POSTGRES_PASSWORD" \
  -e DB_HOST="db" -e DB_PORT="5432" -e DB_USER="postgres" -e DB_NAME="postgres" \
  -v "$(pwd)/supabase/migrations:/migrations:ro" \
  -v "$(pwd)/docker/db/migrate.sh:/migrate.sh:ro" \
  postgres:15-alpine sh /migrate.sh

if [[ "$RELOAD_REST" == "true" ]]; then
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" restart rest || true
fi
