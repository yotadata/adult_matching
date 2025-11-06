#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-docker/env/dev.env}
COMPOSE_FILE=${COMPOSE_FILE:-docker/compose.yml}
HARD=${HARD:-false}

echo "Using env:    $ENV_FILE"
echo "Using compose: $COMPOSE_FILE"

if [[ "$HARD" == "true" ]]; then
  echo "[HARD] Bringing down stack and removing all named volumes ..."
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down -v
  echo "Bringing stack up ..."
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up --build
  exit 0
fi

# Soft reset: drop only DB volume, re-init DB, then restart dependent services

# Detect compose project name to compute volume name
PROJECT_NAME=$(awk 'BEGIN{FS=":"} $1 ~ /^name$/ {gsub(/ /,"",$2); print $2; exit}' "$COMPOSE_FILE")
if [[ -z "$PROJECT_NAME" ]]; then
  # Fallback to directory name
  PROJECT_NAME=$(basename "$(pwd)")
fi

DB_VOLUME="${PROJECT_NAME}_db_data"

echo "Stopping dependent services ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" stop rest auth realtime storage kong studio edge-videos-feed edge-ai-recommend edge-analysis-results || true

echo "Removing DB containers ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" rm -f db db-init db-migrate || true

echo "Removing DB volume: $DB_VOLUME ..."
docker volume rm "$DB_VOLUME" || true

echo "Starting db ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d db

echo "Applying init SQL ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up db-init

echo "Applying pending migrations ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up db-migrate

echo "Restarting dependent services ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" restart rest auth realtime storage edge-videos-feed edge-ai-recommend edge-analysis-results kong

echo "DB reset completed."
