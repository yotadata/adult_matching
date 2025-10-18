#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-docker/env/dev.env}
COMPOSE_FILE=${COMPOSE_FILE:-docker/compose.yml}
HARD=${HARD:-false}

if [[ "$HARD" == "true" ]]; then
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down -v
  exec docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up --build
fi

# Soft reset: drop only DB volume, re-init DB, then apply pending migrations and restart deps

# Determine project name for volume computation
PROJECT_NAME=$(awk 'BEGIN{FS=":"} $1 ~ /^name$/ {gsub(/ /,"",$2); print $2; exit}' "$COMPOSE_FILE")
if [[ -z "$PROJECT_NAME" ]]; then
  PROJECT_NAME=$(basename "$(pwd)")
fi
DB_VOLUME="${PROJECT_NAME}_db_data"

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" stop rest auth realtime storage kong studio edge-videos-feed edge-ai-recommend || true
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" rm -f db db-init db-migrate || true
docker volume rm "$DB_VOLUME" || true

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d db
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up db-init
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up db-migrate
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" restart rest auth realtime storage edge-videos-feed edge-ai-recommend kong
echo "DB reset completed."
