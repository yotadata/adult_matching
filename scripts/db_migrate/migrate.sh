#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-docker/env/dev.env}
COMPOSE_FILE=${COMPOSE_FILE:-docker/compose.yml}
RELOAD_REST=${RELOAD_REST:-true}

echo "Using env:    $ENV_FILE"
echo "Using compose: $COMPOSE_FILE"

echo "Running db-migrate (apply pending supabase/migrations/*.sql) ..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up db-migrate

if [[ "$RELOAD_REST" == "true" ]]; then
  echo "Restarting PostgREST (schema reload) ..."
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" restart rest
fi

echo "Done."

