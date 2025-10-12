#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCHEMA_PATH="$REPO_ROOT/sql/local_schema.sql"

PGHOST="${LOCAL_PG_HOST:-localhost}"
PGPORT="${LOCAL_PG_PORT:-5433}"
PGUSER="${LOCAL_PG_USER:-two_tower}"
PGDATABASE="${LOCAL_PG_DATABASE:-two_tower}"

if [[ ! -f "$SCHEMA_PATH" ]]; then
  echo "Schema file not found: $SCHEMA_PATH" >&2
  exit 1
fi

psql "host=$PGHOST port=$PGPORT user=$PGUSER dbname=$PGDATABASE" -f "$SCHEMA_PATH"
