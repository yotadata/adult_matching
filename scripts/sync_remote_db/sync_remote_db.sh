#!/usr/bin/env bash

# Sync a remote Supabase DB (data) to the local Supabase instance.
#
# Requirements:
# - Supabase CLI installed and logged in: https://supabase.com/docs/guides/cli
# - Your remote project is accessible by the CLI (supabase login; has org/project access)
#
# Default behavior (modes):
# - Reads project info from docker/env/dev.env (NEXT_PUBLIC_SUPABASE_URL) to derive project ref
# - Links the Supabase project (supabase link) and uses `db dump --linked`
# - Starts local stack (unless --no-start)
# - Mode "schema-mirror" (default): Rebuild local DB from the REMOTE schema, then import data.
#   - Drops local user schemas (public, auth, storage, graphql_public) and applies remote DDL, then data.
#   - Does NOT apply local migrations.
# - Mode "data": Keeps local schema (from your local migrations) and imports selected remote data only.
# - Mode "full": Keeps local schema and imports ALL remote data including supabase_migrations history.
#
# Usage:
#   scripts/sync-remote-db-to-local.sh [--env-file docker/env/dev.env] [--project-ref <ref>] [--mode schema-mirror|full|data] [--db-password <pass>] [--exclude <schema.tbl[,..]>] [--exclude-embeddings] [--yes] [--no-start]
#
# Examples:
#   scripts/sync-remote-db-to-local.sh --yes
#   scripts/sync-remote-db-to-local.sh --mode data --yes
#   scripts/sync-remote-db-to-local.sh --env-file docker/env/dev.env --yes
#   scripts/sync-remote-db-to-local.sh --project-ref abcd1234 --yes --no-start
#
# Notes:
# - This approach keeps local `supabase_migrations.schema_migrations` untouched, so `migration up` / `db push`
#   work from your local migration files as usual.
# - We import data only from common schemas (public, auth, storage, graphql_public) by default.
# - If you need a full mirror including schema & migration history, consider a separate "full" mode.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

ENV_FILE="docker/env/dev.env"
PROJECT_REF=""
CONFIRM="false"
START_LOCAL="false" # default to post-apply against existing local stack
MODE="schema-mirror" # schema-mirror | full | data
DB_PASSWORD=""
EXCLUDE_LIST=""
EXCLUDE_EMBEDDINGS="false"
INCLUDE_MANAGED_SCHEMAS="false" # include auth, storage, graphql_public data in schema-mirror

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"; shift 2;;
    --project-ref)
      PROJECT_REF="$2"; shift 2;;
    --mode)
      MODE="$2"; shift 2;;
    --db-password)
      DB_PASSWORD="$2"; shift 2;;
    --exclude)
      EXCLUDE_LIST="$2"; shift 2;;
    --exclude-embeddings)
      EXCLUDE_EMBEDDINGS="true"; shift;;
    --start-local)
      START_LOCAL="true"; shift;;
    --include-managed-schemas)
      INCLUDE_MANAGED_SCHEMAS="true"; shift;;
    --yes|-y)
      CONFIRM="true"; shift;;
    --no-start)
      START_LOCAL="false"; shift;;
    -h|--help)
      sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

if ! command -v supabase >/dev/null 2>&1; then
  echo "Error: supabase CLI not found. Install: https://supabase.com/docs/guides/cli" >&2
  exit 1
fi

# Always load env file for local DB credentials and defaults
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# Load project ref from URL only if not provided
if [[ -z "$PROJECT_REF" ]]; then
  if [[ -n "${NEXT_PUBLIC_SUPABASE_URL:-}" ]]; then
    # Extract subdomain before .supabase.co as project ref
    maybe_ref=$(printf "%s\n" "$NEXT_PUBLIC_SUPABASE_URL" | sed -E 's#^https?://([^.]+)\.supabase\.co.*#\1#')
    if [[ "$maybe_ref" != "$NEXT_PUBLIC_SUPABASE_URL" && -n "$maybe_ref" ]]; then
      PROJECT_REF="$maybe_ref"
    fi
  fi
fi

if [[ -z "$PROJECT_REF" ]]; then
  echo "Error: Could not determine Supabase project ref. Provide via --project-ref or set NEXT_PUBLIC_SUPABASE_URL in $ENV_FILE" >&2
  exit 1
fi

echo "Project ref: $PROJECT_REF"
echo "Env file:    $ENV_FILE"

if [[ "$CONFIRM" != "true" ]]; then
  echo "This will OVERWRITE your local Supabase database with data from remote project '$PROJECT_REF'."
  read -r -p "Continue? (y/N) " reply
  case "$reply" in
    y|Y|yes|YES) :;;
    *) echo "Aborted."; exit 1;;
  esac
fi

if [[ "$START_LOCAL" == "true" ]]; then
  echo "Starting local Supabase (if not running)..."
  supabase start >/dev/null || true
else
  echo "Skipping 'supabase start' (per --no-start)"
fi

echo "Linking to remote project (if not already linked)..."
supabase link --project-ref "$PROJECT_REF" --yes >/dev/null || true

TMP_DIR="$ROOT_DIR/supabase/.tmp/db-sync"
mkdir -p "$TMP_DIR"
DATA_FILE="$TMP_DIR/remote_data.sql"
SCHEMA_FILE="$TMP_DIR/remote_schema.sql"

# Helper to detect local DB connection and run psql
_detect_db_port() {
  local db_port
  if [[ -f supabase/config.toml ]]; then
    db_port=$(awk 'BEGIN{FS="= *"} /^\[db\]/{f=1;next} /^\[/{f=0} f && $1 ~ /^port/{gsub(/[^0-9]/, "", $2); print $2; exit}' supabase/config.toml)
  fi
  printf "%s" "${db_port:-54322}"
}

run_sql() {
  local sql="$1"
  local host="${LOCAL_DB_HOST:-127.0.0.1}"
  local user="${LOCAL_DB_USER:-postgres}"
  local pass="${LOCAL_DB_PASSWORD:-${POSTGRES_PASSWORD:-postgres}}"
  local dbname="${LOCAL_DB_NAME:-postgres}"
  local port
  port=$(_detect_db_port)
  if ! command -v psql >/dev/null 2>&1; then
    echo "Error: psql not found. Please install the Postgres client (psql)." >&2
    exit 1
  fi
  PGPASSWORD="$pass" psql -h "$host" -p "$port" -U "$user" -d "$dbname" -v ON_ERROR_STOP=1 -c "$sql"
}

echo "Dumping remote data ($MODE mode)... (this may take a while)"
dump_data() {
  local mode="$1"
  local -a args=(--linked --data-only --use-copy)
  local -a exclude_args=()
  if [[ "$mode" == "data" ]]; then
    args+=( -s public,auth,storage,graphql_public )
    exclude_args+=( -x supabase_migrations.schema_migrations )
  fi
  if [[ "$EXCLUDE_EMBEDDINGS" == "true" ]]; then
    exclude_args+=( -x public.user_embeddings -x public.video_embeddings )
  fi
  if [[ -n "$EXCLUDE_LIST" ]]; then
    # Convert comma-separated to space and append -x for each
    IFS=',' read -r -a _items <<< "$EXCLUDE_LIST"
    for it in "${_items[@]}"; do
      [[ -n "$it" ]] && exclude_args+=( -x "$it" )
    done
  fi
  # Append excludes only if any were collected to avoid set -u issues
  if [[ ${#exclude_args[@]:-0} -gt 0 ]]; then
    args+=( "${exclude_args[@]}" )
  fi
  args+=( -f "$DATA_FILE" )
  if [[ -n "$DB_PASSWORD" ]]; then
    args+=( -p "$DB_PASSWORD" )
  fi
  supabase db dump "${args[@]}"
}

restore_with_psql() {
  local file="$1"
  local host="${LOCAL_DB_HOST:-127.0.0.1}"
  local user="${LOCAL_DB_USER:-postgres}"
  local pass="${LOCAL_DB_PASSWORD:-${POSTGRES_PASSWORD:-postgres}}"
  local dbname="${LOCAL_DB_NAME:-postgres}"
  local port
  port=$(_detect_db_port)

  if ! command -v psql >/dev/null 2>&1; then
    echo "Error: psql not found. Please install the Postgres client (psql)." >&2
    exit 1
  fi

  echo "Applying file to local database via psql: $file"
  if ! PGPASSWORD="$pass" psql -h "$host" -p "$port" -U "$user" -d "$dbname" -v ON_ERROR_STOP=1 -f "$file"; then
    echo "Direct TCP connection failed. Falling back to docker exec into 'supabase_db'..." >&2
    if ! docker ps --format '{{.Names}}' | grep -Fxq supabase_db; then
      echo "Error: Container 'supabase_db' not found. Set LOCAL_DB_HOST/PORT/USER/PASSWORD or ensure the container name matches." >&2
      exit 1
    fi
    docker exec -i -e PGPASSWORD="$pass" supabase_db psql -U "$user" -d "$dbname" -v ON_ERROR_STOP=1 -f - < "$file"
  fi
}

if [[ "$MODE" == "schema-mirror" ]]; then
  echo "Dumping remote SCHEMA (no data)..."
  # No --data-only to get schema DDL
  supabase db dump --linked -f "$SCHEMA_FILE"

  echo "Dropping local schema: public ..."
  # Drop only 'public' to avoid permission issues on managed schemas (auth/storage/graphql_public)
  run_sql "DROP SCHEMA IF EXISTS public CASCADE;"
  echo "Recreating schema: public ..."
  run_sql "CREATE SCHEMA IF NOT EXISTS public; ALTER SCHEMA public OWNER TO postgres;"

  echo "Applying remote schema to local..."
  restore_with_psql "$SCHEMA_FILE"

  if [[ "$INCLUDE_MANAGED_SCHEMAS" == "true" ]]; then
    echo "Preparing local DB for data import (truncate public/auth/storage/graphql_public as permitted)..."
    run_sql "DO \$\$DECLARE r record; BEGIN FOR r IN SELECT format('%I.%I', n.nspname, c.relname) AS fqtn FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind IN ('r','p') AND n.nspname IN ('public','auth','storage','graphql_public') AND has_table_privilege(c.oid, 'TRUNCATE') LOOP EXECUTE format('TRUNCATE TABLE %s RESTART IDENTITY CASCADE', r.fqtn); END LOOP; END\$\$;"
    echo "Dumping remote DATA (public + managed schemas)..."
    # Include all key schemas
    DATA_ARGS=(--linked --data-only --use-copy -s public,auth,storage,graphql_public -f "$DATA_FILE")
  else
    echo "Preparing local DB for data import (truncate public)..."
    run_sql "DO \$\$DECLARE r record; BEGIN FOR r IN SELECT format('%I.%I', n.nspname, c.relname) AS fqtn FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind IN ('r','p') AND n.nspname IN ('public') LOOP EXECUTE format('TRUNCATE TABLE %s RESTART IDENTITY CASCADE', r.fqtn); END LOOP; END\$\$;"
    echo "Dumping remote DATA (public only)..."
    DATA_ARGS=(--linked --data-only --use-copy -s public -f "$DATA_FILE")
  fi
  if [[ -n "$DB_PASSWORD" ]]; then
    DATA_ARGS+=( -p "$DB_PASSWORD" )
  fi
  supabase db dump "${DATA_ARGS[@]}"

  echo "Applying remote data to local..."
  restore_with_psql "$DATA_FILE"

  echo "Done. Local DB now mirrors remote schema and data ($PROJECT_REF)."
  exit 0
fi

# Non-mirror modes keep local schema created by local migrations
if [[ "$START_LOCAL" == "true" ]]; then
  echo "Resetting local database (apply local migrations; skip seeds)..."
  # Newer Supabase CLI does not support --force; use global --yes to skip prompts
  supabase db reset --local --no-seed --yes >/dev/null
else
  echo "Skipping 'supabase db reset' (per --no-start). Using current local schema."
fi

dump_data "$MODE"

if [[ "$MODE" == "full" ]]; then
  echo "Preparing to replace migration history with remote..."
  COMBINED_FILE="$TMP_DIR/remote_data_with_truncate.sql"
  echo "TRUNCATE TABLE supabase_migrations.schema_migrations;" > "$COMBINED_FILE"
  cat "$DATA_FILE" >> "$COMBINED_FILE"
  echo "Restoring data (including migration history) to local..."
  restore_with_psql "$COMBINED_FILE"
else
  echo "Restoring data to local (keeping local migration history)..."
  restore_with_psql "$DATA_FILE"
fi

echo "Done. Local DB now mirrors remote ($PROJECT_REF)."
