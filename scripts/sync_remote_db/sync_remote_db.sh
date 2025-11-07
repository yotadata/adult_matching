#!/usr/bin/env bash

# Sync a remote Supabase DB (data) to the local Supabase instance.
#
# Requirements:
# - Supabase CLI installed and logged in: https://supabase.com/docs/guides/cli
# - Your remote project is accessible by the CLI (supabase login; has org/project access)
#
# Default挙動:
# - docker/env/dev.env（NEXT_PUBLIC_SUPABASE_URL）から project ref を推測
# - `supabase link` / `supabase db dump` を使いリモート DB からダンプ
# - public と auth スキーマをリモートのスキーマ＋データでまるっと置換し、`--include-managed-schemas` 指定時のみ storage / graphql_public も対象
#
# Usage:
#   scripts/sync-remote-db-to-local.sh [--env-file docker/env/dev.env] [--local-env-file docker/env/dev.env] [--project-ref <ref>] [--db-password <pass>] [--exclude <schema.tbl[,..]>] [--exclude-embeddings] [--public-only] [--yes] [--no-start]
#
# Examples:
#   scripts/sync-remote-db-to-local.sh --yes
#   scripts/sync-remote-db-to-local.sh --yes --exclude-embeddings
#   scripts/sync-remote-db-to-local.sh --env-file docker/env/dev.env --yes
#   scripts/sync-remote-db-to-local.sh --project-ref abcd1234 --yes --no-start
#
# Notes:
# - `public.schema_migrations`（pop が参照）もリモートの値で上書きする。
# - If you need a full mirror including schema & migration history, consider a separate "full" mode.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"

ENV_FILE="docker/env/dev.env"
LOCAL_ENV_FILE="docker/env/dev.env"
PROJECT_REF=""
CONFIRM="false"
START_LOCAL="false" # default to post-apply against existing local stack
DB_PASSWORD=""
EXCLUDE_LIST=""
EXCLUDE_EMBEDDINGS="false"
INCLUDE_MANAGED_SCHEMAS="false" # include storage, graphql_public when explicitly requested
LOCAL_DB_HOST_VALUE=""
LOCAL_DB_PORT_VALUE=""
LOCAL_DB_USER_VALUE=""
LOCAL_DB_PASS_VALUE=""
LOCAL_DB_NAME_VALUE=""
LOCAL_DB_SSLMODE_VALUE=""
LOCAL_DB_SSLROOTCERT_VALUE=""
EXCLUDE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"; shift 2;;
    --project-ref)
      PROJECT_REF="$2"; shift 2;;
    --db-password)
      DB_PASSWORD="$2"; shift 2;;
    --local-env-file)
      LOCAL_ENV_FILE="$2"; shift 2;;
    --exclude)
      EXCLUDE_LIST="$2"; shift 2;;
    --exclude-embeddings)
      EXCLUDE_EMBEDDINGS="true"; shift;;
    --start-local)
      START_LOCAL="true"; shift;;
    --include-managed-schemas)
      INCLUDE_MANAGED_SCHEMAS="true"; shift;;
    --public-only)
      INCLUDE_MANAGED_SCHEMAS="false"; shift;;
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
# Load local env first so we capture local DB creds before remote overrides
if [[ -n "$LOCAL_ENV_FILE" && -f "$LOCAL_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$LOCAL_ENV_FILE"
  set +a
fi

# Snapshot local DB connection details before loading remote env (which may override POSTGRES_PASSWORD, etc.)
LOCAL_DB_HOST_VALUE="${LOCAL_DB_HOST:-127.0.0.1}"
LOCAL_DB_PORT_VALUE="${LOCAL_DB_PORT:-}"
LOCAL_DB_USER_VALUE="${LOCAL_DB_USER:-${POSTGRES_USER:-postgres}}"
LOCAL_DB_PASS_VALUE="${LOCAL_DB_PASSWORD:-${POSTGRES_PASSWORD:-postgres}}"
LOCAL_DB_NAME_VALUE="${LOCAL_DB_NAME:-postgres}"
LOCAL_DB_SSLMODE_VALUE="${LOCAL_DB_SSLMODE:-disable}"
LOCAL_DB_SSLROOTCERT_VALUE="${LOCAL_DB_SSLROOTCERT:-}"

# Load remote env (may be same as local; skip re-sourcing to avoid double work)
if [[ -f "$ENV_FILE" ]]; then
  if [[ "$ENV_FILE" != "$LOCAL_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
  fi
else
  echo "Warning: env file '$ENV_FILE' not found. Continuing with current environment." >&2
fi

# Remote dumps require password; default to PGPASSWORD from env if DB_PASSWORD not set
DB_PASSWORD="${DB_PASSWORD:-${PGPASSWORD:-}}"

# Prefer direct DB URL when available to avoid pooler TLS issues
REMOTE_DB_URL="${REMOTE_DB_URL:-${REMOTE_DATABASE_URL:-${SUPABASE_DB_URL:-}}}"

# Ensure stored local defaults are set even if env lacked values
LOCAL_DB_HOST_VALUE="${LOCAL_DB_HOST_VALUE:-127.0.0.1}"
LOCAL_DB_USER_VALUE="${LOCAL_DB_USER_VALUE:-postgres}"
LOCAL_DB_PASS_VALUE="${LOCAL_DB_PASS_VALUE:-postgres}"
LOCAL_DB_NAME_VALUE="${LOCAL_DB_NAME_VALUE:-postgres}"
LOCAL_DB_SSLMODE_VALUE="${LOCAL_DB_SSLMODE_VALUE:-disable}"
LOCAL_DB_SSLROOTCERT_VALUE="${LOCAL_DB_SSLROOTCERT_VALUE:-}"

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

if [[ -z "$REMOTE_DB_URL" ]]; then
  echo "Linking to remote project (if not already linked)..."
  supabase link --project-ref "$PROJECT_REF" --yes >/dev/null || true
else
  echo "Using direct database URL; skipping 'supabase link'."
fi

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

_ensure_ipv4_conninfo() {
  local conninfo="$1"
  python - <<'PY'
import os, socket, urllib.parse, sys
conn = sys.argv[1]
try:
    parsed = urllib.parse.urlsplit(conn)
except Exception:
    print(conn)
    sys.exit()
if not parsed.hostname:
    print(conn)
    sys.exit()
try:
    infos = socket.getaddrinfo(parsed.hostname, parsed.port or 5432, socket.AF_INET)
except socket.gaierror:
    infos = []
ipv4 = next((info[4][0] for info in infos if info[0] == socket.AF_INET), None)
if not ipv4:
    print(conn)
    sys.exit()
query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
query.setdefault('hostaddr', []).append(ipv4)
new_query = urllib.parse.urlencode(query, doseq=True)
print(urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment)))
PY
}

run_sql() {
  local sql="$1"
  local host="$LOCAL_DB_HOST_VALUE"
  local user="$LOCAL_DB_USER_VALUE"
  local pass="$LOCAL_DB_PASS_VALUE"
  local dbname="$LOCAL_DB_NAME_VALUE"
  local port="${LOCAL_DB_PORT_VALUE:-$(_detect_db_port)}"
  local sslmode="$LOCAL_DB_SSLMODE_VALUE"
  local sslroot="$LOCAL_DB_SSLROOTCERT_VALUE"
  if ! command -v psql >/dev/null 2>&1; then
    echo "Error: psql not found. Please install the Postgres client (psql)." >&2
    exit 1
  fi
  if [[ "$sslmode" == "disable" || -z "$sslmode" ]]; then
    PGSSLMODE=disable PGSSLROOTCERT= PGPASSWORD="$pass" \
      psql -h "$host" -p "$port" -U "$user" -d "$dbname" -v ON_ERROR_STOP=1 -c "$sql"
  else
    PGSSLMODE="$sslmode" PGSSLROOTCERT="$sslroot" PGPASSWORD="$pass" \
      psql -h "$host" -p "$port" -U "$user" -d "$dbname" -v ON_ERROR_STOP=1 -c "$sql"
  fi
}


restore_with_psql() {
  local file="$1"
  local host="$LOCAL_DB_HOST_VALUE"
  local user="$LOCAL_DB_USER_VALUE"
  local pass="$LOCAL_DB_PASS_VALUE"
  local dbname="$LOCAL_DB_NAME_VALUE"
  local port="${LOCAL_DB_PORT_VALUE:-$(_detect_db_port)}"
  local sslmode="$LOCAL_DB_SSLMODE_VALUE"
  local sslroot="$LOCAL_DB_SSLROOTCERT_VALUE"

  if ! command -v psql >/dev/null 2>&1; then
    echo "Error: psql not found. Please install the Postgres client (psql)." >&2
    exit 1
  fi

  echo "Applying file to local database via psql: $file"
  if [[ "$sslmode" == "disable" || -z "$sslmode" ]]; then
    PGSSLMODE=disable PGSSLROOTCERT= PGPASSWORD="$pass" \
      psql -h "$host" -p "$port" -U "$user" -d "$dbname" -v ON_ERROR_STOP=1 -f "$file"
  else
    PGSSLMODE="$sslmode" PGSSLROOTCERT="$sslroot" PGPASSWORD="$pass" \
      psql -h "$host" -p "$port" -U "$user" -d "$dbname" -v ON_ERROR_STOP=1 -f "$file"
  fi
}

ensure_supabase_roles() {
  local ensure_roles_sql
  ensure_roles_sql=$(cat <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_auth_admin') THEN
    CREATE ROLE supabase_auth_admin NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_admin') THEN
    CREATE ROLE supabase_admin NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_storage_admin') THEN
    CREATE ROLE supabase_storage_admin NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'dashboard_user') THEN
    CREATE ROLE dashboard_user NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pg_database_owner') THEN
    CREATE ROLE pg_database_owner NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticator') THEN
    CREATE ROLE authenticator NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
    CREATE ROLE authenticated NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    CREATE ROLE service_role NOLOGIN;
  END IF;
END$$;
SQL
)
  run_sql "$ensure_roles_sql"
}

drop_conflicting_types() {
  local drop_sql
  drop_sql=$(cat <<'SQL'
DO $$
BEGIN
  IF to_regtype('public.aal_level') IS NOT NULL THEN
    EXECUTE 'DROP TYPE public.aal_level CASCADE';
  END IF;
  IF to_regtype('public.session_status') IS NOT NULL THEN
    EXECUTE 'DROP TYPE public.session_status CASCADE';
  END IF;
  IF to_regtype('auth.aal_level') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.aal_level CASCADE';
  END IF;
  IF to_regtype('auth.code_challenge_method') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.code_challenge_method CASCADE';
  END IF;
  IF to_regtype('auth.factor_status') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.factor_status CASCADE';
  END IF;
  IF to_regtype('auth.factor_type') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.factor_type CASCADE';
  END IF;
  IF to_regtype('auth.mfa_factor_type') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.mfa_factor_type CASCADE';
  END IF;
  IF to_regtype('auth.oauth_authorization_status') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.oauth_authorization_status CASCADE';
  END IF;
  IF to_regtype('auth.oauth_client_type') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.oauth_client_type CASCADE';
  END IF;
  IF to_regtype('auth.oauth_registration_type') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.oauth_registration_type CASCADE';
  END IF;
  IF to_regtype('auth.oauth_response_type') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.oauth_response_type CASCADE';
  END IF;
  IF to_regtype('auth.one_time_token_type') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.one_time_token_type CASCADE';
  END IF;
  IF to_regtype('auth.session_status') IS NOT NULL THEN
    EXECUTE 'DROP TYPE auth.session_status CASCADE';
  END IF;
  IF to_regtype('storage.buckettype') IS NOT NULL THEN
    EXECUTE 'DROP TYPE storage.buckettype CASCADE';
  END IF;
END$$;
SQL
)
  run_sql "$drop_sql"
}

ensure_supabase_extensions() {
  run_sql "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;"
  run_sql "CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;"
  run_sql "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\" WITH SCHEMA public;"
  run_sql "CREATE EXTENSION IF NOT EXISTS citext WITH SCHEMA public;"
}

TARGET_SCHEMAS=()
TARGET_SCHEMA_CSV=""
TARGET_SCHEMA_SQL_LIST=""

build_target_schemas() {
  TARGET_SCHEMAS=("public" "auth")
  if [[ "$INCLUDE_MANAGED_SCHEMAS" == "true" ]]; then
    TARGET_SCHEMAS+=("storage" "graphql_public")
  fi
  TARGET_SCHEMA_CSV=$(IFS=,; printf "%s" "${TARGET_SCHEMAS[*]}")
  local sql_list=""
  for schema in "${TARGET_SCHEMAS[@]}"; do
    if [[ -n "$sql_list" ]]; then
      sql_list+=","
    fi
    sql_list+="'${schema}'"
  done
  TARGET_SCHEMA_SQL_LIST="$sql_list"
}

sync_schema_migrations() {
  if [[ -z "$REMOTE_DB_URL" ]]; then
    echo "[warn] REMOTE_DB_URL not set; skipping schema_migrations sync"
    return
  fi
  local supa_dump="$TMP_DIR/supabase_migrations.sql"
  echo "Dumping supabase_migrations schema via supabase CLI..."
  supabase db dump --db-url "$REMOTE_DB_URL" --data-only --use-copy -s supabase_migrations -f "$supa_dump"
  echo "Restoring supabase_migrations schema..."
  run_sql "DROP SCHEMA IF EXISTS supabase_migrations CASCADE;"
  run_sql "CREATE SCHEMA IF NOT EXISTS supabase_migrations;"
  restore_with_psql "$supa_dump"
}

run_sync() {
  build_target_schemas

  echo "Dumping remote SCHEMA (no data)..."
  SCHEMA_ARGS=(-f "$SCHEMA_FILE")
  for schema in "${TARGET_SCHEMAS[@]}"; do
    SCHEMA_ARGS+=(-s "$schema")
  done
  if [[ -n "$REMOTE_DB_URL" ]]; then
    supabase db dump --db-url "$REMOTE_DB_URL" "${SCHEMA_ARGS[@]}"
  else
    supabase db dump --linked "${SCHEMA_ARGS[@]}"
  fi

  echo "Dropping target schemas: ${TARGET_SCHEMA_CSV}"
  for schema in "${TARGET_SCHEMAS[@]}"; do
    run_sql "DROP SCHEMA IF EXISTS ${schema} CASCADE;"
    local owner="postgres"
    case "$schema" in
      auth|storage)
        owner="supabase_admin"
        ;;
    esac
    run_sql "CREATE SCHEMA IF NOT EXISTS ${schema}; ALTER SCHEMA ${schema} OWNER TO ${owner};"
  done
  ensure_supabase_roles
  drop_conflicting_types
  ensure_supabase_extensions

  echo "Applying remote schema to local..."
  # Ensure auxiliary schemas referenced by Supabase extensions exist before replay
  for schema in graphql extensions vault; do
    run_sql "CREATE SCHEMA IF NOT EXISTS ${schema};"
  done
  # Supabase's schema dump assumes publication exists; create stub if missing
  pub_sql=$(cat <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
    CREATE PUBLICATION supabase_realtime;
  END IF;
END$$;
SQL
)
  run_sql "$pub_sql"
  restore_with_psql "$SCHEMA_FILE"

  echo "Preparing local DB for data import (truncate target schemas)..."
  truncate_sql=$(cat <<SQL
DO \$\$DECLARE r record;
BEGIN
  FOR r IN
    SELECT format('%I.%I', n.nspname, c.relname) AS fqtn
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind IN ('r','p')
      AND n.nspname IN (${TARGET_SCHEMA_SQL_LIST})
      AND has_table_privilege(c.oid, 'TRUNCATE')
  LOOP
    EXECUTE format('TRUNCATE TABLE %s RESTART IDENTITY CASCADE', r.fqtn);
  END LOOP;
END\$\$;
SQL
)
  run_sql "$truncate_sql"

  echo "Dumping remote DATA (${TARGET_SCHEMA_CSV})..."
  DATA_ARGS=(--data-only --use-copy -f "$DATA_FILE")
  for schema in "${TARGET_SCHEMAS[@]}"; do
    DATA_ARGS+=(-s "$schema")
  done
  if [[ -n "$DB_PASSWORD" ]]; then
    DATA_ARGS+=( -p "$DB_PASSWORD" )
  fi
  if [[ -n "$REMOTE_DB_URL" ]]; then
    DATA_ARGS+=( --db-url "$REMOTE_DB_URL" )
  else
    DATA_ARGS+=( --linked )
  fi
  supabase db dump "${DATA_ARGS[@]}"

  echo "Applying remote data to local..."
  restore_with_psql "$DATA_FILE"

  sync_schema_migrations

  echo "Done. Local DB now mirrors remote schema and data ($PROJECT_REF)."
}

run_sync
exit 0
