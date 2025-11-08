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

DOCKER_COMPOSE_FILE="$ROOT_DIR/docker/compose.yml"
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

require_ipv6() {
  if docker run --rm --network host alpine:3.20 sh -c 'ping6 -c1 -w2 ::1 >/dev/null 2>&1'; then
    return 0
  fi
  cat >&2 <<'MSG'
[error] IPv6 networking is required for scripts/sync_remote_db.
Please enable IPv6 for Docker (e.g., Docker Desktop → Settings → Docker Engine → add `"ipv6": true`) and restart Docker Desktop.
MSG
  exit 1
}

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

require_ipv6

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

append_sslmode_require() {
  local url="$1"
  if [[ "$url" == *"sslmode="* || -z "$url" ]]; then
    printf "%s" "$url"
    return
  fi
  if [[ "$url" == *\?* ]]; then
    printf "%s&sslmode=require" "$url"
  else
    printf "%s?sslmode=require" "$url"
  fi
}

pg_dump_table_ipv6() {
  local table="$1"
  local dest="$2"
  local dump_url
  dump_url=$(append_sslmode_require "$REMOTE_DB_URL")
  if [[ -z "$dump_url" ]]; then
    echo "[warn] REMOTE_DB_URL not set; cannot pg_dump $table"
    return 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "[warn] docker CLI not found; cannot pg_dump $table"
    return 1
  fi
  if [[ ! -f "$DOCKER_COMPOSE_FILE" ]]; then
    echo "[warn] docker compose file missing; cannot pg_dump $table"
    return 1
  fi
  mkdir -p "$(dirname "$dest")"
  local temp_name
  temp_name=$(basename "$dest")
  local container_cmd='set -euo pipefail; pg_dump "$DB_URL" --data-only --column-inserts --no-owner --table "$TABLE_NAME" > "$OUTPUT_FILE"'
  if docker run --rm --network host \
      -e DB_URL="$dump_url" \
      -e TABLE_NAME="$table" \
      -e OUTPUT_FILE="/tmp/out/$temp_name" \
      -v "$TMP_DIR":/tmp/out \
      postgres:16-alpine \
      /bin/sh -c "$container_cmd"; then
    return 0
  fi
  echo "[warn] pg_dump over IPv6 failed for $table"
  return 1
}

generate_auth_schema_migrations_from_image() {
  local dest_file="$1"
  local tmp_list="$TMP_DIR/auth_migration_files.txt"
  if [[ ! -f "$DOCKER_COMPOSE_FILE" ]]; then
    echo "[error] docker compose file not found at $DOCKER_COMPOSE_FILE" >&2
    return 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "[error] docker CLI not found; cannot derive auth schema migrations" >&2
    return 1
  fi
  echo "[info] auth.schema_migrations not in dump; deriving versions via supabase_auth image..."
  if ! docker compose -f "$DOCKER_COMPOSE_FILE" run --rm --no-deps supabase_auth \
    sh -c 'set -euo pipefail; ls -1 /app/migrations/*_up.sql' >"$tmp_list"; then
    echo "[error] Failed to list migrations from supabase_auth image" >&2
    return 1
  fi
  local rows
  if ! rows=$(python - "$tmp_list" "$dest_file" <<'PY'
import pathlib
import re
import sys

src = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
pattern = re.compile(r'(\d{8,})')

versions = []
for line in src.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    match = pattern.search(line)
    if match:
        versions.append(match.group(1))

uniq = sorted(set(versions))
if not uniq:
    raise SystemExit("no migrations found in image listing")

with dest.open('w', encoding='utf-8') as fout:
    fout.write('COPY "auth"."schema_migrations" ("version") FROM stdin;\n')
    for v in uniq:
        fout.write(f"{v}\n")
    fout.write('\\.\n')

print(len(uniq))
PY
  ); then
    echo "[error] Failed to build auth.schema_migrations fixture from image" >&2
    return 1
  fi
  echo "$rows"
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
  run_sql "CREATE TABLE IF NOT EXISTS supabase_migrations.schema_migrations (version bigint PRIMARY KEY);"
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

  # Sync supabase_migrations schema/data separately
  local history_schema="$TMP_DIR/history_schema.sql"
  local history_data="$TMP_DIR/history_data.sql"
  echo "Dumping supabase_migrations schema/data via supabase CLI..."
  supabase db dump --db-url "$REMOTE_DB_URL" --schema supabase_migrations -f "$history_schema"
  supabase db dump --db-url "$REMOTE_DB_URL" --use-copy --data-only --schema supabase_migrations -f "$history_data"
  echo "Restoring supabase_migrations schema/data..."
  run_sql "DROP SCHEMA IF EXISTS supabase_migrations CASCADE; CREATE SCHEMA supabase_migrations;"
  restore_with_psql "$history_schema"
  restore_with_psql "$history_data"

  # Sync auth.schema_migrations separately to keep GoTrue migrations in sync
  if [[ -n "$REMOTE_DB_URL" ]]; then
    local auth_dump_full="$TMP_DIR/auth_schema_full.sql"
    local auth_filtered="$TMP_DIR/auth_schema_migrations.sql"
    local auth_direct_dump="$TMP_DIR/auth_schema_migrations_direct.sql"
    echo "Dumping auth schema via supabase CLI..."
    supabase db dump --db-url "$REMOTE_DB_URL" --use-copy --data-only --schema auth -f "$auth_dump_full"
    local auth_rows=""
    echo "Attempting direct dump of auth.schema_migrations via pg_dump (IPv6)..."
    if ! pg_dump_table_ipv6 "auth.schema_migrations" "$auth_direct_dump"; then
      echo "[error] pg_dump for auth.schema_migrations failed. IPv6 connectivity to Supabase is required." >&2
      echo "Please enable IPv6 for Docker (e.g., Docker Desktop Settings → Docker Engine → set \"ipv6\": true) and retry." >&2
      exit 1
    fi
    if [[ -s "$auth_direct_dump" ]]; then
      cp "$auth_direct_dump" "$auth_filtered"
      auth_rows=$(python - "$auth_filtered" <<'PY'
import sys

count = 0
with open(sys.argv[1], encoding='utf-8') as fin:
    for line in fin:
        stripped = line.strip()
        upper = stripped.upper()
        if not stripped or stripped == r'\.' or stripped.startswith('COPY '):
            continue
        if upper.startswith('INSERT INTO'):
            count += 1
        elif '\t' in stripped and not upper.startswith('--'):
            count += 1
print(count)
PY
)
    else
      local extract_status=0
      auth_rows=$(python - "$auth_dump_full" "$auth_filtered" <<'PY'
import re
import sys

src, dst = sys.argv[1:3]
pattern = re.compile(r'^COPY\s+"?auth"?\."schema_migrations"\b', re.IGNORECASE)
capture = False
rows = 0

with open(src, 'r', encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not capture and pattern.match(line):
            capture = True
        if capture:
            fout.write(line)
            stripped = line.strip()
            if stripped and not stripped.startswith('COPY ') and stripped != r'\.':
                rows += 1
            if stripped == r'\.':
                break

if not capture:
    raise SystemExit("COPY block for auth.schema_migrations not found in dump")

print(rows)
PY
    ) || extract_status=$?
    if [[ $extract_status -ne 0 || ! -s "$auth_filtered" ]]; then
      if ! auth_rows=$(generate_auth_schema_migrations_from_image "$auth_filtered"); then
        echo "[error] Failed to hydrate auth.schema_migrations from any source" >&2
        exit 1
      fi
    fi
    fi
    if [[ -s "$auth_filtered" ]]; then
      echo "Restoring auth.schema_migrations (rows: ${auth_rows})..."
      run_sql "TRUNCATE TABLE IF EXISTS auth.schema_migrations;"
      restore_with_psql "$auth_filtered"
    else
      echo "[warn] auth.schema_migrations block not found; skipping" >&2
    fi
  fi

  echo "Done. Local DB now mirrors remote schema and data ($PROJECT_REF)."
}

run_sync
exit 0
