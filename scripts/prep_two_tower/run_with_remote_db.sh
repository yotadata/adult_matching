#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prep_two_tower/run_with_remote_db.sh --remote-db-url <URL> [prep_args...]

Description:
  1. Dumps the required tables from the remote Postgres database.
  2. Spins up an ephemeral Postgres container seeded with the dump.
  3. Runs prep_two_tower_dataset.py inside its Docker container against that DB.
  4. Cleans up the ephemeral Postgres container.

Environment variables:
  REMOTE_DB_URL     Remote Postgres connection string (postgres://...)
  RUN_ID            Optional identifier for dump / snapshot directory (defaults to UTC timestamp)
  DUMP_TABLES       Space separated list of tables to dump (default: "public.videos public.video_tags public.tags")
  LOCAL_DB_NAME     Ephemeral Postgres database name (default: tt_prep)
  LOCAL_DB_USER     Ephemeral Postgres user (default: tt_user)
  LOCAL_DB_PASS     Ephemeral Postgres password (default: tt_pass)
  LOCAL_DB_PORT     Host port to expose Postgres on (default: 6543)

Any additional arguments after the known flags are forwarded to prep_two_tower_dataset.py.
EOF
}

REMOTE_DB_URL="${REMOTE_DB_URL:-}"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-db-url)
      REMOTE_DB_URL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARD_ARGS+=("$@")
      break
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$REMOTE_DB_URL" ]]; then
  echo "[ERROR] --remote-db-url is required (or set REMOTE_DB_URL env)" >&2
  usage
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DUMP_TABLES_DEFAULT="public.videos public.video_tags public.tags"
DUMP_TABLES="${DUMP_TABLES:-$DUMP_TABLES_DEFAULT}"

LOCAL_DB_NAME="${LOCAL_DB_NAME:-tt_prep}"
LOCAL_DB_USER="${LOCAL_DB_USER:-tt_user}"
LOCAL_DB_PASS="${LOCAL_DB_PASS:-tt_pass}"
LOCAL_DB_PORT="${LOCAL_DB_PORT:-6543}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:15-alpine}"
NETWORK_NAME="tt-prep-net"
DB_CONTAINER="tt-prep-db-$RUN_ID"
PREP_IMAGE="adult-matching-prep:latest"

DUMP_DIR="$REPO_ROOT/ml/data/raw/db_dumps/$RUN_ID"
mkdir -p "$DUMP_DIR"
DUMP_FILE="$DUMP_DIR/prep_subset.sql"

echo "[INFO] Dumping tables ($DUMP_TABLES) from remote DB..."
TABLE_ARGS=()
for tbl in $DUMP_TABLES; do
  TABLE_ARGS+=("--table=$tbl")
done

docker run --rm \
  -e PGPASSWORD="${PGPASSWORD:-}" \
  -e REMOTE_DB_URL="$REMOTE_DB_URL" \
  -v "$DUMP_DIR":/dump \
  "$POSTGRES_IMAGE" \
  sh -c 'pg_dump "$REMOTE_DB_URL" --no-owner --no-privileges '"${TABLE_ARGS[*]}"' --file=/dump/prep_subset.sql'

echo "[INFO] Creating docker network $NETWORK_NAME (if not exists)..."
if ! docker network inspect "$NETWORK_NAME" > /dev/null 2>&1; then
  docker network create "$NETWORK_NAME" > /dev/null
fi

echo "[INFO] Starting ephemeral Postgres container $DB_CONTAINER..."
docker run --rm -d \
  --name "$DB_CONTAINER" \
  --network "$NETWORK_NAME" \
  -e POSTGRES_USER="$LOCAL_DB_USER" \
  -e POSTGRES_PASSWORD="$LOCAL_DB_PASS" \
  -e POSTGRES_DB="$LOCAL_DB_NAME" \
  -p "${LOCAL_DB_PORT}:5432" \
  "$POSTGRES_IMAGE" > /dev/null

cleanup() {
  echo "[INFO] Cleaning up Postgres container..."
  docker rm -f "$DB_CONTAINER" > /dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[INFO] Waiting for Postgres to become ready..."
tries=0
until docker exec "$DB_CONTAINER" pg_isready -U "$LOCAL_DB_USER" -d "$LOCAL_DB_NAME" > /dev/null 2>&1; do
  sleep 1
  tries=$((tries+1))
  if [[ $tries -gt 30 ]]; then
    echo "[ERROR] Postgres container did not become ready in time" >&2
    exit 1
  fi
done

echo "[INFO] Loading dump into local Postgres..."
docker exec -i "$DB_CONTAINER" psql -U "$LOCAL_DB_USER" -d "$LOCAL_DB_NAME" < "$DUMP_FILE"

echo "[INFO] Building prep image..."
docker build -f "$SCRIPT_DIR/Dockerfile" -t "$PREP_IMAGE" "$REPO_ROOT"

DB_URL_FOR_PREP="postgresql://${LOCAL_DB_USER}:${LOCAL_DB_PASS}@${DB_CONTAINER}:5432/${LOCAL_DB_NAME}"

echo "[INFO] Running prep script against local Postgres..."
docker run --rm -it \
  --network "$NETWORK_NAME" \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  "$PREP_IMAGE" \
  --db-url "$DB_URL_FOR_PREP" "${FORWARD_ARGS[@]}"

echo "[INFO] Prep completed successfully (run_id=$RUN_ID)"
