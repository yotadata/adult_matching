#!/usr/bin/env bash

# Sync a remote Supabase DB (data) to the local Supabase instance.
#
# Requirements:
# - Supabase CLI installed and logged in: https://supabase.com/docs/guides/cli
# - Your remote project is accessible by the CLI (supabase login; has org/project access)
#
# Default behavior (two modes):
# - Reads project info from .env.remote (NEXT_PUBLIC_SUPABASE_URL) to derive project ref
# - Starts local stack (unless --no-start)
# - Resets local DB using local migrations (keeps local migration model)
# - Mode "full" (default): dumps ALL remote data, including supabase_migrations, and restores it.
#   - We TRUNCATE supabase_migrations.schema_migrations locally before restore, so remote history takes effect.
# - Mode "data": dumps remote data for selected schemas only, EXCLUDING migration history (safer; keeps local history)
#
# Usage:
#   scripts/sync-remote-db-to-local.sh [--env-file .env.remote] [--project-ref <ref>] [--mode full|data] [--yes] [--no-start]
#
# Examples:
#   scripts/sync-remote-db-to-local.sh --yes
#   scripts/sync-remote-db-to-local.sh --mode data --yes
#   scripts/sync-remote-db-to-local.sh --env-file .env.remote --yes
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

ENV_FILE=".env.remote"
PROJECT_REF=""
CONFIRM="false"
START_LOCAL="true"
MODE="full" # full | data

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"; shift 2;;
    --project-ref)
      PROJECT_REF="$2"; shift 2;;
    --mode)
      MODE="$2"; shift 2;;
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

# Load env file if exists to get NEXT_PUBLIC_SUPABASE_URL as default source
if [[ -z "$PROJECT_REF" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
  fi
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
fi

TMP_DIR="supabase/.tmp/db-sync"
mkdir -p "$TMP_DIR"
DATA_FILE="$TMP_DIR/remote_data.sql"

echo "Resetting local database (apply local migrations; skip seeds)..."
supabase db reset --force --no-seed >/dev/null

echo "Dumping remote data ($MODE mode)... (this may take a while)"
if [[ "$MODE" == "data" ]]; then
  supabase db dump \
    --project-ref "$PROJECT_REF" \
    --remote \
    --data-only \
    --use-copy \
    -s public,auth,storage,graphql_public \
    -x supabase_migrations.schema_migrations \
    -f "$DATA_FILE"
else
  supabase db dump \
    --project-ref "$PROJECT_REF" \
    --remote \
    --data-only \
    --use-copy \
    -f "$DATA_FILE"
fi

if [[ "$MODE" == "full" ]]; then
  echo "Preparing to replace migration history with remote..."
  COMBINED_FILE="$TMP_DIR/remote_data_with_truncate.sql"
  echo "TRUNCATE TABLE supabase_migrations.schema_migrations;" > "$COMBINED_FILE"
  cat "$DATA_FILE" >> "$COMBINED_FILE"
  echo "Restoring data (including migration history) to local..."
  supabase db restore -f "$COMBINED_FILE"
else
  echo "Restoring data to local (keeping local migration history)..."
  supabase db restore -f "$DATA_FILE"
fi

echo "Done. Local DB now mirrors remote ($PROJECT_REF)."
