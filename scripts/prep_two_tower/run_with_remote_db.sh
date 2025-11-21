#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prep_two_tower/run_with_remote_db.sh [options] [prep_args...]

Description:
  1. Uses Supabase CLI (`supabase db dump --db-url`) to export the required tables from the remote project.
  2. Spins up an ephemeral Postgres container seeded with that dump.
  3. Runs prep_two_tower_dataset.py inside its Dockerコンテナ against the local DB.
  4. Cleans upリソース after completion.

Options:
  --env-file <path>     Env file to load before running (default: docker/env/prd.env).
  --project-ref <ref>   Supabase project ref (informational / future use).
  --remote-db-url <url> Override REMOTE_DATABASE_URL instead of relying on env files.
  -h, --help            Show this help.

Environment variables (via env file or shell):
  REMOTE_DATABASE_URL   Remote Postgres connection string (postgresql://user:pass@host:port/db).
  RUN_ID                Optional identifier for dump directory (default: UTC timestamp).
  DUMP_TABLES           Space separated tables to export (default: "public.videos public.video_tags public.tags public.video_performers").
  LOCAL_DB_NAME         Ephemeral Postgres database name (default: tt_prep).
  LOCAL_DB_USER         Ephemeral Postgres user (default: supabase_admin).
  LOCAL_DB_PASS         Ephemeral Postgres password (default: postgres).
  LOCAL_DB_PORT         Host port to expose Postgres on (default: 6543).
  LOCAL_DB_CONNECT_DB   Database used for readiness checks (default: postgres).
  LOCAL_DB_STARTUP_TIMEOUT  Seconds to wait for Postgres readiness (default: 90).
  POSTGRES_IMAGE        Docker image for ephemeral Postgres (default: supabase/postgres:17.6.1.021).
  POSTGRES_PLATFORM     Docker platform for the image (auto-detected).

Any追加引数 after `--` are forwarded to prep_two_tower_dataset.py.
EOF
}

ENV_FILE="${SUPABASE_ENV_FILE:-docker/env/prd.env}"
PROJECT_REF="${PROJECT_REF:-}"
FORWARD_ARGS=()
REMOTE_DB_URL_ARG="${REMOTE_DB_URL_OVERRIDE:-}"
USE_POOLER_FOR_DUMP="${USE_POOLER_FOR_DUMP:-0}"
USE_SUPABASE_LINK_DUMP="${USE_SUPABASE_LINK_DUMP:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --project-ref)
      PROJECT_REF="$2"
      shift 2
      ;;
    --remote-db-url)
      REMOTE_DB_URL_ARG="$2"
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

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [[ -n "${REMOTE_DB_URL_ARG:-}" ]]; then
  REMOTE_DATABASE_URL="$REMOTE_DB_URL_ARG"
fi

if [[ -z "$PROJECT_REF" && -n "${SUPABASE_PROJECT_ID:-}" ]]; then
  PROJECT_REF="$SUPABASE_PROJECT_ID"
elif [[ -z "$PROJECT_REF" && -n "${SUPABASE_PROJECT_REF:-}" ]]; then
  echo "[WARN] SUPABASE_PROJECT_REF is deprecated; set SUPABASE_PROJECT_ID instead." >&2
  PROJECT_REF="$SUPABASE_PROJECT_REF"
fi

if [[ -z "$PROJECT_REF" ]] && [[ -n "${NEXT_PUBLIC_SUPABASE_URL:-${SUPABASE_URL:-}}" ]]; then
  maybe_ref=$(printf "%s\n" "${NEXT_PUBLIC_SUPABASE_URL:-$SUPABASE_URL}" | sed -E 's#^https?://([^.]+)\.supabase\.co.*#\1#')
  if [[ -n "$maybe_ref" ]]; then
    PROJECT_REF="$maybe_ref"
  fi
fi

if ! command -v supabase >/dev/null 2>&1; then
  echo "[ERROR] Supabase CLI が見つかりません。https://supabase.com/docs/guides/cli を参照してセットアップしてください。" >&2
  exit 1
fi

if [[ -z "${REMOTE_DATABASE_URL:-}" ]]; then
  echo "[ERROR] REMOTE_DATABASE_URL が未設定です。docker/env/prd.env などで定義してください。" >&2
  usage
  exit 1
fi

printf '[DEBUG] Env check: REMOTE_DATABASE_URL=%s\n' "$( [[ -n ${REMOTE_DATABASE_URL:-} ]] && echo set || echo unset )"
printf '[DEBUG] Env check: PGPASSWORD=%s\n' "$( [[ -n ${PGPASSWORD:-} ]] && echo set || echo unset )"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DUMP_TABLES_DEFAULT="public.videos public.video_tags public.tags public.video_performers"
DUMP_TABLES="${DUMP_TABLES:-$DUMP_TABLES_DEFAULT}"

LOCAL_DB_NAME="${LOCAL_DB_NAME:-tt_prep}"
LOCAL_DB_USER="${LOCAL_DB_USER:-supabase_admin}"
LOCAL_DB_PASS="${LOCAL_DB_PASS:-postgres}"
LOCAL_DB_CONNECT_DB="${LOCAL_DB_CONNECT_DB:-postgres}"
LOCAL_DB_STARTUP_TIMEOUT="${LOCAL_DB_STARTUP_TIMEOUT:-90}"
LOCAL_DB_PORT="${LOCAL_DB_PORT:-6543}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-supabase/postgres:17.6.1.021}"
DEFAULT_PLATFORM="linux/amd64"
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
  DEFAULT_PLATFORM="linux/arm64/v8"
fi
POSTGRES_PLATFORM="${POSTGRES_PLATFORM:-$DEFAULT_PLATFORM}"
NETWORK_NAME="tt-prep-net"
DB_CONTAINER="tt-prep-db-$RUN_ID"
PREP_IMAGE="adult-matching-prep:latest"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] python / python3 が見つかりません。REMOTE_DATABASE_URL の解析に必要です。" >&2
  exit 1
fi

TEMP_CA_FILE=""
: "${PGSSLROOTCERT:=}"
: "${SSL_CERT_FILE:=}"

eval "$("$PYTHON_BIN" - <<'PY'
from urllib.parse import urlparse
import os
import shlex

url = os.environ.get("REMOTE_DATABASE_URL", "")
parsed = urlparse(url)

host = parsed.hostname or ""
port = str(parsed.port or 5432)
db = parsed.path.lstrip("/") or "postgres"
user = parsed.username or ""
password = parsed.password or ""

print(f"REMOTE_DB_HOST={shlex.quote(host)}")
print(f"REMOTE_DB_PORT={shlex.quote(port)}")
print(f"REMOTE_DB_NAME={shlex.quote(db)}")
print(f"REMOTE_DB_USER={shlex.quote(user)}")
print(f"REMOTE_DB_PASS={shlex.quote(password)}")
PY
)"

if [[ -z "${REMOTE_DB_HOST:-}" || -z "${REMOTE_DB_USER:-}" ]]; then
  echo "[ERROR] REMOTE_DATABASE_URL からホスト/ユーザーを特定できません。" >&2
  exit 1
fi

if [[ -z "${REMOTE_DB_PASS:-}" ]]; then
  if [[ -n "${PGPASSWORD:-}" ]]; then
    REMOTE_DB_PASS="$PGPASSWORD"
  else
    echo "[ERROR] REMOTE_DATABASE_URL にパスワードが含まれていません (PGPASSWORD も未設定)。" >&2
    exit 1
  fi
fi

SQL_DB_URL="postgresql://${REMOTE_DB_USER}:${REMOTE_DB_PASS}@${REMOTE_DB_HOST}:${REMOTE_DB_PORT}/${REMOTE_DB_NAME}"
if [[ "$SQL_DB_URL" != *"pgbouncer="* ]]; then
  SQL_DB_URL="${SQL_DB_URL}?pgbouncer=false"
else
  SQL_DB_URL="${SQL_DB_URL}"
fi
if [[ "$SQL_DB_URL" != *"sslmode="* ]]; then
  if [[ "$SQL_DB_URL" == *"?"* ]]; then
    SQL_DB_URL="${SQL_DB_URL}&sslmode=require"
  else
    SQL_DB_URL="${SQL_DB_URL}?sslmode=require"
  fi
fi

# Optional pooler fallback for IPv4-only environments
if [[ "${USE_POOLER_FOR_DUMP}" == "1" ]]; then
  if [[ -z "$PROJECT_REF" || -z "${SUPABASE_REGION:-}" ]]; then
    echo "[WARN] USE_POOLER_FOR_DUMP=1 ですが PROJECT_REF または SUPABASE_REGION が未設定のためプーラー URL を構築できません。" >&2
  else
    POOLER_HOST="${SUPABASE_REGION}.pooler.supabase.com"
    POOLER_PORT="${SUPABASE_POOLER_PORT:-6543}"
    POOL_USER="$REMOTE_DB_USER"
    if [[ "$POOL_USER" != *.* ]]; then
      POOL_USER="${POOL_USER}.${PROJECT_REF}"
    fi
    RAW_URL="$SQL_DB_URL"
    SQL_DB_URL="postgresql://${POOL_USER}:${REMOTE_DB_PASS}@${POOLER_HOST}:${POOLER_PORT}/${REMOTE_DB_NAME}"
    if [[ "$SQL_DB_URL" != *"sslmode="* ]]; then
      if [[ "$SQL_DB_URL" == *"?"* ]]; then
        SQL_DB_URL="${SQL_DB_URL}&sslmode=require"
      else
        SQL_DB_URL="${SQL_DB_URL}?sslmode=require"
      fi
    fi
    echo "[INFO] USE_POOLER_FOR_DUMP=1: ${RAW_URL} -> ${SQL_DB_URL}"
  fi
fi

if [[ -z "${PGSSLROOTCERT:-}" ]]; then
  if [[ "${SKIP_SSL_PROBE:-}" == "1" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "[WARN] Skipping SSL cert probe (SKIP_SSL_PROBE=1 or CI detected); relying on system CAs." >&2
  else
    TEMP_CA_FILE=$(mktemp)
    if "$PYTHON_BIN" - "$REMOTE_DB_HOST" "$REMOTE_DB_PORT" > "$TEMP_CA_FILE" <<'PY'
import ssl, socket, sys

host = sys.argv[1]
port = int(sys.argv[2])
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
with socket.create_connection((host, port)) as sock:
    with ctx.wrap_socket(sock, server_hostname=host) as ssock:
        der = ssock.getpeercert(True)
pem = ssl.DER_cert_to_PEM_cert(der)
sys.stdout.write(pem)
PY
    then
      PGSSLROOTCERT="$TEMP_CA_FILE"
      echo "[INFO] REMOTE_DB_HOST のサーバー証明書を取得し、PGSSLROOTCERT=$PGSSLROOTCERT に保存しました。"
    else
      echo "[WARN] SSL cert probe failed; continuing without PGSSLROOTCERT (sslmode=require)." >&2
      if [[ -n "$TEMP_CA_FILE" && -f "$TEMP_CA_FILE" ]]; then
        rm -f "$TEMP_CA_FILE"
        TEMP_CA_FILE=""
      fi
    fi
  fi
fi

: "${PGSSLROOTCERT:=}"
: "${SSL_CERT_FILE:=}"

if [[ -n "${SSL_CERT_FILE:-}" && ! -f "$SSL_CERT_FILE" ]]; then
  echo "[WARN] SSL_CERT_FILE=$SSL_CERT_FILE が存在しません。PGSSLROOTCERT を共有します。" >&2
  SSL_CERT_FILE=""
fi

if [[ -z "${SSL_CERT_FILE:-}" ]]; then
SSL_CERT_FILE="$PGSSLROOTCERT"
fi

DUMP_DIR="$REPO_ROOT/ml/data/raw/db_dumps/$RUN_ID"
mkdir -p "$DUMP_DIR"
SCHEMA_SQL="$DUMP_DIR/prep_dump.schema.sql"
DATA_SQL="$DUMP_DIR/prep_dump.data.sql"
FILTERED_SCHEMA_SQL="$DUMP_DIR/prep_dump.schema.filtered.sql"
FILTERED_DATA_SQL="$DUMP_DIR/prep_dump.data.filtered.sql"

cleanup() {
  echo "[INFO] Cleaning up Postgres container..."
  docker rm -f "$DB_CONTAINER" > /dev/null 2>&1 || true
  if [[ -n "$TEMP_CA_FILE" && -f "$TEMP_CA_FILE" ]]; then
    rm -f "$TEMP_CA_FILE"
  fi
}
trap cleanup EXIT

require_ipv6() {
  if [[ "${ALLOW_IPV4_FALLBACK:-}" == "1" || "${USE_POOLER_FOR_DUMP:-0}" == "1" ]]; then
    return 0
  fi
  if "$PYTHON_BIN" - "$REMOTE_DB_HOST" "$REMOTE_DB_PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
try:
    infos = socket.getaddrinfo(host, port, socket.AF_INET6, socket.SOCK_STREAM)
    if not infos:
        sys.exit(1)
    # Try a short connect to surface "network unreachable" early
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.settimeout(3)
    sock.connect(infos[0][4])
    sock.close()
sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    return 0
  fi
  cat >&2 <<'MSG'
[ERROR] IPv6 で Supabase Postgres へ到達できません。GitHub Hosted Runner など IPv6 非対応環境では失敗します。
- IPv6 が有効な Self-hosted Runner で実行する
- もしくは USE_POOLER_FOR_DUMP=1 でプーラー経由の IPv4 を使う
- あるいは ALLOW_IPV4_FALLBACK=1 を指定し、IPv4 で到達可能な環境のみで実行する
MSG
  exit 1
}

if [[ "${USE_SUPABASE_LINK_DUMP:-0}" == "1" && -n "${SUPABASE_ACCESS_TOKEN:-}" && -n "$PROJECT_REF" ]]; then
  echo "[INFO] ダンプモード: supabase link + supabase db dump --linked"
else
  require_ipv6
  if [[ "${USE_POOLER_FOR_DUMP:-0}" == "1" ]]; then
    echo "[INFO] ダンプモード: pooler 経由 (IPv4 想定)"
  else
    echo "[INFO] ダンプモード: direct DB 接続 (IPv6 必須)"
  fi
fi

SCHEMAS_ARG=(--schema public)

echo "[INFO] Supabase CLI で schema dump を取得します..."
if [[ "${USE_SUPABASE_LINK_DUMP:-0}" == "1" && -n "${SUPABASE_ACCESS_TOKEN:-}" && -n "$PROJECT_REF" ]]; then
  SUPABASE_ACCESS_TOKEN="$SUPABASE_ACCESS_TOKEN" supabase link --project-ref "$PROJECT_REF" --password "$REMOTE_DB_PASS" --workdir "$REPO_ROOT/supabase"
  SUPABASE_ACCESS_TOKEN="$SUPABASE_ACCESS_TOKEN" supabase db dump --linked --schema public -f "$SCHEMA_SQL" --workdir "$REPO_ROOT/supabase"
else
  PGPASSWORD="$REMOTE_DB_PASS" PGSSLROOTCERT="$PGSSLROOTCERT" SSL_CERT_FILE="$SSL_CERT_FILE" \
    supabase db dump --db-url "$SQL_DB_URL" "${SCHEMAS_ARG[@]}" -f "$SCHEMA_SQL"
fi

echo "[INFO] Supabase CLI で data dump を取得します..."
if [[ "${USE_SUPABASE_LINK_DUMP:-0}" == "1" && -n "${SUPABASE_ACCESS_TOKEN:-}" && -n "$PROJECT_REF" ]]; then
  SUPABASE_ACCESS_TOKEN="$SUPABASE_ACCESS_TOKEN" supabase db dump --linked --schema public --data-only -f "$DATA_SQL" --workdir "$REPO_ROOT/supabase"
else
  PGPASSWORD="$REMOTE_DB_PASS" PGSSLROOTCERT="$PGSSLROOTCERT" SSL_CERT_FILE="$SSL_CERT_FILE" \
    supabase db dump --db-url "$SQL_DB_URL" "${SCHEMAS_ARG[@]}" --data-only -f "$DATA_SQL"
fi

echo "[INFO] Schema dump をフィルタリングしています (functions/auth/storage を除外)..."
"$PYTHON_BIN" - "$SCHEMA_SQL" "$FILTERED_SCHEMA_SQL" <<'PY'
import sys
import re

source, target = sys.argv[1], sys.argv[2]

dollar_pattern = re.compile(r'\$[A-Za-z0-9_]*\$')

def iter_statements(lines):
    statement = []
    in_dollar = None
    copy_mode = False
    for line in lines:
        statement.append(line)
        stripped = line.strip()

        if copy_mode:
            if stripped == r'\.':
                copy_mode = False
                yield ''.join(statement)
                statement = []
            continue

        if in_dollar:
            if in_dollar in line:
                remainder = line.split(in_dollar, 1)[1]
                if ';' in remainder:
                    in_dollar = None
                    yield ''.join(statement)
                    statement = []
            continue

        match = dollar_pattern.search(line)
        if match:
            in_dollar = match.group(0)
            remainder = line.split(in_dollar, 1)[1]
            if ';' in remainder:
                in_dollar = None
                yield ''.join(statement)
                statement = []
            continue

        upper = stripped.upper()
        if upper.startswith('COPY '):
            copy_mode = True
            continue

        if ';' in line:
            yield ''.join(statement)
            statement = []

    if statement:
        yield ''.join(statement)

skip_patterns = [
    re.compile(r'\bCREATE\s+(OR\s+REPLACE\s+)?FUNCTION\b', re.IGNORECASE),
    re.compile(r'\bALTER\s+FUNCTION\b', re.IGNORECASE),
    re.compile(r'\bCOMMENT\s+ON\s+FUNCTION\b', re.IGNORECASE),
    re.compile(r'\bGRANT\b.+\bFUNCTION\b', re.IGNORECASE),
]
skip_keywords = ('auth.', 'auth"', 'storage.')

with open(source, encoding='utf-8') as f:
    statements = list(iter_statements(f))

filtered = []
for stmt in statements:
    upper = stmt.upper()
    lower = stmt.lower()
    normalized = lower.replace('"', '')
    if any(pattern.search(upper) for pattern in skip_patterns):
        continue
    if any(keyword in lower or keyword in normalized for keyword in skip_keywords):
        continue
    filtered.append(stmt)

with open(target, 'w', encoding='utf-8') as out:
    out.write(''.join(filtered))
PY

echo "[INFO] Data dump をフィルタリングしています (auth/storage を除外)..."
"$PYTHON_BIN" - "$DATA_SQL" "$FILTERED_DATA_SQL" <<'PY'
import sys

source, target = sys.argv[1], sys.argv[2]

skip_keywords = ('auth.', 'auth"', 'storage.')

with open(source, encoding='utf-8') as f:
    statements = f.read().splitlines()

filtered_lines = []
for line in statements:
    lowered = line.lower()
    normalized = lowered.replace('"', '')
    if any(keyword in lowered or keyword in normalized for keyword in skip_keywords):
        continue
    filtered_lines.append(line)

with open(target, 'w', encoding='utf-8') as out:
    out.write('\n'.join(filtered_lines))
PY

if [[ ! -s "$FILTERED_SCHEMA_SQL" ]]; then
  echo "[ERROR] フィルタ後の schema SQL が空です: $FILTERED_SCHEMA_SQL" >&2
  exit 1
fi

if [[ ! -s "$FILTERED_DATA_SQL" ]]; then
  echo "[ERROR] フィルタ後の data SQL が空です: $FILTERED_DATA_SQL" >&2
  exit 1
fi

if ! grep -q '"public"."profiles"' "$FILTERED_SCHEMA_SQL"; then
  cat >&2 <<'MSG'
[ERROR] profiles テーブルがダンプに含まれていません。
- Supabase CLI 権限/設定を見直し、ダンプ対象に public.profiles（auth.users を参照するビュー/テーブル）が含まれるようにしてください。
- GitHub Actions から実行する場合、SUPABASE_ACCESS_TOKEN の権限（DB読み取り）を確認してください。
- どうしても含められない場合は、prep 用に profiles の代替テーブルを用意するなど再現性が担保できる形で対応してください。
MSG
  exit 1
fi

echo "[INFO] Creating docker network $NETWORK_NAME (if not exists)..."
if ! docker network inspect "$NETWORK_NAME" > /dev/null 2>&1; then
  docker network create "$NETWORK_NAME" > /dev/null
fi

echo "[INFO] Starting ephemeral Postgres container $DB_CONTAINER..."
docker run -d \
  --platform "$POSTGRES_PLATFORM" \
  --name "$DB_CONTAINER" \
  --network "$NETWORK_NAME" \
  -e POSTGRES_USER="$LOCAL_DB_USER" \
  -e POSTGRES_PASSWORD="$LOCAL_DB_PASS" \
  -e POSTGRES_DB="$LOCAL_DB_CONNECT_DB" \
  -p "${LOCAL_DB_PORT}:5432" \
  "$POSTGRES_IMAGE" > /dev/null

echo "[INFO] Waiting for Postgres to become ready..."
tries=0
until docker exec "$DB_CONTAINER" env PGPASSWORD="$LOCAL_DB_PASS" pg_isready -h 127.0.0.1 -p 5432 -U "$LOCAL_DB_USER" -d "$LOCAL_DB_CONNECT_DB" > /dev/null 2>&1; do
  sleep 1
  tries=$((tries+1))
  if [[ $tries -gt $LOCAL_DB_STARTUP_TIMEOUT ]]; then
    echo "[ERROR] Postgres container did not become ready in time" >&2
    docker logs "$DB_CONTAINER" 2>&1 || true
    exit 1
  fi
done

# Give the server a brief moment before applying migrations, to avoid race conditions
sleep 2

if [[ "$LOCAL_DB_NAME" != "$LOCAL_DB_CONNECT_DB" ]]; then
  db_exists=$(docker exec "$DB_CONTAINER" \
    env PGPASSWORD="$LOCAL_DB_PASS" \
    psql --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_CONNECT_DB" \
         -At -c "SELECT 1 FROM pg_database WHERE datname = '$LOCAL_DB_NAME'" | tr -d '[:space:]')
  if [[ "$db_exists" != "1" ]]; then
    docker exec "$DB_CONTAINER" \
      env PGPASSWORD="$LOCAL_DB_PASS" \
      psql --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_CONNECT_DB" \
           -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"$LOCAL_DB_NAME\""
  fi
fi

docker exec "$DB_CONTAINER" \
  env PGPASSWORD="$LOCAL_DB_PASS" \
  psql --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_NAME" \
       -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION IF NOT EXISTS vector' >/dev/null

docker exec "$DB_CONTAINER" \
  env PGPASSWORD="$LOCAL_DB_PASS" \
  psql --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_NAME" \
       -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION IF NOT EXISTS pg_trgm' >/dev/null

# Fallback: ensure profiles table exists when dump omits it (e.g., link/permissions filtered)
docker exec "$DB_CONTAINER" \
  env PGPASSWORD="$LOCAL_DB_PASS" \
  psql --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_NAME" \
       -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE IF NOT EXISTS public.profiles (
  user_id uuid PRIMARY KEY,
  display_name text,
  created_at timestamp with time zone DEFAULT now()
);
SQL

# Ensure auxiliary functions required by downstream schema exist (some dumps may reference them even if not present)
docker exec "$DB_CONTAINER" \
  env PGPASSWORD="$LOCAL_DB_PASS" \
  psql --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_NAME" <<'SQL'
CREATE OR REPLACE FUNCTION public.set_ai_recommend_playlists_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;
SQL

vector_half_support=$(docker exec "$DB_CONTAINER" \
  env PGPASSWORD="$LOCAL_DB_PASS" \
  psql --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_NAME" \
       -At -c "SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = 'public' AND p.proname = 'halfvec_in' LIMIT 1" \
  | tr -d '[:space:]')

if [[ "$vector_half_support" != "1" ]]; then
  echo "[ERROR] pgvector の halfvec 型が利用できません。POSTGRES_IMAGE=${POSTGRES_IMAGE} が pgvector 0.5 以降を含むことを確認してください。" >&2
  echo "[HINT] 例: POSTGRES_IMAGE=supabase/postgres:17.6.1.021 bash scripts/prep_two_tower/run_with_remote_db.sh ..." >&2
  exit 1
fi

echo "[INFO] Applying schema to local Postgres..."
cat "$FILTERED_SCHEMA_SQL" | docker exec -i "$DB_CONTAINER" \
  env PGPASSWORD="$LOCAL_DB_PASS" \
  psql -v ON_ERROR_STOP=1 --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_NAME"

echo "[INFO] Loading data into local Postgres..."
cat "$FILTERED_DATA_SQL" | docker exec -i "$DB_CONTAINER" \
  env PGPASSWORD="$LOCAL_DB_PASS" \
  psql --single-transaction -v ON_ERROR_STOP=1 --host=127.0.0.1 --port=5432 --username "$LOCAL_DB_USER" --dbname "$LOCAL_DB_NAME"

echo "[INFO] Building prep image..."
docker build -f "$SCRIPT_DIR/Dockerfile" -t "$PREP_IMAGE" "$REPO_ROOT"

DB_URL_FOR_PREP="postgresql://${LOCAL_DB_USER}:${LOCAL_DB_PASS}@${DB_CONTAINER}:5432/${LOCAL_DB_NAME}"

echo "[INFO] Running prep script against local Postgres..."
docker run --rm -it \
  --network "$NETWORK_NAME" \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  "$PREP_IMAGE" \
  python scripts/prep_two_tower/prep_two_tower_dataset.py \
  --db-url "$DB_URL_FOR_PREP" "${FORWARD_ARGS[@]}"

echo "[INFO] Prep completed successfully (run_id=$RUN_ID)"
