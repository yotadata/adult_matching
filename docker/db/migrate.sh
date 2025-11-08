#!/usr/bin/env sh
set -eu

DB_HOST="${DB_HOST:-db}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-postgres}"

echo "Waiting for database ${DB_HOST}:${DB_PORT} ..."
pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -t 1 >/dev/null 2>&1 || true
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; do
  sleep 1
done

psql() {
  command psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 "$@"
}

# Prefer Supabase CLI's tracking table if present; otherwise fall back to local tracking
HAS_SUPABASE_TABLE=$(psql -At -c "select 1 from information_schema.tables where table_schema='supabase_migrations' and table_name='schema_migrations' limit 1;" || true)
if [ -z "$HAS_SUPABASE_TABLE" ]; then
  # Local tracking table (non-intrusive; lives beside official table if it appears later)
  psql -c "create schema if not exists supabase_migrations;"
  psql -c "create table if not exists supabase_migrations.local_sql_migrations (filename text primary key, checksum text not null, applied_at timestamptz not null default now());"
fi

# Ensure minimal roles exist so migrations that refer to them don't fail
psql <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN CREATE ROLE service_role NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_admin') THEN CREATE ROLE supabase_admin NOLOGIN; END IF;
END $$;
SQL

# Allow reading server files for extension pre/post hooks in Supabase image
psql -c "GRANT pg_read_server_files TO \"$DB_USER\";" || true

# Provide an extension-free UUID v4 generator for environments where CREATE EXTENSION is restricted
psql <<'SQL'
CREATE OR REPLACE FUNCTION public.uuid_v4()
RETURNS uuid
LANGUAGE SQL
IMMUTABLE
AS $$
  SELECT (
    lpad(to_hex((random()*4294967295)::int),8,'0') || '-' ||
    lpad(to_hex((random()*65535)::int),4,'0') || '-' ||
    lpad(to_hex((random()*65535)::int),4,'0') || '-' ||
    lpad(to_hex((random()*65535)::int),4,'0') || '-' ||
    lpad(to_hex((random()*281474976710655)::bigint),12,'0')
  )::uuid;
$$;
SQL

VECTOR_EXTENSION_PRESENT=$(psql -At -c "select 1 from pg_extension where extname = 'vector' limit 1;" || true)

transform_sql() {
  src="$1"
  dest="$2"
  if [ -z "$VECTOR_EXTENSION_PRESENT" ]; then
    sed -r \
      -e 's/^\s*create\s+extension\s+if\s+not\s+exists.*$//' \
      -e 's/\bgen_random_uuid\s*\(\s*\)/public.uuid_v4()/g' \
      -e 's/\buuid_generate_v4\s*\(\s*\)/public.uuid_v4()/g' \
      -e 's/\bvector\s*\([0-9]+\)/double precision[]/g' \
      -e '/gin_trgm_ops/d' \
      -e '/^CREATE INDEX IF NOT EXISTS idx_video_embeddings_cosine/d' \
      -e '/^CREATE INDEX IF NOT EXISTS idx_user_embeddings_cosine/d' \
      -e '/ON public\.video_embeddings USING ivfflat/d' \
      -e '/ON public\.user_embeddings USING ivfflat/d' \
      "$src" > "$dest"
  else
    sed -r \
      -e 's/\bgen_random_uuid\s*\(\s*\)/public.uuid_v4()/g' \
      -e 's/\buuid_generate_v4\s*\(\s*\)/public.uuid_v4()/g' \
      "$src" > "$dest"
  fi
}

calc_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    # Fallback to openssl
    openssl dgst -sha256 -r "$1" | awk '{print $1}'
  fi
}

MIG_DIR="/migrations"
if [ ! -d "$MIG_DIR" ]; then
  echo "No migrations directory mounted at $MIG_DIR; nothing to do."
  exit 0
fi

EXIT_CODE=0
for f in "$MIG_DIR"/*.sql; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  sum=$(calc_sha256 "$f")

  if [ -n "$HAS_SUPABASE_TABLE" ]; then
    # Use supabase_migrations.schema_migrations based on version from filename prefix
    ver=$(printf "%s" "$base" | sed -E 's/^([0-9]+).*/\1/')
    if [ -z "$ver" ]; then
      echo "WARNING: $base has no numeric version prefix; falling back to local tracking."
    fi

    if [ -n "$ver" ]; then
      applied=$(psql -At -c "select 1 from supabase_migrations.schema_migrations where version::text = '$ver' limit 1;" || true)
      if [ -n "$applied" ]; then
        echo "Skipping $base (version $ver already recorded in supabase_migrations.schema_migrations)"
        continue
      fi

      echo "Applying $base (version $ver) ..."
      tmpf=$(mktemp)
      transform_sql "$f" "$tmpf"
      if psql -f "$tmpf"; then
        # Detect optional name column
        has_name=$(psql -At -c "select 1 from information_schema.columns where table_schema='supabase_migrations' and table_name='schema_migrations' and column_name='name' limit 1;" || true)
        if [ -n "$has_name" ]; then
          psql -c "insert into supabase_migrations.schema_migrations(version, name) values ('$ver', '$base');"
        else
          psql -c "insert into supabase_migrations.schema_migrations(version) values ('$ver');"
        fi
        echo "Applied $base"
        rm -f "$tmpf"
      else
        echo "--- Failed SQL (transformed) from: $base ---" >&2
        sed -n '1,120p' "$tmpf" >&2 || true
        rm -f "$tmpf"
        echo "ERROR applying $base" >&2
        EXIT_CODE=1
        break
      fi
      continue
    fi
  fi

  # Local tracking path (used when official table absent or no version prefix)
  existing=$(psql -At -c "select checksum from supabase_migrations.local_sql_migrations where filename = '$base';" || true)
  if [ -z "$existing" ]; then
    echo "Applying $base ..."
    tmpf=$(mktemp)
    transform_sql "$f" "$tmpf"
    if psql -f "$tmpf"; then
      psql -c "insert into supabase_migrations.local_sql_migrations(filename, checksum) values ('$base', '$sum');"
      echo "Applied $base"
      rm -f "$tmpf"
    else
      echo "--- Failed SQL (transformed) from: $base ---" >&2
      sed -n '1,120p' "$tmpf" >&2 || true
      rm -f "$tmpf"
      echo "ERROR applying $base" >&2
      EXIT_CODE=1
      break
    fi
  elif [ "$existing" = "$sum" ]; then
    echo "Skipping $base (already applied)"
  else
    echo "ERROR: checksum mismatch for $base. Previously: $existing, Now: $sum" >&2
    echo "Refuse to re-apply changed migration. Create a new migration file instead." >&2
    EXIT_CODE=1
    break
  fi
done

exit $EXIT_CODE
