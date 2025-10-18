#!/usr/bin/env bash
set -euo pipefail

# Prefer host-installed Supabase CLI to avoid pulling container images
if command -v supabase >/dev/null 2>&1; then
  exec bash scripts/sync_remote_db/sync_remote_db.sh "$@"
fi

echo "Error: supabase CLI not found on host, and containerized runner is disabled due to registry access." >&2
echo "Install CLI (macOS): brew install supabase/tap/supabase" >&2
echo "Docs: https://supabase.com/docs/guides/cli" >&2
exit 1
