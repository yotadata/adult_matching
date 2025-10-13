#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-gen-keys:latest

docker build -f scripts/gen_supabase_keys/Dockerfile -t "$IMAGE" .

exec docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE"

