#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-sync-db:latest

docker build -f scripts/sync_remote_db/Dockerfile -t "$IMAGE" .

exec docker run --rm -it \
  -v "$(pwd)":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace \
  "$IMAGE" "$@"

