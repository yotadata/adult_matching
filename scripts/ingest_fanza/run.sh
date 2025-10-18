#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-ingest-fanza:latest

docker build -f scripts/ingest_fanza/Dockerfile -t "$IMAGE" .

exec docker run --rm -it \
  --env-file docker/env/dev.env \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"
