#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-scrape:latest

docker build -f scripts/scrape_dmm_reviews/Dockerfile -t "$IMAGE" .

exec docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"

