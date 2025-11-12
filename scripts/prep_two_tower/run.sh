#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-prep:latest

DOCKER_SCAN_SUGGEST=false docker build -f scripts/prep_two_tower/Dockerfile -t "$IMAGE" .

exec docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"
