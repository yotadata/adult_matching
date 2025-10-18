#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-train:latest

docker build -f scripts/train_two_tower/Dockerfile -t "$IMAGE" .

exec docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"

