#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-train:latest

docker build -f scripts/train_two_tower/Dockerfile -t "$IMAGE" .

DOCKER_FLAGS=(--rm)
if [ -t 1 ]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi

exec docker run "${DOCKER_FLAGS[@]}" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"
