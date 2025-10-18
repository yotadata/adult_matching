#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-setup-dev-env:latest

docker build -f scripts/setup_dev_env/Dockerfile -t "$IMAGE" .

exec docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" "$@"

