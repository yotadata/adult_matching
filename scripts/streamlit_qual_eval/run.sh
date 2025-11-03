#!/usr/bin/env bash
set -euo pipefail

IMAGE=adult-matching-streamlit:latest

docker build -f scripts/streamlit_qual_eval/Dockerfile -t "$IMAGE" .

DOCKER_FLAGS=(--rm -p 8501:8501)
if [ -t 1 ]; then
  DOCKER_FLAGS+=(-it)
else
  DOCKER_FLAGS+=(-i)
fi

exec docker run "${DOCKER_FLAGS[@]}" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "$IMAGE" \
  streamlit run scripts/streamlit_qual_eval/app.py --server.address 0.0.0.0 --server.port 8501
