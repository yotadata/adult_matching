#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage: scripts/publish_two_tower/run.sh [--env-file <path>] [--network <name>] <command> [args...]

Commands:
  upload     Upload model artifacts to Supabase Storage.
  activate   Update manifest to point at a specific run_id.
  fetch      Download artifacts referenced by the manifest.
USAGE
  exit 1
fi

IMAGE=${PUBLISH_TWO_TOWER_IMAGE:-adult-matching-publish-two-tower:latest}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

ENV_FILE=${PUBLISH_TWO_TOWER_ENV_FILE:-docker/env/prd.env}
NETWORK=${PUBLISH_TWO_TOWER_NETWORK:-}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --network)
      NETWORK="$2"
      shift 2
      ;;
    upload|activate|fetch)
      POSITIONAL+=("$1")
      shift
      POSITIONAL+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -eq 0 ]]; then
  echo "Error: command (upload|activate|fetch) is required." >&2
  exit 1
fi

COMMAND=${POSITIONAL[0]}
shift 0

docker build -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE" "$REPO_ROOT"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

TMP_ENV_FILE=$(mktemp)
trap 'rm -f "$TMP_ENV_FILE"' EXIT

env_keys=$(grep -E "^[A-Za-z_][A-Za-z0-9_]*=" "$ENV_FILE" | cut -d= -f1)
printed=""
while IFS= read -r key; do
  [[ -z "$key" ]] && continue
  case " $printed " in
    *" $key "*) continue ;;
  esac
  printed+=" $key"
  if [[ -n ${!key+x} ]]; then
    printf "%s=%s\n" "$key" "${!key}" >>"$TMP_ENV_FILE"
  fi
done <<<"$env_keys"

DOCKER_CMD=(docker run --rm)
if [[ -t 1 ]]; then
  DOCKER_CMD+=(-it)
else
  DOCKER_CMD+=(-i)
fi
if [[ -n "$NETWORK" ]]; then
  DOCKER_CMD+=(--network "$NETWORK")
fi
DOCKER_CMD+=(--env-file "$TMP_ENV_FILE" -v "$REPO_ROOT":/workspace -w /workspace "$IMAGE" python scripts/publish_two_tower/publish.py)
DOCKER_CMD+=("${POSITIONAL[@]}")

"${DOCKER_CMD[@]}"
