#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ml_weekly_pipeline/run.sh [options]

Runs the same steps as the weekly GitHub Actions pipeline (ml-train-model → ml-release-embeddings).
All heavy processing happens inside the existing Dockerized scripts.

Options:
  --env-file <path>       Environment file with DB/Supabase secrets (default: docker/env/dev.env)
  --run-id <id>           Override RUN_ID used across steps (default: UTC timestamp)
  --mode <name>           Data prep mode forwarded to prep_two_tower (default: explicit)
  --min-stars <value>     Minimum star rating treated as positive (default: 4)
  --neg-per-pos <value>   Negative sampling ratio (default: 3)
  --lookback-days <n>     Video embedding sync lookback window (default: 3)
  --recent-hours <n>      User embedding sync recency window (default: 8)
  --train-only            Run prep/train/eval only (skip release phase)
  --release-only          Run release phase only (skip prep/train/eval)
  -h, --help              Show this help.
EOF
}

ENV_FILE="${SUPABASE_ENV_FILE:-$REPO_ROOT/docker/env/dev.env}"
RUN_ID_OVERRIDE=""
DATA_MODE="explicit"
MIN_STARS="4"
NEG_PER_POS="3"
LOOKBACK_DAYS="3"
RECENT_HOURS="8"
RUN_TRAIN=true
RUN_RELEASE=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID_OVERRIDE="$2"
      shift 2
      ;;
    --mode)
      DATA_MODE="$2"
      shift 2
      ;;
    --min-stars)
      MIN_STARS="$2"
      shift 2
      ;;
    --neg-per-pos)
      NEG_PER_POS="$2"
      shift 2
      ;;
    --lookback-days)
      LOOKBACK_DAYS="$2"
      shift 2
      ;;
    --recent-hours)
      RECENT_HOURS="$2"
      shift 2
      ;;
    --train-only)
      RUN_TRAIN=true
      RUN_RELEASE=false
      shift
      ;;
    --release-only)
      RUN_TRAIN=false
      RUN_RELEASE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ml-weekly] Env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID=$(date -u +'%Y%m%dT%H%M%SZ')
fi

echo "[ml-weekly] Using env file: $ENV_FILE"
echo "[ml-weekly] RUN_ID: $RUN_ID"
echo "[ml-weekly] Train phase: $RUN_TRAIN  / Release phase: $RUN_RELEASE"

cd "$REPO_ROOT"

if [[ "$RUN_TRAIN" == "true" ]]; then
  echo "[ml-weekly] Step 1/3: Preparing dataset"
  RUN_ID="$RUN_ID" \
  bash "$REPO_ROOT/scripts/prep_two_tower/run_with_remote_db.sh" \
    --env-file "$ENV_FILE" \
    --mode "$DATA_MODE" \
    --input ml/data/raw/reactions.csv \
    --run-id "$RUN_ID" \
    --min-stars "$MIN_STARS" \
    --neg-per-pos "$NEG_PER_POS"

  echo "[ml-weekly] Step 2/3: Training TwoTower"
  bash "$REPO_ROOT/scripts/train_two_tower/run.sh" \
    --run-id "$RUN_ID" \
    --train ml/data/processed/two_tower/latest/interactions_train.parquet \
    --val ml/data/processed/two_tower/latest/interactions_val.parquet \
    --user-features ml/data/processed/two_tower/latest/user_features.parquet \
    --item-features ml/data/processed/two_tower/latest/item_features.parquet \
    --item-key video_id \
    --embedding-dim 128 \
    --hidden-dim 512 \
    --epochs 5 \
    --batch-size 2048 \
    --lr 1e-3

  echo "[ml-weekly] Step 3/3: Evaluating model"
  bash "$REPO_ROOT/scripts/eval_two_tower/run.sh" \
    --run-id "$RUN_ID" \
    --val ml/data/processed/two_tower/latest/interactions_val.parquet \
    --recall-k 20
fi

if [[ "$RUN_RELEASE" == "true" ]]; then
  echo "[ml-weekly] Release phase: syncing video embeddings"
  bash "$REPO_ROOT/scripts/sync_video_embeddings/run.sh" \
    --env-file "$ENV_FILE" \
    --lookback-days "$LOOKBACK_DAYS"

  echo "[ml-weekly] Release phase: syncing user embeddings"
  bash "$REPO_ROOT/scripts/sync_user_embeddings/run.sh" \
    --env-file "$ENV_FILE" \
    --run-id "$RUN_ID" \
    --recent-hours "$RECENT_HOURS"

  echo "[ml-weekly] Release phase: uploading and activating manifest"
  bash "$REPO_ROOT/scripts/publish_two_tower/run.sh" \
    --env-file "$ENV_FILE" \
    upload \
    --run-id "$RUN_ID"

  bash "$REPO_ROOT/scripts/publish_two_tower/run.sh" \
    --env-file "$ENV_FILE" \
    activate \
    --run-id "$RUN_ID"
fi

echo "[ml-weekly] Completed."
