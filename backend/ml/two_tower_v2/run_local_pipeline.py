#!/usr/bin/env python3
"""Convenience wrapper that runs the local Two-Tower training pipeline end-to-end.

Steps:
  1. Train the model (src/train.py)
  2. Export ONNX + artifacts (src/export.py)
  3. Generate item embeddings parquet (src/generate_embeddings.py)

All commands are executed with the current Python interpreter, so activate the
virtualenv (if any) before running this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data_processing" / "local_compatible_data"


def run_subprocess(cmd: Sequence[str], cwd: Path) -> None:
    print(f"→ Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Two-Tower local training pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--profiles", type=Path, default=DEFAULT_DATA_ROOT / "profiles.json")
    parser.add_argument("--videos", type=Path, default=DEFAULT_DATA_ROOT / "videos_subset.json")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DATA_ROOT / "user_video_decisions.json")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/video_embeddings.parquet"))
    parser.add_argument("--use-remote", action="store_true", help="Fetch data from Supabase before training")
    parser.add_argument(
        "--remote-output",
        type=Path,
        default=Path("tmp/remote_data"),
        help="Directory to store fetched remote data",
    )
    parser.add_argument(
        "--remote-page-size",
        type=int,
        default=1000,
        help="Page size when fetching remote data",
    )
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip item embedding export")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX + artifact export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    profiles_path = Path(args.profiles)
    videos_path = Path(args.videos)
    decisions_path = Path(args.decisions)

    if args.use_remote:
        remote_dir = args.remote_output
        remote_dir.mkdir(parents=True, exist_ok=True)
        fetch_cmd = [
            sys.executable,
            "src/fetch_remote_data.py",
            "--output-dir",
            str(remote_dir),
            "--page-size",
            str(args.remote_page_size),
        ]
        run_subprocess(fetch_cmd, cwd=project_root)
        profiles_path = remote_dir / "profiles.json"
        videos_path = remote_dir / "videos_subset.json"
        decisions_path = remote_dir / "user_video_decisions.json"

    train_cmd = [
        sys.executable,
        "src/train.py",
        "--config",
        str(args.config),
        "--profiles",
        str(profiles_path),
        "--videos",
        str(videos_path),
        "--decisions",
        str(decisions_path),
        "--output-dir",
        str(Path(args.checkpoint).parent),
    ]
    run_subprocess(train_cmd, cwd=project_root)

    if not args.skip_export:
        export_cmd = [
            sys.executable,
            "src/export.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(args.artifacts),
        ]
        run_subprocess(export_cmd, cwd=project_root)

    if not args.skip_embeddings:
        embeddings_cmd = [
            sys.executable,
            "src/generate_embeddings.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--videos",
            str(args.videos),
            "--output",
            str(args.embeddings),
        ]
        run_subprocess(embeddings_cmd, cwd=project_root)

    print("✅ Pipeline completed")


if __name__ == "__main__":
    main()
