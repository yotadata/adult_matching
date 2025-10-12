#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_artifacts(directory: Path) -> List[Dict[str, object]]:
    files: List[Dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(directory).as_posix()
            size = path.stat().st_size
            files.append(
                {
                    "path": rel_path,
                    "sha256": compute_sha256(path),
                    "size": size,
                }
            )
    return files


def load_model_meta(directory: Path) -> Dict[str, object] | None:
    meta_path = directory / "model_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manifest.json for Two-Tower artifacts")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--commit-sha", type=str, default=os.environ.get("GITHUB_SHA", ""))
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    files = collect_artifacts(artifacts_dir)
    if not files:
        raise RuntimeError(f"No files found in artifacts directory {artifacts_dir}")

    manifest: Dict[str, object] = {
        "model_version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": args.commit_sha,
        "files": files,
    }

    model_meta = load_model_meta(artifacts_dir)
    if model_meta:
        manifest["model_meta"] = model_meta

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
