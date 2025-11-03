#!/usr/bin/env python3
import argparse
import datetime as dt
import gzip
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class StorageClient:
    def __init__(self, base_url: str, service_key: str, bucket: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_key = service_key
        self.bucket = bucket
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.service_key}",
            }
        )

    def upload_file(self, local_path: Path, object_path: str, content_type: str) -> None:
        url = f"{self.base_url}/storage/v1/object/{self.bucket}/{object_path.lstrip('/')}"
        headers = {
            "x-upsert": "true",
            "Content-Type": content_type,
        }
        with local_path.open("rb") as fh:
            data = fh.read()
        resp = self.session.post(url, headers=headers, data=data, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"Upload failed for {object_path}: {resp.status_code} {resp.text}")

    def download_json(self, object_path: str) -> Optional[Dict]:
        url = f"{self.base_url}/storage/v1/object/{self.bucket}/{object_path.lstrip('/')}"
        resp = self.session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 400:
            try:
                payload = resp.json()
            except ValueError:
                payload = None
            if payload and str(payload.get("statusCode")) == "404":
                return None
        if resp.status_code >= 400:
            raise RuntimeError(f"Download failed for {object_path}: {resp.status_code} {resp.text}")
        return resp.json()

    def upload_json(self, object_path: str, data: Dict) -> None:
        url = f"{self.base_url}/storage/v1/object/{self.bucket}/{object_path.lstrip('/')}"
        headers = {
            "x-upsert": "true",
            "Content-Type": "application/json",
        }
        resp = self.session.post(url, headers=headers, data=json.dumps(data, ensure_ascii=False).encode("utf-8"), timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"Upload failed for {object_path}: {resp.status_code} {resp.text}")

    def upload_content(self, data: bytes, object_path: str, content_type: str) -> None:
        url = f"{self.base_url}/storage/v1/object/{self.bucket}/{object_path.lstrip('/')}"
        headers = {
            "x-upsert": "true",
            "Content-Type": content_type,
        }
        resp = self.session.post(url, headers=headers, data=data, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"Upload failed for {object_path}: {resp.status_code} {resp.text}")

    def download_file(self, object_path: str) -> bytes:
        url = f"{self.base_url}/storage/v1/object/{self.bucket}/{object_path.lstrip('/')}"
        resp = self.session.get(url, timeout=120)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Object not found: {object_path}")
        if resp.status_code == 400:
            try:
                payload = resp.json()
            except ValueError:
                payload = None
            if payload and str(payload.get("statusCode")) == "404":
                raise FileNotFoundError(f"Object not found: {object_path}")
        if resp.status_code >= 400:
            raise RuntimeError(f"Download failed for {object_path}: {resp.status_code} {resp.text}")
        return resp.content


def resolve_base_url() -> str:
    url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    if not url:
        raise ValueError("SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL must be set.")
    return url.rstrip("/")


def resolve_service_key() -> str:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY must be set in environment.")
    return key


def load_run_id(artifacts_dir: Path, run_id: Optional[str]) -> str:
    if run_id:
        return run_id
    meta_path = artifacts_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found in {artifacts_dir}")
    meta = json.loads(meta_path.read_text())
    derived = meta.get("run_id")
    if not derived:
        raise ValueError("run_id not found in model_meta.json")
    return str(derived)


def build_upload_plan(
    artifacts_dir: Path,
    prefix: str,
    run_id: str,
    summary_path: Optional[Path],
) -> List[Tuple[Path, str, str, bool]]:
    plan: List[Tuple[Path, str, str, bool]] = []
    plan.append(
        (
            artifacts_dir / "two_tower_latest.onnx",
            f"{prefix}/{run_id}/two_tower_{run_id}.onnx",
            "application/octet-stream",
            False,
        )
    )
    onnx_data = artifacts_dir / "two_tower_latest.onnx.data"
    if onnx_data.exists():
        plan.append(
            (
                onnx_data,
                f"{prefix}/{run_id}/two_tower_{run_id}.onnx.data",
                "application/gzip",
                True,
            )
        )
    plan.append(
        (
            artifacts_dir / "two_tower_latest.pt",
            f"{prefix}/{run_id}/two_tower_{run_id}.pt",
            "application/gzip",
            True,
        )
    )
    plan.append(
        (
            artifacts_dir / "model_meta.json",
            f"{prefix}/{run_id}/model_meta.json",
            "application/json",
            False,
        )
    )
    plan.append(
        (
            artifacts_dir / "metrics.json",
            f"{prefix}/{run_id}/metrics.json",
            "application/json",
            False,
        )
    )
    if summary_path:
        plan.append((summary_path, f"{prefix}/{run_id}/summary.md", "text/markdown", False))
    return plan


def cmd_upload(args: argparse.Namespace) -> None:
    artifacts_dir = Path(args.artifacts_dir).resolve()
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")
    summary_path = Path(args.summary).resolve() if args.summary else artifacts_dir / "summary.md"
    if not summary_path.exists():
        summary_path = None
    run_id = load_run_id(artifacts_dir, args.run_id)

    base_url = resolve_base_url()
    service_key = resolve_service_key()
    client = StorageClient(base_url, service_key, args.bucket)

    plan = build_upload_plan(artifacts_dir, args.prefix.strip("/"), run_id, summary_path)
    for src, dst, content_type, compress in plan:
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")
        if compress:
            dst_object = dst + ".gz"
            if args.dry_run:
                print(json.dumps({"dry_run": True, "source": str(src), "destination": dst_object}))
                continue
            buffer = io.BytesIO()
            with src.open("rb") as fh, gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    gz.write(chunk)
            client.upload_content(buffer.getvalue(), dst_object, content_type)
            print(json.dumps({"uploaded": dst_object, "size": src.stat().st_size}))
            continue
        if args.dry_run:
            print(json.dumps({"dry_run": True, "source": str(src), "destination": dst}))
            continue
        client.upload_file(src, dst, content_type)
        print(json.dumps({"uploaded": dst, "size": src.stat().st_size}))


def dedupe_previous(previous: List[Dict], new_run: str, limit: int) -> List[Dict]:
    deduped: List[Dict] = []
    seen = {new_run}
    for entry in previous:
        run_id = str(entry.get("run_id", "")).strip()
        if not run_id or run_id in seen:
            continue
        deduped.append(entry)
        seen.add(run_id)
        if len(deduped) >= limit:
            break
    return deduped


def cmd_activate(args: argparse.Namespace) -> None:
    base_url = resolve_base_url()
    service_key = resolve_service_key()
    client = StorageClient(base_url, service_key, args.bucket)

    run_id = args.run_id
    if not run_id:
        artifacts_dir = Path(args.artifacts_dir).resolve()
        run_id = load_run_id(artifacts_dir, None)
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_prefix = args.prefix.strip("/")

    manifest_path = args.manifest_path.strip("/")
    manifest = client.download_json(manifest_path) or {}
    previous_entries = manifest.get("previous", [])

    current = manifest.get("current")
    if current and current.get("run_id") != run_id:
        previous_entry = {
            "run_id": current.get("run_id"),
            "onnx_path": current.get("onnx_path"),
            "pt_path": current.get("pt_path"),
            "meta_path": current.get("meta_path"),
            "metrics_path": current.get("metrics_path"),
            "summary_path": current.get("summary_path"),
            "onnx_data_path": current.get("onnx_data_path"),
        }
        if previous_entry["run_id"]:
            previous_entries = [previous_entry] + previous_entries

    previous_entries = dedupe_previous(previous_entries, run_id, args.max_previous)

    now_iso = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    summary_path = None
    if (artifacts_dir / "summary.md").exists():
        summary_path = f"{artifacts_prefix}/{run_id}/summary.md"
    onnx_data_path = None
    if (artifacts_dir / "two_tower_latest.onnx.data").exists():
        onnx_data_path = f"{artifacts_prefix}/{run_id}/two_tower_{run_id}.onnx.data.gz"

    current_entry = {
        "run_id": run_id,
        "published_at": now_iso,
        "onnx_path": f"{artifacts_prefix}/{run_id}/two_tower_{run_id}.onnx",
        "pt_path": f"{artifacts_prefix}/{run_id}/two_tower_{run_id}.pt.gz",
        "meta_path": f"{artifacts_prefix}/{run_id}/model_meta.json",
        "metrics_path": f"{artifacts_prefix}/{run_id}/metrics.json",
    }
    if summary_path:
        current_entry["summary_path"] = summary_path
    if onnx_data_path:
        current_entry["onnx_data_path"] = onnx_data_path

    manifest["model_name"] = args.model_name
    manifest["current"] = current_entry
    manifest["previous"] = previous_entries

    if args.release_notes:
        manifest["release_notes"] = args.release_notes
    elif args.release_notes_file:
        notes_path = Path(args.release_notes_file).resolve()
        if notes_path.exists():
            manifest["release_notes"] = notes_path.read_text().strip()

    if args.dry_run:
        print(json.dumps({"dry_run": True, "manifest": manifest}, ensure_ascii=False))
        return

    client.upload_json(manifest_path, manifest)
    print(json.dumps({"manifest_updated": manifest_path, "run_id": run_id}, ensure_ascii=False))


def select_manifest_entry(manifest: Dict, run_id: Optional[str]) -> Tuple[Dict, str]:
    candidates: List[Dict] = []
    current = manifest.get("current")
    if current:
        candidates.append(current)
    previous = manifest.get("previous") or []
    candidates.extend(previous)
    if not candidates:
        raise ValueError("Manifest does not contain any entries.")
    if run_id:
        for entry in candidates:
            if str(entry.get("run_id")) == str(run_id):
                return entry, str(entry.get("run_id"))
        raise ValueError(f"run_id {run_id} not found in manifest.")
    entry = candidates[0]
    return entry, str(entry.get("run_id"))


def cmd_fetch(args: argparse.Namespace) -> None:
    base_url = resolve_base_url()
    service_key = resolve_service_key()
    client = StorageClient(base_url, service_key, args.bucket)

    manifest = client.download_json(args.manifest_path.strip("/"))
    if not manifest:
        raise FileNotFoundError(f"Manifest not found at {args.manifest_path}")

    entry, run_id = select_manifest_entry(manifest, args.run_id)

    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloads: List[Tuple[str, str, bool]] = [
        ("onnx_path", "two_tower_latest.onnx", False),
        ("pt_path", "two_tower_latest.pt", True),
        ("meta_path", "model_meta.json", False),
        ("metrics_path", "metrics.json", False),
    ]
    if entry.get("onnx_data_path"):
        downloads.append(("onnx_data_path", "two_tower_latest.onnx.data", True))
    if entry.get("summary_path"):
        downloads.append(("summary_path", "summary.md", False))

    written: List[Dict[str, str]] = []
    for key, filename, decompress in downloads:
        storage_path = entry.get(key)
        if not storage_path:
            continue
        raw = client.download_file(storage_path)
        data = gzip.decompress(raw) if decompress else raw
        target = dest_dir / filename
        target.write_bytes(data)
        written.append({"source": storage_path, "dest": str(target)})

    print(
        json.dumps(
            {
                "event": "artifacts_downloaded",
                "run_id": run_id,
                "count": len(written),
                "dest": str(dest_dir),
                "files": written,
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish Two-Tower model artifacts to Supabase Storage.")
    sub = parser.add_subparsers(dest="command", required=True)

    upload = sub.add_parser("upload", help="Upload model artifacts.")
    upload.add_argument("--artifacts-dir", default="ml/artifacts/latest", help="Path to latest artifacts directory.")
    upload.add_argument("--run-id", default=None, help="Override run_id (defaults to model_meta.json).")
    upload.add_argument("--bucket", default="models", help="Supabase Storage bucket.")
    upload.add_argument("--prefix", default="two_tower", help="Prefix inside the bucket.")
    upload.add_argument("--summary", default=None, help="Path to release summary markdown.")
    upload.add_argument("--dry-run", action="store_true", help="Print planned uploads without sending.")
    upload.set_defaults(func=cmd_upload)

    activate = sub.add_parser("activate", help="Update manifest to point to a run_id.")
    activate.add_argument("--artifacts-dir", default="ml/artifacts/latest", help="Path to latest artifacts directory.")
    activate.add_argument("--run-id", default=None, help="Run ID to activate (defaults to model_meta.json).")
    activate.add_argument("--bucket", default="models", help="Supabase Storage bucket.")
    activate.add_argument("--prefix", default="two_tower", help="Prefix inside the bucket.")
    activate.add_argument("--manifest-path", default="two_tower/latest/manifest.json", help="Manifest object path.")
    activate.add_argument("--model_name", default="two_tower", help="Model name recorded in manifest.")
    activate.add_argument("--max-previous", type=int, default=5, help="Maximum previous versions to keep.")
    activate.add_argument("--release-notes", default=None, help="Inline release notes text.")
    activate.add_argument("--release-notes-file", default=None, help="Path to release notes text file.")
    activate.add_argument("--dry-run", action="store_true", help="Print manifest diff without uploading.")
    activate.set_defaults(func=cmd_activate)

    fetch = sub.add_parser("fetch", help="Download the artifacts referenced by the manifest.")
    fetch.add_argument("--manifest-path", default="two_tower/latest/manifest.json", help="Manifest object path.")
    fetch.add_argument("--bucket", default="models", help="Supabase Storage bucket.")
    fetch.add_argument("--run-id", default=None, help="Specific run_id to download (defaults to current).")
    fetch.add_argument("--dest", default="ml/artifacts/latest", help="Destination directory.")
    fetch.set_defaults(func=cmd_fetch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
