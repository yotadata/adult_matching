#!/usr/bin/env python3
"""Upload video embeddings to Supabase via REST API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

DEFAULT_BATCH_SIZE = 200


def chunk_rows(rows: Iterable[dict], size: int) -> Iterable[List[dict]]:
    batch: List[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_embeddings(parquet_path: Path) -> List[dict]:
    df = pd.read_parquet(parquet_path)
    records = []
    for row in df.itertuples(index=False):
        vector = row.embedding
        if isinstance(vector, (list, tuple)):
            embedding = [float(x) for x in vector]
        else:
            embedding = [float(x) for x in list(vector)]
        records.append({
            "video_id": str(row.video_id),
            "embedding": embedding,
        })
    return records


def upload_embeddings(
    url: str,
    service_role_key: str,
    records: List[dict],
    batch_size: int,
) -> None:
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    for batch in chunk_rows(records, batch_size):
        resp = requests.post(url, json=batch, headers=headers, timeout=60)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"Failed to upsert batch (status={resp.status_code}): {resp.text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload video embeddings to Supabase")
    parser.add_argument("--parquet", type=Path, default=Path("artifacts/video_embeddings.parquet"))
    parser.add_argument("--supabase-url", type=str, default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--service-role-key", type=str, default=os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.supabase_url or not args.service_role_key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be provided")

    records = load_embeddings(args.parquet)
    if args.dry_run:
        print(f"Loaded {len(records)} embeddings (dry run)")
        return

    endpoint = args.supabase_url.rstrip("/") + "/rest/v1/video_embeddings"
    upload_embeddings(endpoint, args.service_role_key, records, args.batch_size)
    print(f"Uploaded {len(records)} embeddings to Supabase")


if __name__ == "__main__":
    main()
