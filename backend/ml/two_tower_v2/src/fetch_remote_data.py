from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_PAGE_SIZE = 1000


@dataclass
class SupabaseFetcher:
    base_url: str
    service_role_key: str
    page_size: int = DEFAULT_PAGE_SIZE

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()
        self.base_headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Prefer": "count=exact",
        }

    def fetch_all(self, resource: str, *, select: str, order: Optional[str] = None) -> List[Dict[str, Any]]:
        offset = 0
        results: List[Dict[str, Any]] = []
        total: Optional[int] = None

        while True:
            headers = {
                **self.base_headers,
                "Range": f"{offset}-{offset + self.page_size - 1}",
            }
            params = {"select": select}
            if order:
                params["order"] = order

            response = self.session.get(
                f"{self.base_url}/rest/v1/{resource}",
                headers=headers,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break

            results.extend(batch)
            content_range = response.headers.get("content-range")
            if content_range and "/" in content_range:
                try:
                    total = int(content_range.split("/")[-1])
                except ValueError:
                    total = None

            offset += len(batch)
            if total is not None and offset >= total:
                break

            if len(batch) < self.page_size:
                break

        return results


def transform_videos(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transformed: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)

        video_tags = new_row.pop("video_tags", None) or []
        new_row["tags"] = [
            entry.get("tags", {}).get("name")
            for entry in video_tags
            if isinstance(entry, dict) and entry.get("tags") and entry["tags"].get("name")
        ]

        video_performers = new_row.pop("video_performers", None) or []
        new_row["performers"] = [
            entry.get("performers", {}).get("name")
            for entry in video_performers
            if isinstance(entry, dict) and entry.get("performers") and entry["performers"].get("name")
        ]

        transformed.append(new_row)
    return transformed


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Supabase data for Two-Tower training")
    parser.add_argument("--supabase-url", type=str, default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--service-role-key", type=str, default=os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/remote_data"))
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.supabase_url or not args.service_role_key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be provided")

    fetcher = SupabaseFetcher(
        base_url=args.supabase_url,
        service_role_key=args.service_role_key,
        page_size=args.page_size,
    )

    print("📥 Fetching profiles...")
    profiles = fetcher.fetch_all(
        "profiles",
        select="user_id, display_name, created_at",
        order="user_id.asc",
    )

    print("📥 Fetching videos...")
    videos_raw = fetcher.fetch_all(
        "videos",
        select=(
            "id, external_id, title, description, duration_seconds, thumbnail_url, preview_video_url, "
            "distribution_code, maker_code, director, series, maker, label, price, distribution_started_at, "
            "product_released_at, sample_video_url, image_urls, source, published_at, product_url, created_at, "
            "video_tags ( tags ( name ) ), video_performers ( performers ( name ) )"
        ),
        order="id.asc",
    )
    videos = transform_videos(videos_raw)

    print("📥 Fetching user decisions...")
    decisions = fetcher.fetch_all(
        "user_video_decisions",
        select="user_id, video_id, decision_type, created_at",
        order="created_at.asc",
    )

    output_dir = args.output_dir
    write_json(output_dir / "profiles.json", profiles)
    write_json(output_dir / "videos_subset.json", videos)
    write_json(output_dir / "user_video_decisions.json", decisions)

    print(f"✅ Remote data written to {output_dir}")


if __name__ == "__main__":
    main()
