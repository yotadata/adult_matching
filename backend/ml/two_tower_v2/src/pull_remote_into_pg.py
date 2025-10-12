from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

from psycopg.types.json import Json

from db import PostgresConfig, delete_rows_by_ids, upsert_rows

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
        tags = [
            entry.get("tags", {}).get("name")
            for entry in video_tags
            if isinstance(entry, dict) and entry.get("tags") and entry["tags"].get("name")
        ]

        video_performers = new_row.pop("video_performers", None) or []
        performers = [
            entry.get("performers", {}).get("name")
            for entry in video_performers
            if isinstance(entry, dict) and entry.get("performers") and entry["performers"].get("name")
        ]

        new_row["tags"] = [tag for tag in tags if tag]
        new_row["performers"] = [name for name in performers if name]
        transformed.append(new_row)
    return transformed


def chunked(iterable: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def upsert_profiles(cfg: PostgresConfig, profiles: List[Dict[str, Any]]) -> None:
    upsert_rows(
        cfg,
        table="profiles",
        columns=["user_id", "display_name", "created_at"],
        rows=profiles,
        conflict_columns=["user_id"],
    )


def upsert_videos(cfg: PostgresConfig, videos: List[Dict[str, Any]]) -> None:
    if not videos:
        return

    base_rows = []
    tag_rows: List[Dict[str, Any]] = []
    performer_rows: List[Dict[str, Any]] = []
    video_ids: List[str] = []

    columns = [
        "id",
        "external_id",
        "title",
        "description",
        "duration_seconds",
        "thumbnail_url",
        "preview_video_url",
        "distribution_code",
        "maker_code",
        "director",
        "series",
        "maker",
        "label",
        "price",
        "distribution_started_at",
        "product_released_at",
        "sample_video_url",
        "image_urls",
        "source",
        "published_at",
        "product_url",
        "created_at",
    ]

    for video in videos:
        video_id = video.get("id")
        if not video_id:
            continue
        video_ids.append(video_id)
        tags = video.pop("tags", []) or []
        performers = video.pop("performers", []) or []

        row = {col: video.get(col) for col in columns}
        if isinstance(row.get("image_urls"), list):
            row["image_urls"] = Json(row["image_urls"])
        base_rows.append(row)
        tag_rows.extend({"video_id": video_id, "tag_name": tag} for tag in tags)
        performer_rows.extend({"video_id": video_id, "performer_name": name} for name in performers)

    upsert_rows(
        cfg,
        table="videos",
        columns=columns,
        rows=base_rows,
        conflict_columns=["id"],
    )

    delete_rows_by_ids(cfg, "video_tags", "video_id", video_ids)
    delete_rows_by_ids(cfg, "video_performers", "video_id", video_ids)

    if tag_rows:
        upsert_rows(
            cfg,
            table="video_tags",
            columns=["video_id", "tag_name"],
            rows=tag_rows,
            conflict_columns=["video_id", "tag_name"],
        )
    if performer_rows:
        upsert_rows(
            cfg,
            table="video_performers",
            columns=["video_id", "performer_name"],
            rows=performer_rows,
            conflict_columns=["video_id", "performer_name"],
        )


def upsert_decisions(cfg: PostgresConfig, decisions: List[Dict[str, Any]], batch_size: int = 5000) -> None:
    columns = ["user_id", "video_id", "decision_type", "created_at"]
    conflicts = ["user_id", "video_id", "decision_type", "created_at"]
    for batch in chunked(decisions, batch_size):
        upsert_rows(cfg, "user_video_decisions", columns, batch, conflicts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull Supabase data into local Postgres")
    parser.add_argument("--supabase-url", type=str, default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--service-role-key", type=str, default=os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    parser.add_argument("--pg-dsn", type=str, required=True)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.supabase_url or not args.service_role_key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be provided")

    cfg = PostgresConfig(dsn=args.pg_dsn)
    fetcher = SupabaseFetcher(
        base_url=args.supabase_url,
        service_role_key=args.service_role_key,
        page_size=args.page_size,
    )

    print("📥 Fetching profiles…")
    profiles = fetcher.fetch_all(
        "profiles",
        select="user_id, display_name, created_at",
        order="user_id.asc",
    )

    print("📥 Fetching videos…")
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

    print("📥 Fetching user decisions…")
    decisions = fetcher.fetch_all(
        "user_video_decisions",
        select="user_id, video_id, decision_type, created_at",
        order="created_at.asc",
    )

    print("⬆️ Upserting profiles…")
    upsert_profiles(cfg, profiles)

    print("⬆️ Upserting videos…")
    upsert_videos(cfg, videos)

    print("⬆️ Upserting decisions…")
    upsert_decisions(cfg, decisions)

    print("✅ Remote data import complete")


if __name__ == "__main__":
    main()
