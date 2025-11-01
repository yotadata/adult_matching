#!/usr/bin/env python3
import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from tqdm import tqdm


@dataclass
class UpsertResult:
    table: str
    inserted: int
    skipped: int
    dropped: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upsert Two-Tower embeddings into Supabase Postgres.")
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("ml/artifacts/latest"),
        help="Directory containing latest artifacts (user_embeddings.parquet, video_embeddings.parquet, model_meta.json).",
    )
    ap.add_argument(
        "--db-url",
        default=None,
        help="Postgres connection string. Defaults to SUPABASE_DB_URL or REMOTE_DATABASE_URL from env.",
    )
    ap.add_argument(
        "--include-users",
        action="store_true",
        help="Upsert user embeddings as well (requires reviewer_id to be Supabase user UUID).",
    )
    ap.add_argument(
        "--min-user-interactions",
        type=int,
        default=0,
        help="Minimum interactions required to upsert a user embedding (only applies when --include-users is set).",
    )
    ap.add_argument(
        "--interactions",
        type=Path,
        action="append",
        help="Parquet files containing interactions (reviewer_id) used to compute interaction counts. "
        "If omitted, defaults to latest train/val parquets.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of records to upsert per batch.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Load artifacts and report counts without writing to the database.",
    )
    return ap.parse_args()


def load_embeddings(parquet_path: Path, id_column: str) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if id_column not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"{parquet_path} must contain '{id_column}' and 'embedding' columns.")
    return df


def normalize_uuid(value: object, column: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except Exception as exc:
        raise ValueError(f"Invalid UUID in column '{column}': {value}") from exc


def to_vector_list(embedding: Iterable[float]) -> List[float]:
    return [float(x) for x in embedding]


def prepare_rows(df: pd.DataFrame, id_column: str) -> Tuple[List[Tuple[str, List[float]]], int, int]:
    rows: List[Tuple[uuid.UUID, List[float]]] = []
    skipped = 0
    non_uuid = 0
    for record in df.itertuples(index=False):
        try:
            entity_id = normalize_uuid(getattr(record, id_column), id_column)
            embedding = getattr(record, "embedding")
            if embedding is None:
                skipped += 1
                continue
            rows.append((entity_id, to_vector_list(embedding)))
        except Exception as exc:
            if isinstance(exc, ValueError) and "Invalid UUID" in str(exc):
                non_uuid += 1
                continue
            skipped += 1
            print(f"[WARN] Skipping row due to error: {exc}", file=sys.stderr)
    return rows, skipped, non_uuid


def prepare_user_rows(
    df: pd.DataFrame,
    id_column: str,
    eligible_ids: Iterable[str] | None,
) -> Tuple[List[Tuple[str, List[float]]], int, int, int]:
    allowed: set[str] | None = set(str(x) for x in eligible_ids) if eligible_ids is not None else None
    rows: List[Tuple[uuid.UUID, List[float]]] = []
    skipped = 0
    filtered = 0
    non_uuid = 0
    for record in df.itertuples(index=False):
        reviewer_id_raw = getattr(record, id_column)
        reviewer_id_str = str(reviewer_id_raw)
        if allowed is not None and reviewer_id_str not in allowed:
            filtered += 1
            continue
        try:
            entity_id = normalize_uuid(reviewer_id_raw, id_column)
            embedding = getattr(record, "embedding")
            if embedding is None:
                skipped += 1
                continue
            rows.append((entity_id, to_vector_list(embedding)))
        except Exception as exc:
            if isinstance(exc, ValueError) and "Invalid UUID" in str(exc):
                non_uuid += 1
                continue
            skipped += 1
            print(f"[WARN] Skipping user row due to error: {exc}", file=sys.stderr)
    return rows, skipped, filtered, non_uuid


def chunk(sequence: Sequence[Tuple[uuid.UUID, List[float]]], size: int) -> Iterable[List[Tuple[uuid.UUID, List[float]]]]:
    for idx in range(0, len(sequence), size):
        yield list(sequence[idx : idx + size])


def upsert_embeddings(
    conn: psycopg.Connection,
    table: str,
    id_column: str,
    rows: List[Tuple[uuid.UUID, List[float]]],
    chunk_size: int,
) -> UpsertResult:
    if not rows:
        return UpsertResult(table=table, inserted=0, skipped=0, dropped=0)

    inserted = 0
    with conn.cursor() as cur:
        for batch in tqdm(chunk(rows, chunk_size), desc=f"Upserting {table}", unit="batch"):
            params = [(row_id, embedding) for row_id, embedding in batch]
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} ({id_col}, embedding, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT ({id_col}) DO UPDATE
                      SET embedding = EXCLUDED.embedding,
                          updated_at = EXCLUDED.updated_at
                    """
                ).format(table=sql.Identifier(table), id_col=sql.Identifier(id_column)),
                params,
            )
            inserted += len(batch)
    return UpsertResult(table=table, inserted=inserted, skipped=0, dropped=0)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()

    db_url = args.db_url or os.environ.get("SUPABASE_DB_URL") or os.environ.get("REMOTE_DATABASE_URL")
    if not db_url:
        raise ValueError("Database URL not provided. Set --db-url or SUPABASE_DB_URL / REMOTE_DATABASE_URL env.")

    model_meta_path = artifacts_dir / "model_meta.json"
    if model_meta_path.exists():
        meta = json.loads(model_meta_path.read_text())
        print(json.dumps({"info": "model_meta", "path": str(model_meta_path), "meta": meta}, ensure_ascii=False))
    else:
        print(f"[WARN] model_meta.json not found in {artifacts_dir}", file=sys.stderr)

    video_df = load_embeddings(artifacts_dir / "video_embeddings.parquet", "video_id")
    video_rows, video_skipped, video_non_uuid = prepare_rows(video_df, "video_id")
    print(json.dumps({
        "info": "video_embeddings_loaded",
        "rows": len(video_rows),
        "skipped": video_skipped,
        "non_uuid": video_non_uuid,
    }))

    user_rows: List[Tuple[uuid.UUID, List[float]]] = []
    user_skipped = 0
    user_filtered = 0
    if args.include_users:
        user_df = load_embeddings(artifacts_dir / "user_embeddings.parquet", "reviewer_id")
        eligible_ids: Iterable[str] | None = None
        if args.min_user_interactions > 0:
            interaction_paths = args.interactions
            if not interaction_paths:
                interaction_paths = [
                    Path("ml/data/processed/two_tower/latest/interactions_train.parquet"),
                    Path("ml/data/processed/two_tower/latest/interactions_val.parquet"),
                ]
            frames: List[pd.DataFrame] = []
            for ipath in interaction_paths:
                if ipath.exists():
                    frames.append(pd.read_parquet(ipath, columns=["reviewer_id"]))
            if not frames:
                raise FileNotFoundError("No interactions parquet found for user filtering. Provide --interactions explicitly.")
            interaction_df = pd.concat(frames, ignore_index=True)
            counts = interaction_df["reviewer_id"].astype(str).value_counts()
            eligible_ids = counts[counts >= args.min_user_interactions].index.tolist()
            print(json.dumps({
                "info": "user_interaction_filter",
                "min_interactions": args.min_user_interactions,
                "eligible_users": len(eligible_ids),
            }))
        user_rows, user_skipped, user_filtered, user_non_uuid = prepare_user_rows(user_df, "reviewer_id", eligible_ids)
        print(json.dumps({
            "info": "user_embeddings_loaded",
            "rows": len(user_rows),
            "skipped": user_skipped,
            "filtered": user_filtered,
            "non_uuid": user_non_uuid,
        }))

    if args.dry_run:
        print(json.dumps({"dry_run": True, "video_rows": len(video_rows), "user_rows": len(user_rows)}))
        return

    conn = psycopg.connect(db_url, autocommit=False, row_factory=dict_row)
    try:
        register_vector(conn)
        with conn:
            video_result = upsert_embeddings(conn, "video_embeddings", "video_id", video_rows, args.chunk_size)
            user_result = None
            if args.include_users and user_rows:
                user_result = upsert_embeddings(conn, "user_embeddings", "user_id", user_rows, args.chunk_size)
        summaries = [
            {"table": video_result.table, "inserted": video_result.inserted, "skipped": video_skipped},
        ]
        if args.include_users:
            summaries.append({
                "table": "user_embeddings",
                "inserted": user_result.inserted if user_result else 0,
                "skipped": user_skipped,
                "filtered": user_filtered,
                "min_interactions": args.min_user_interactions,
            })
        print(json.dumps({"event": "upsert_completed", "summaries": summaries}, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
