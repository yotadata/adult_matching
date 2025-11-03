#!/usr/bin/env python3
import argparse
import json
import os
import sys
import uuid
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit, quote

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


def _eligible_users_from_parquet(
    interaction_paths: Iterable[Path] | None,
    min_interactions: int,
) -> Optional[List[str]]:
    paths = list(interaction_paths or [])
    if not paths:
        paths = [
            Path("ml/data/processed/two_tower/latest/interactions_train.parquet"),
            Path("ml/data/processed/two_tower/latest/interactions_val.parquet"),
        ]
    frames: List[pd.DataFrame] = []
    for ipath in paths:
        if ipath.exists():
            frames.append(pd.read_parquet(ipath, columns=["reviewer_id"]))
    if not frames:
        return None
    interaction_df = pd.concat(frames, ignore_index=True)
    counts = interaction_df["reviewer_id"].astype(str).value_counts()
    return counts[counts >= min_interactions].index.tolist()


def _eligible_users_from_db(db_url: str, min_interactions: int) -> Optional[List[str]]:
    sql_query = """
        SELECT au.id AS user_id
        FROM public.user_video_decisions uvd
        JOIN auth.users au ON au.id = uvd.user_id
        WHERE uvd.decision_type = 'like'
        GROUP BY au.id
        HAVING COUNT(*) >= %s
    """
    try:
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query, (min_interactions,))
                rows = cur.fetchall()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "warn": "user_interaction_filter_db_failed",
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return None
    if not rows:
        return []
    eligible: List[str] = []
    for row in rows:
        value = row["user_id"] if isinstance(row, dict) else row[0]
        if value is None:
            continue
        eligible.append(str(value))
    return eligible


def _filter_existing_auth_users(
    conn: psycopg.Connection,
    rows: List[Tuple[uuid.UUID, List[float]]],
) -> Tuple[List[Tuple[uuid.UUID, List[float]]], int]:
    if not rows:
        return rows, 0
    user_ids = [row_id for row_id, _ in rows]
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM auth.users WHERE id = ANY(%s)", (user_ids,))
        fetched = cur.fetchall()
    existing_ids: set[uuid.UUID] = set()
    for row in fetched:
        value = row["id"] if isinstance(row, dict) else row[0]
        if value is None:
            continue
        existing_ids.add(uuid.UUID(str(value)))
    filtered_rows = [row for row in rows if row[0] in existing_ids]
    dropped = len(rows) - len(filtered_rows)
    return filtered_rows, dropped


def chunk(sequence: Sequence[Tuple[uuid.UUID, List[float]]], size: int) -> Iterable[List[Tuple[uuid.UUID, List[float]]]]:
    for idx in range(0, len(sequence), size):
        yield list(sequence[idx : idx + size])


def upsert_embeddings(
    conn: psycopg.Connection,
    table: str,
    id_column: str,
    rows: List[Tuple[uuid.UUID, List[float]]],
    chunk_size: int,
    *,
    version_column: str | None = None,
    version_value: str | None = None,
) -> UpsertResult:
    if not rows:
        return UpsertResult(table=table, inserted=0, skipped=0, dropped=0)

    if version_column and version_value is None:
        raise ValueError(f"version_value must be provided when version_column={version_column}")

    inserted = 0
    with conn.cursor() as cur:
        for batch in tqdm(chunk(rows, chunk_size), desc=f"Upserting {table}", unit="batch"):
            if version_column:
                params = [(row_id, embedding, version_value) for row_id, embedding in batch]
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {table} ({id_col}, embedding, {version_col}, updated_at)
                        VALUES (%s, %s, %s, now())
                        ON CONFLICT ({id_col}) DO UPDATE
                          SET embedding = EXCLUDED.embedding,
                              {version_col} = EXCLUDED.{version_col},
                              updated_at = EXCLUDED.updated_at
                        """
                    ).format(
                        table=sql.Identifier(table),
                        id_col=sql.Identifier(id_column),
                        version_col=sql.Identifier(version_column),
                    ),
                    params,
                )
            else:
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


def _ensure_embedding_schema(conn: psycopg.Connection, target_dim: int | None) -> None:
    if target_dim is None:
        target_dim = 256
    dim_sql = sql.SQL(str(target_dim))
    changed = False
    with conn.cursor() as cur:
        current_video_dim = _get_vector_dim(cur, "public.video_embeddings", "embedding")
        current_user_dim = _get_vector_dim(cur, "public.user_embeddings", "embedding")
        if current_video_dim != target_dim:
            print(json.dumps({"info": "adjust_video_dim", "from": current_video_dim, "to": target_dim}, ensure_ascii=False))
            cur.execute("DROP INDEX IF EXISTS idx_video_embeddings_cosine")
            cur.execute("TRUNCATE TABLE public.video_embeddings RESTART IDENTITY")
            cur.execute(sql.SQL("ALTER TABLE public.video_embeddings ALTER COLUMN embedding TYPE vector({dim})").format(dim=dim_sql))
            cur.execute("CREATE INDEX IF NOT EXISTS idx_video_embeddings_cosine ON public.video_embeddings USING ivfflat (embedding vector_cosine_ops)")
            changed = True
        if current_user_dim and current_user_dim != target_dim:
            print(json.dumps({"info": "adjust_user_dim", "from": current_user_dim, "to": target_dim}, ensure_ascii=False))
            cur.execute("DROP INDEX IF EXISTS idx_user_embeddings_cosine")
            cur.execute("TRUNCATE TABLE public.user_embeddings RESTART IDENTITY")
            cur.execute(sql.SQL("ALTER TABLE public.user_embeddings ALTER COLUMN embedding TYPE vector({dim})").format(dim=dim_sql))
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_embeddings_cosine ON public.user_embeddings USING ivfflat (embedding vector_cosine_ops)")
            changed = True
    if changed:
        conn.commit()


def _get_vector_dim(cur: psycopg.Cursor, table: str, column: str) -> int | None:
    cur.execute(
        """
        SELECT NULLIF(atttypmod, -1) AS dims
        FROM pg_attribute
        WHERE attrelid = %s::regclass
          AND attname = %s
          AND NOT attisdropped
        """,
        (table, column),
    )
    row = cur.fetchone()
    if not row:
        return None
    if isinstance(row, dict):
        return row.get("dims")
    return row[0]


def _ensure_ipv4_hostaddr(conninfo: str, allow_pooler: bool = True) -> str:
    """Append hostaddr=<ipv4> to conninfo; fall back to pooler URL when needed."""
    try:
        parsed = urlsplit(conninfo)
    except Exception:
        return conninfo

    if not parsed.hostname:
        return conninfo

    query = parse_qs(parsed.query, keep_blank_values=True)
    if "hostaddr" in query:
        return conninfo

    addrinfo = []
    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, parsed.port or 5432, family=socket.AF_INET)
    except socket.gaierror:
        addrinfo = []

    ipv4_addr = next((info[4][0] for info in addrinfo if info[0] == socket.AF_INET), None)

    if not ipv4_addr and allow_pooler:
        project_ref = os.environ.get("SUPABASE_PROJECT_REF")
        if not project_ref:
            parts = parsed.hostname.split('.')
            if len(parts) >= 3 and parts[0] == "db":
                project_ref = parts[1]

        # Prefer explicit pooler host via env
        pooler_host = os.environ.get("SUPABASE_POOLER_HOST")
        if not pooler_host:
            region = os.environ.get("SUPABASE_REGION")
            if region:
                pooler_host = f"{region}.pooler.supabase.com"

        if pooler_host:
            pooler_port = os.environ.get("SUPABASE_POOLER_PORT") or "6543"
            pooler_user = os.environ.get("SUPABASE_POOLER_USER")
            if not pooler_user and project_ref and parsed.username:
                pooler_user = f"{parsed.username}.{project_ref}"
            username = pooler_user or parsed.username or ""
            password = parsed.password or ""
            auth = quote(username)
            if password:
                auth = f"{auth}:{quote(password)}"
            netloc = f"{auth}@{pooler_host}:{pooler_port}"
            pooler_conn = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
            print(json.dumps({
                "info": "pooler_url_computed",
                "url": pooler_conn
            }, ensure_ascii=False))
            return _ensure_ipv4_hostaddr(pooler_conn, allow_pooler=False)

        token = os.environ.get("SUPABASE_ACCESS_TOKEN")
        pooler_url = _fetch_pooler_connection(project_ref, token)
        if pooler_url:
            return _ensure_ipv4_hostaddr(pooler_url, allow_pooler=False)

    if not ipv4_addr:
        return conninfo

    query.setdefault("hostaddr", []).append(ipv4_addr)
    new_query = urlencode(query, doseq=True)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))


def _fetch_pooler_connection(project_ref: str | None, token: str | None) -> str | None:
    if not project_ref or not token:
        return None
    try:
        params = urlencode({"pooler": "true", "type": "psql"})
        url = f"https://api.supabase.com/v1/projects/{project_ref}/db/connection-string?{params}"
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": token,
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        conn = data.get("connectionString") or data.get("connection_string")
        if conn:
            print(json.dumps({"info": "pooler_url_used", "url": conn}, ensure_ascii=False))
        return conn
    except urllib.error.HTTPError as exc:
        print(json.dumps({"warn": "pooler_fetch_failed", "status": exc.code}, ensure_ascii=False), file=sys.stderr)
    except Exception as exc:
        print(json.dumps({"warn": "pooler_fetch_failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
    return None


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()

    db_url = args.db_url or os.environ.get("SUPABASE_DB_URL") or os.environ.get("REMOTE_DATABASE_URL")
    if not db_url:
        raise ValueError("Database URL not provided. Set --db-url or SUPABASE_DB_URL / REMOTE_DATABASE_URL env.")

    db_url = _ensure_ipv4_hostaddr(db_url)
    print(json.dumps({"info": "db_url", "url": db_url}, ensure_ascii=False))

    model_meta_path = artifacts_dir / "model_meta.json"
    meta: dict | None = None
    if model_meta_path.exists():
        meta = json.loads(model_meta_path.read_text())
        print(json.dumps({"info": "model_meta", "path": str(model_meta_path), "meta": meta}, ensure_ascii=False))
    else:
        print(f"[WARN] model_meta.json not found in {artifacts_dir}", file=sys.stderr)

    model_version = None
    if meta:
        raw_version = meta.get("run_id")
        if raw_version:
            model_version = str(raw_version).strip()
    if not model_version:
        model_version = "unknown"
    print(json.dumps({"info": "model_version", "value": model_version}, ensure_ascii=False))

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
            eligible_ids = _eligible_users_from_db(db_url, args.min_user_interactions)
            source = "database"
            if eligible_ids is None:
                eligible_ids = _eligible_users_from_parquet(args.interactions, args.min_user_interactions)
                source = "parquet"
            if eligible_ids is None:
                raise FileNotFoundError(
                    "Failed to determine eligible users. Database query failed and no interactions parquet available."
                )
            print(
                json.dumps(
                    {
                        "info": "user_interaction_filter",
                        "source": source,
                        "min_interactions": args.min_user_interactions,
                        "eligible_users": len(eligible_ids),
                    },
                    ensure_ascii=False,
                )
            )
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
        _ensure_embedding_schema(conn, target_dim=len(video_rows[0][1]) if video_rows else None)
        with conn:
            dropped_fk = 0
            if args.include_users and user_rows:
                user_rows, dropped_fk = _filter_existing_auth_users(conn, user_rows)
                if dropped_fk:
                    print(
                        json.dumps(
                            {
                                "warn": "user_embeddings_fk_missing",
                                "dropped": dropped_fk,
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                    )
            video_result = upsert_embeddings(
                conn,
                "video_embeddings",
                "video_id",
                video_rows,
                args.chunk_size,
                version_column="model_version",
                version_value=model_version,
            )
            user_result = None
            if args.include_users and user_rows:
                user_result = upsert_embeddings(conn, "user_embeddings", "user_id", user_rows, args.chunk_size)
        summaries = [
            {
                "table": video_result.table,
                "inserted": video_result.inserted,
                "skipped": video_skipped,
                "model_version": model_version,
            },
        ]
        if args.include_users:
            summaries.append({
                "table": "user_embeddings",
                "inserted": user_result.inserted if user_result else 0,
                "skipped": user_skipped,
                "filtered": user_filtered,
                "dropped_fk": dropped_fk,
                "min_interactions": args.min_user_interactions,
            })
        print(json.dumps({"event": "upsert_completed", "summaries": summaries}, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
def _get_vector_dim(cur: psycopg.Cursor, table: str, column: str) -> int | None:
    cur.execute(
        """
        SELECT NULLIF(atttypmod, -1)
        FROM pg_attribute
        WHERE attrelid = %s::regclass
          AND attname = %s
          AND NOT attisdropped
        """,
        (table, column),
    )
    row = cur.fetchone()
    return row[0] if row else None
