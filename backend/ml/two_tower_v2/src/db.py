from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row


@dataclass
class PostgresConfig:
    dsn: str


def fetch_profiles(cfg: PostgresConfig) -> pd.DataFrame:
    query = "SELECT user_id, display_name, created_at FROM public.profiles"
    with psycopg.connect(cfg.dsn, row_factory=dict_row) as conn:
        rows = conn.execute(query).fetchall()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["user_id"] = df["user_id"].astype(str)
    return df


def fetch_videos(cfg: PostgresConfig) -> pd.DataFrame:
    video_query = """
        SELECT id, external_id, title, description, duration_seconds, thumbnail_url,
               preview_video_url, distribution_code, maker_code, director, series,
               maker, label, price, distribution_started_at, product_released_at,
               sample_video_url, image_urls, source, published_at, product_url, created_at
        FROM public.videos
    """
    tag_query = "SELECT video_id, tag_name FROM public.video_tags"
    performer_query = "SELECT video_id, performer_name FROM public.video_performers"

    with psycopg.connect(cfg.dsn, row_factory=dict_row) as conn:
        videos = conn.execute(video_query).fetchall()
        tags = conn.execute(tag_query).fetchall()
        performers = conn.execute(performer_query).fetchall()

    video_df = pd.DataFrame(videos)
    if video_df.empty:
        video_df = pd.DataFrame(columns=[
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
        ])

    if not video_df.empty:
        datetime_cols = [
            "distribution_started_at",
            "product_released_at",
            "published_at",
            "created_at",
        ]
        for col in datetime_cols:
            if col in video_df.columns:
                video_df[col] = pd.to_datetime(video_df[col], errors="coerce")
        video_df["id"] = video_df["id"].astype(str)

    tags_df = pd.DataFrame(tags)
    performers_df = pd.DataFrame(performers)

    def build_lookup(frame: pd.DataFrame, column: str) -> dict[str, List[str]]:
        if frame.empty:
            return {}
        frame["video_id"] = frame["video_id"].astype(str)
        grouped = frame.groupby("video_id")[column].apply(lambda s: [v for v in s.tolist() if v]).to_dict()
        return grouped

    tag_lookup = build_lookup(tags_df, "tag_name")
    performer_lookup = build_lookup(performers_df, "performer_name")

    if not video_df.empty:
        video_df["tags"] = video_df["id"].map(tag_lookup).apply(lambda x: x if isinstance(x, list) else [])
        video_df["performers"] = video_df["id"].map(performer_lookup).apply(lambda x: x if isinstance(x, list) else [])
    else:
        video_df["tags"] = []
        video_df["performers"] = []

    return video_df


def fetch_decisions(cfg: PostgresConfig) -> pd.DataFrame:
    query = "SELECT user_id, video_id, decision_type, created_at FROM public.user_video_decisions"
    with psycopg.connect(cfg.dsn, row_factory=dict_row) as conn:
        rows = conn.execute(query).fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["user_id"] = df["user_id"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def upsert_rows(
    cfg: PostgresConfig,
    table: str,
    columns: Iterable[str],
    rows: List[dict],
    conflict_columns: Iterable[str],
) -> None:
    if not rows:
        return
    column_list = list(columns)
    conflict_list = list(conflict_columns)
    insert_stmt = sql.SQL("""
        INSERT INTO {table} ({cols})
        VALUES ({values})
        ON CONFLICT ({conflict}) DO UPDATE SET {updates}
    """)

    updates = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(column)) for column in column_list
    )

    stmt = insert_stmt.format(
        table=sql.Identifier("public", table),
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in column_list),
        values=sql.SQL(", ").join(sql.Placeholder() for _ in column_list),
        conflict=sql.SQL(", ").join(sql.Identifier(c) for c in conflict_list),
        updates=updates,
    )

    value_rows = [[row.get(col) for col in column_list] for row in rows]

    with psycopg.connect(cfg.dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(stmt, value_rows)
        conn.commit()


def delete_rows_by_ids(cfg: PostgresConfig, table: str, column: str, ids: List[str]) -> None:
    if not ids:
        return
    with psycopg.connect(cfg.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {table} WHERE {column} = ANY(%s)").format(
                    table=sql.Identifier("public", table),
                    column=sql.Identifier(column),
                ),
                (ids,),
            )
        conn.commit()
