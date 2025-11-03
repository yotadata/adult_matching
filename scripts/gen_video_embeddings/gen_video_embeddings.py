#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit, quote
import urllib.error
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn
import torch.nn.functional as F
from psycopg.rows import dict_row
from tqdm import tqdm


JST = timezone(timedelta(hours=9))


@lru_cache(maxsize=1)
def _resolve_project_ref() -> Optional[str]:
    project_id = os.environ.get("SUPABASE_PROJECT_ID")
    if project_id:
        return project_id
    legacy = os.environ.get("SUPABASE_PROJECT_REF")
    if legacy:
        print(
            json.dumps(
                {
                    "warn": "deprecated_env_var",
                    "details": "Set SUPABASE_PROJECT_ID; SUPABASE_PROJECT_REF is deprecated.",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return legacy
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video embeddings for newly ingested items.")
    parser.add_argument("--db-url", type=str, default=None, help="Postgres connection string (defaults to SUPABASE_DB_URL / REMOTE_DATABASE_URL).")
    parser.add_argument("--model-meta", type=Path, default=Path("ml/artifacts/latest/model_meta.json"), help="Model metadata JSON.")
    parser.add_argument("--model-path", type=Path, default=Path("ml/artifacts/latest/two_tower_latest.pt"), help="Two-Tower PyTorch weights.")
    parser.add_argument(
        "--reference-item-features",
        type=Path,
        default=Path("ml/data/processed/two_tower/latest/item_features.parquet"),
        help="Training-time item_features parquet used to rebuild vocabularies.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/live/video_embeddings"), help="Directory to write embeddings.")
    parser.add_argument("--output-name", type=str, default="video_embeddings.parquet", help="Output parquet filename.")
    parser.add_argument("--since", type=str, default=None, help="Lower bound for product_released_at (ISO date). Interpreted as JST if timezone omitted.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of videos to encode (for debugging).")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Encode videos even if current model_version already exists. By default only missing/outdated entries are processed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data and report counts without writing parquet.")
    return parser.parse_args()


def _ensure_list(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(value)]


def _normalize_array(value, *, sort: bool = False) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text:
            if text.startswith("[") or text.startswith("{"):
                try:
                    parsed = json.loads(text.replace("'", '"'))
                except json.JSONDecodeError:
                    parsed = [v.strip() for v in text.replace("{", "").replace("}", "").split(",") if v.strip()]
            else:
                parsed = [v.strip() for v in text.split(",") if v.strip()]
        vals = _ensure_list(parsed if parsed is not None else text)
    else:
        vals = _ensure_list(value)
    if sort:
        vals = sorted({v for v in vals if v})
    return vals


def _log1p_safe(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0).astype(float))


def load_meta(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"model_meta.json not found at {path}")
    with path.open("r") as fh:
        meta = json.load(fh)
    return meta


def load_reference_item_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"reference item_features parquet not found: {path}")
    df = pd.read_parquet(path)
    # Ensure required columns exist, even if empty
    for col in ["video_id", "source", "maker", "label", "series", "tag_ids", "performer_ids", "price"]:
        if col not in df.columns:
            if col in ("tag_ids", "performer_ids"):
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = ""
    return df


def _parse_since(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid --since value: {value}") from exc
    if dt.tzinfo is None:
        dt = datetime(dt.year, dt.month, dt.day, tzinfo=JST)
    dt_utc = dt.astimezone(timezone.utc)
    if dt_utc.time() == datetime.min.time():
        return dt_utc.date()
    return dt_utc


class FeatureSpace:
    """Utility copied from training script to build consistent categorical/multi-hot encodings."""

    def __init__(self, *, categorical_fields: Dict[str, Iterable[str]], multi_fields: Dict[str, Iterable[str]]):
        self.offsets: Dict[str, Sequence[int]] = {}
        self.multi_offsets: Dict[str, Sequence[int]] = {}
        self.cat_vocab: Dict[str, Dict[str, int]] = {}
        self.multi_vocab: Dict[str, Dict[str, int]] = {}
        offset = 0

        for field, values in categorical_fields.items():
            vocab = sorted({v for v in values if v})
            self.cat_vocab[field] = {v: i for i, v in enumerate(vocab)}
            self.offsets[field] = (offset, offset + len(vocab))
            offset += len(vocab)

        for field, values in multi_fields.items():
            vocab = sorted({v for v in values if v})
            self.multi_vocab[field] = {v: i for i, v in enumerate(vocab)}
            self.multi_offsets[field] = (offset, offset + len(vocab))
            offset += len(vocab)

        self.base_dim = offset

    def encode_categorical(self, vec: np.ndarray, field: str, value: str) -> None:
        mapping = self.cat_vocab.get(field)
        if not mapping:
            return
        start, _ = self.offsets[field]
        idx = mapping.get(value)
        if idx is not None:
            vec[start + idx] = 1.0

    def encode_multi(self, vec: np.ndarray, field: str, values: Iterable[str]) -> None:
        mapping = self.multi_vocab.get(field)
        if not mapping:
            return
        start, _ = self.multi_offsets[field]
        for val in values:
            idx = mapping.get(val)
            if idx is not None:
                vec[start + idx] = 1.0


class TwoTower(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        out = self.user_encoder(user_features)
        return F.normalize(out, p=2, dim=-1)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        out = self.item_encoder(item_features)
        return F.normalize(out, p=2, dim=-1)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        return (user_emb * item_emb).sum(dim=-1, keepdim=True)


def _ensure_ipv4_hostaddr(conninfo: str, allow_pooler: bool = True) -> str:
    parsed = urlsplit(conninfo)
    if parsed.scheme not in ("postgresql", "postgres", "postgresql+psycopg"):
        return conninfo

    query = parse_qs(parsed.query)
    if "sslmode" not in query:
        query["sslmode"] = ["require"]
    if "sslrootcert" not in query and os.environ.get("PGSSLROOTCERT"):
        query["sslrootcert"] = [os.environ["PGSSLROOTCERT"]]
    if query.get("hostaddr"):
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), parsed.fragment))

    host = parsed.hostname
    port = parsed.port or 5432
    if not host:
        return conninfo

    ipv4_addr = None
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, port):
            if family == socket.AF_INET:
                ipv4_addr = sockaddr[0]
                break
    except socket.gaierror:
        ipv4_addr = None

    if allow_pooler and not ipv4_addr:
        project_ref = _resolve_project_ref()
        if not project_ref:
            parts = parsed.hostname.split(".") if parsed.hostname else []
            if len(parts) >= 3 and parts[0] == "db":
                project_ref = parts[1]
        pooler_host = os.environ.get("SUPABASE_POOLER_HOST")
        if not pooler_host:
            region = os.environ.get("SUPABASE_REGION")
            if region:
                pooler_host = f"{region}.pooler.supabase.com"
        if pooler_host:
            pooler_port = os.environ.get("SUPABASE_POOLER_PORT") or "6543"
            username = parsed.username or ""
            password = parsed.password or ""
            if project_ref and username and "." not in username:
                username = f"{username}.{project_ref}"
            auth = quote(username)
            if password:
                auth = f"{auth}:{quote(password)}"
            netloc = f"{auth}@{pooler_host}:{pooler_port}"
            pooler_conn = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
            print(json.dumps({"info": "pooler_url_computed", "url": pooler_conn}, ensure_ascii=False))
            return _ensure_ipv4_hostaddr(pooler_conn, allow_pooler=False)

    if not ipv4_addr:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), parsed.fragment))

    query.setdefault("hostaddr", []).append(ipv4_addr)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), parsed.fragment))


def fetch_target_videos(
    db_url: str,
    *,
    since: Optional[datetime],
    limit: Optional[int],
    model_version: str,
    include_existing: bool,
) -> pd.DataFrame:
    sql = """
        SELECT
          v.id,
          v.source,
          v.maker,
          v.label,
          v.series,
          v.price,
          v.product_released_at,
          array_remove(array_agg(DISTINCT vt.tag_id), NULL) AS tag_ids,
          array_remove(array_agg(DISTINCT vp.performer_id), NULL) AS performer_ids,
          max(ve.model_version) AS current_model_version
        FROM public.videos v
        LEFT JOIN public.video_tags vt ON vt.video_id = v.id
        LEFT JOIN public.video_performers vp ON vp.video_id = v.id
        LEFT JOIN public.video_embeddings ve ON ve.video_id = v.id
        WHERE (%(since)s IS NULL OR v.product_released_at >= %(since)s)
          AND (
            %(include_existing)s
            OR ve.video_id IS NULL
            OR ve.model_version IS NULL
            OR ve.model_version <> %(model_version)s
          )
        GROUP BY v.id, v.source, v.maker, v.label, v.series, v.price, v.product_released_at
        ORDER BY v.product_released_at DESC NULLS LAST, v.id DESC
    """
    params = {
        "since": since,
        "include_existing": include_existing,
        "model_version": model_version,
    }
    if limit is not None and limit > 0:
        sql = sql + " LIMIT %(limit)s"
        params["limit"] = limit

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["video_id"])
    df = pd.DataFrame(rows)
    df.rename(columns={"id": "video_id"}, inplace=True)
    df["video_id"] = df["video_id"].astype(str)
    df["source"] = df.get("source", "").fillna("").astype(str)
    df["maker"] = df.get("maker", "").fillna("").astype(str)
    df["label"] = df.get("label", "").fillna("").astype(str)
    df["series"] = df.get("series", "").fillna("").astype(str)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = np.nan
    df["tag_ids"] = df.get("tag_ids", []).apply(lambda v: _normalize_array(v, sort=True))
    df["performer_ids"] = df.get("performer_ids", []).apply(lambda v: _normalize_array(v, sort=True))
    return df


@dataclass
class ItemFeatureBundle:
    item_space: FeatureSpace
    numeric_fields: List[str]


def build_item_feature_space(reference_df: pd.DataFrame, *, use_price_feature: bool) -> ItemFeatureBundle:
    cat_vocabs = {
        "source": reference_df.get("source", []).astype(str),
        "maker": reference_df.get("maker", []).astype(str),
        "label": reference_df.get("label", []).astype(str),
        "series": reference_df.get("series", []).astype(str),
    }
    tag_vocab: set[str] = set()
    for values in reference_df.get("tag_ids", []):
        tag_vocab.update(_normalize_array(values))
    performer_vocab: set[str] = set()
    for values in reference_df.get("performer_ids", []):
        performer_vocab.update(_normalize_array(values))

    item_space = FeatureSpace(
        categorical_fields=cat_vocabs,
        multi_fields={
            "tag_ids": tag_vocab,
            "performer_ids": performer_vocab,
        },
    )

    numeric_fields: List[str] = []
    if use_price_feature:
        numeric_fields.append("price")
    return ItemFeatureBundle(item_space=item_space, numeric_fields=numeric_fields)


def build_item_vectors(
    df: pd.DataFrame,
    *,
    bundle: ItemFeatureBundle,
    expected_dim: int,
) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    numeric_matrix: Optional[np.ndarray]

    if bundle.numeric_fields:
        numeric_df = df[bundle.numeric_fields].copy()
        if "price" in numeric_df.columns:
            numeric_df["price"] = _log1p_safe(numeric_df["price"].fillna(0))
        numeric_matrix = numeric_df.to_numpy(dtype=np.float32)
    else:
        numeric_matrix = np.zeros((len(df), 0), dtype=np.float32)

    for idx, row in enumerate(df.itertuples(index=False)):
        vec = np.zeros(bundle.item_space.base_dim + numeric_matrix.shape[1], dtype=np.float32)
        bundle.item_space.encode_categorical(vec, "source", str(getattr(row, "source", "")))
        bundle.item_space.encode_categorical(vec, "maker", str(getattr(row, "maker", "")))
        bundle.item_space.encode_categorical(vec, "label", str(getattr(row, "label", "")))
        bundle.item_space.encode_categorical(vec, "series", str(getattr(row, "series", "")))
        bundle.item_space.encode_multi(vec, "tag_ids", _ensure_list(getattr(row, "tag_ids", [])))
        bundle.item_space.encode_multi(vec, "performer_ids", _ensure_list(getattr(row, "performer_ids", [])))
        if numeric_matrix.size > 0:
            vec[bundle.item_space.base_dim :] = numeric_matrix[idx]
        vectors[str(getattr(row, "video_id"))] = vec

    computed_dim = bundle.item_space.base_dim + numeric_matrix.shape[1]
    if computed_dim != expected_dim:
        print(
            json.dumps(
                {
                    "warn": "item_feature_dim_mismatch",
                    "expected": expected_dim,
                    "computed": computed_dim,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    return vectors


def encode_embeddings(model: TwoTower, item_vectors: Dict[str, np.ndarray], device: torch.device) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        iterator = tqdm(item_vectors.items(), desc="Encoding videos", unit="video")
        for video_id, vec in iterator:
            tensor = torch.from_numpy(vec).unsqueeze(0).to(device)
            embedding = model.encode_item(tensor).cpu().numpy()[0]
            rows.append({"video_id": video_id, "embedding": embedding.tolist()})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    db_url = args.db_url or os.environ.get("SUPABASE_DB_URL") or os.environ.get("REMOTE_DATABASE_URL")
    if not db_url:
        raise ValueError("Database URL not provided. Set --db-url or SUPABASE_DB_URL / REMOTE_DATABASE_URL env.")
    db_url = _ensure_ipv4_hostaddr(db_url)

    meta = load_meta(args.model_meta)
    model_version = str(meta.get("run_id") or "unknown")
    embedding_dim = int(meta["embedding_dim"])
    hidden_dim = int(meta["hidden_dim"])
    user_feature_dim = int(meta["user_feature_dim"])
    item_feature_dim = int(meta["item_feature_dim"])
    use_price_feature = bool(meta.get("use_price_feature", False))

    reference_df = load_reference_item_features(args.reference_item_features)
    bundle = build_item_feature_space(reference_df, use_price_feature=use_price_feature)

    print(
        json.dumps(
            {
                "info": "item_feature_space",
                "model_version": model_version,
                "categorical_dims": bundle.item_space.base_dim,
                "numeric_fields": bundle.numeric_fields,
                "expected_item_dim": item_feature_dim,
            },
            ensure_ascii=False,
        )
    )

    print(
        json.dumps(
            {
                "debug": "connection_info",
                "db_url": db_url,
                "env_sslmode": os.environ.get("PGSSLMODE"),
                "env_sslrootcert": os.environ.get("PGSSLROOTCERT"),
            },
            ensure_ascii=False,
        )
    )

    since_dt = _parse_since(args.since)
    print(
        json.dumps(
            {
                "debug": "db_connect_params",
                "db_url": db_url,
                "has_pgsslrootcert": bool(os.environ.get("PGSSLROOTCERT")),
            },
            ensure_ascii=False,
        )
    )

    selected_df = fetch_target_videos(
        db_url,
        since=since_dt,
        limit=args.limit,
        model_version=model_version,
        include_existing=args.include_existing,
    )

    if selected_df.empty:
        print(
            json.dumps(
                {
                    "event": "no_videos_selected",
                    "since": since_dt.isoformat() if since_dt else None,
                    "include_existing": args.include_existing,
                },
                ensure_ascii=False,
            )
        )
        return

    print(
        json.dumps(
            {
                "info": "video_candidates",
                "count": len(selected_df),
                "since": since_dt.isoformat() if since_dt else None,
                "include_existing": args.include_existing,
            },
            ensure_ascii=False,
        )
    )

    item_vectors = build_item_vectors(selected_df, bundle=bundle, expected_dim=item_feature_dim)
    if not item_vectors:
        print(json.dumps({"event": "no_vectors_generated"}, ensure_ascii=False))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(
        user_dim=user_feature_dim,
        item_dim=item_feature_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    )
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    embeddings_df = encode_embeddings(model, item_vectors, device)
    embeddings_df["model_version"] = model_version
    print(json.dumps({"event": "video_embeddings_encoded", "count": len(embeddings_df)}, ensure_ascii=False))

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    embeddings_df.sort_values("video_id").to_parquet(output_path, index=False)
    print(json.dumps({"event": "video_embeddings_saved", "path": str(output_path)}, ensure_ascii=False))

    # Persist run metadata for downstream steps
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": model_version,
        "rows": int(len(embeddings_df)),
        "since": since_dt.isoformat() if since_dt else None,
        "include_existing": args.include_existing,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"event": "video_embeddings_summary_saved", "path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
