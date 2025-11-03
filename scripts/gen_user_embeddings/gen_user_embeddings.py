#!/usr/bin/env python3
import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from psycopg.rows import dict_row
from tqdm import tqdm
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit, quote
import urllib.error
import urllib.request
import socket


REQUIRED_COLUMNS = [
    "reviewer_id",
    "preferred_tag_ids",
    "like_count_30d",
    "positive_ratio_30d",
    "signup_days",
]


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
    parser = argparse.ArgumentParser(description="Generate live user embeddings from Supabase data.")
    parser.add_argument("--db-url", type=str, default=None, help="Postgres connection string (defaults to SUPABASE_DB_URL / REMOTE_DATABASE_URL).")
    parser.add_argument(
        "--reference-user-features",
        type=Path,
        default=Path("ml/data/processed/two_tower/latest/user_features.parquet"),
        help="Reference user_features parquet used during training (to rebuild vocab).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/artifacts/latest/two_tower_latest.pt"),
        help="Trained Two-Tower PyTorch weights.",
    )
    parser.add_argument(
        "--model-meta",
        type=Path,
        default=Path("ml/artifacts/latest/model_meta.json"),
        help="Model metadata JSON containing feature dimensions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/live"),
        help="Directory to write generated embeddings (user_embeddings.parquet).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="user_embeddings.parquet",
        help="Filename for generated embeddings inside --output-dir.",
    )
    parser.add_argument(
        "--no-copy-video-embeddings",
        dest="copy_video_embeddings",
        action="store_false",
        help="Skip copying video embeddings into --output-dir.",
    )
    parser.add_argument(
        "--video-embeddings-src",
        type=Path,
        default=Path("ml/artifacts/latest/video_embeddings.parquet"),
        help="Source video embeddings parquet used when --copy-video-embeddings is set.",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=0,
        help="Minimum LIKE interactions (all-time) required to generate a user embedding.",
    )
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Optional hard limit of users to encode (useful for debugging).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data and report counts without writing parquet.")
    parser.set_defaults(copy_video_embeddings=True)
    return parser.parse_args()


def _ensure_list(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v and str(v).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(value)]


def _log1p_safe(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0).astype(float))


def _normalize_array(value, *, sort: bool = False) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = [v.strip() for v in text.split(',') if v.strip()]
        vals = _ensure_list(parsed if parsed is not None else value)
    else:
        vals = _ensure_list(value)
    if sort:
        vals = sorted(set(vals))
    return vals


def _normalize_uuid(value: object, column: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except Exception as exc:
        raise ValueError(f"Invalid UUID in column '{column}': {value}") from exc


def _load_live_user_features(db_url: str, min_interactions: int) -> pd.DataFrame:
    sql = (
        "with base as ("
        "  select"
        "    uvd.user_id,"
        "    (array_agg(uvd.video_id order by uvd.created_at desc)"
        "      filter (where uvd.decision_type = 'like'))[1:20] as recent_positive_video_ids,"
        "    count(*) filter (where uvd.decision_type = 'like' and uvd.created_at >= now() - interval '30 days') as like_count_30d,"
        "    count(*) filter (where uvd.created_at >= now() - interval '30 days') as total_count_30d,"
        "    count(*) filter (where uvd.decision_type = 'like') as like_count_all"
        "  from public.user_video_decisions uvd"
        "  group by uvd.user_id"
        "), tag_pref as ("
        "  select"
        "    ranked.user_id,"
        "    (array_agg(ranked.tag_id order by ranked.like_count desc nulls last))[1:10] as preferred_tag_ids"
        "  from ("
        "    select"
        "      uvd.user_id,"
        "      vt.tag_id,"
        "      count(*) filter (where uvd.decision_type = 'like') as like_count"
        "    from public.user_video_decisions uvd"
        "    join public.video_tags vt on vt.video_id = uvd.video_id"
        "    group by uvd.user_id, vt.tag_id"
        "  ) ranked"
        "  group by ranked.user_id"
        "), profiles as ("
        "  select"
        "    user_id,"
        "    greatest(0, cast(date_part('day', now() - created_at) as int)) as signup_days"
        "  from public.profiles"
        ") "
        "select"
        "  b.user_id as reviewer_id,"
        "  coalesce(b.recent_positive_video_ids, array[]::uuid[]) as recent_positive_video_ids,"
        "  coalesce(b.like_count_30d, 0) as like_count_30d,"
        "  coalesce(b.like_count_all, 0) as like_count_all,"
        "  case when coalesce(b.total_count_30d, 0) > 0"
        "       then b.like_count_30d::double precision / b.total_count_30d"
        "       else null end as positive_ratio_30d,"
        "  p.signup_days,"
        "  coalesce(tp.preferred_tag_ids, array[]::uuid[]) as preferred_tag_ids"
        " from base b"
        " left join profiles p on p.user_id = b.user_id"
        " left join tag_pref tp on tp.user_id = b.user_id"
    )
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["reviewer_id"])
    df = pd.DataFrame(rows)
    df["reviewer_id"] = df["reviewer_id"].apply(lambda v: str(v) if v is not None else "")
    df["preferred_tag_ids"] = df["preferred_tag_ids"].apply(lambda v: _normalize_array(v, sort=True))
    df["like_count_30d"] = pd.to_numeric(df["like_count_30d"], errors="coerce").fillna(0).astype("int64")
    df["like_count_all"] = pd.to_numeric(df.get("like_count_all", 0), errors="coerce").fillna(0).astype("int64")
    df["positive_ratio_30d"] = pd.to_numeric(df["positive_ratio_30d"], errors="coerce")
    df["signup_days"] = pd.to_numeric(df["signup_days"], errors="coerce").round().astype("float32")

    if min_interactions > 0:
        df = df[df["like_count_all"] >= min_interactions].copy()
    return df


class UserFeatureSpace:
    def __init__(self, preferred_tag_vocab: Iterable[str], numeric_fields: List[str]):
        vocab = sorted({v for v in preferred_tag_vocab if v})
        self.preferred_vocab = {v: i for i, v in enumerate(vocab)}
        self.numeric_fields = numeric_fields
        self.base_dim = len(self.preferred_vocab)

    def encode(self, record_tags: Iterable[str], numeric_values: np.ndarray) -> np.ndarray:
        vec = np.zeros(self.base_dim + len(numeric_values), dtype=np.float32)
        for tag in record_tags:
            idx = self.preferred_vocab.get(tag)
            if idx is not None:
                vec[idx] = 1.0
        vec[self.base_dim :] = numeric_values
        return vec


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

    def encode_user(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_encoder(x), p=2, dim=-1)


def load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"model meta not found: {meta_path}")
    return json.loads(meta_path.read_text())


def load_reference_user_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"reference user_features parquet not found: {path}")
    df = pd.read_parquet(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"reference file missing columns: {missing}")
    df["preferred_tag_ids"] = df["preferred_tag_ids"].apply(lambda v: _normalize_array(v, sort=True))
    return df


def build_user_vectors(
    encoder: UserFeatureSpace,
    df: pd.DataFrame,
    expected_dim: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    numeric_fields = ["like_count_30d", "positive_ratio_30d", "signup_days"]
    for col in numeric_fields:
        if col not in df.columns:
            df[col] = 0
    numeric = df[numeric_fields].copy()
    numeric["like_count_30d"] = _log1p_safe(numeric["like_count_30d"].fillna(0))
    numeric["signup_days"] = _log1p_safe(numeric["signup_days"].fillna(0))
    numeric["positive_ratio_30d"] = numeric["positive_ratio_30d"].fillna(0).clip(0, 1)
    numeric_array = numeric.to_numpy(dtype=np.float32)

    vectors: Dict[str, np.ndarray] = {}
    for idx, row in enumerate(df.itertuples(index=False)):
        reviewer_id = str(row.reviewer_id)
        try:
            reviewer_id = _normalize_uuid(reviewer_id, "reviewer_id")
        except ValueError:
            continue
        vec = encoder.encode(
            _ensure_list(getattr(row, "preferred_tag_ids", [])),
            numeric_array[idx],
        )
        if expected_dim is not None and expected_dim != vec.size:
            padded = np.zeros(expected_dim, dtype=np.float32)
            length = min(vec.size, expected_dim)
            padded[:length] = vec[:length]
            vec = padded
        vectors[reviewer_id] = vec
    return vectors


def encode_embeddings(
    model: TwoTower,
    vectors: Dict[str, np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for user_id, vec in tqdm(vectors.items(), desc="Encoding user embeddings"):
            tensor = torch.from_numpy(vec).unsqueeze(0).to(device)
            embedding = model.encode_user(tensor).cpu().numpy()[0]
            rows.append({"reviewer_id": user_id, "embedding": embedding.tolist()})
    return pd.DataFrame(rows)


def _ensure_ipv4_hostaddr(conninfo: str, allow_pooler: bool = True) -> str:
    try:
        parsed = urlsplit(conninfo)
    except Exception:
        return conninfo
    if not parsed.hostname:
        return conninfo
    query = parse_qs(parsed.query, keep_blank_values=True)
    if "sslmode" not in query:
        query["sslmode"] = ["require"]
    if "sslrootcert" not in query:
        query["sslrootcert"] = ["/etc/ssl/certs/ca-certificates.crt"]
    if "hostaddr" in query:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), parsed.fragment))

    addrinfo = []
    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, parsed.port or 5432, family=socket.AF_INET)
    except socket.gaierror:
        addrinfo = []
    ipv4_addr = next((info[4][0] for info in addrinfo if info[0] == socket.AF_INET), None)

    if not ipv4_addr and allow_pooler:
        project_ref = _resolve_project_ref()
        if not project_ref:
            parts = parsed.hostname.split(".")
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


def main() -> None:
    args = parse_args()

    db_url = args.db_url or os.environ.get("SUPABASE_DB_URL") or os.environ.get("REMOTE_DATABASE_URL")
    if not db_url:
        raise ValueError("Database URL not provided. Set --db-url or SUPABASE_DB_URL / REMOTE_DATABASE_URL env.")

    db_url = _ensure_ipv4_hostaddr(db_url)

    meta = load_meta(args.model_meta)
    user_feature_dim = int(meta["user_feature_dim"])
    item_feature_dim = int(meta["item_feature_dim"])
    embedding_dim = int(meta["embedding_dim"])
    hidden_dim = int(meta["hidden_dim"])

    reference_df = load_reference_user_features(args.reference_user_features)
    preferred_vocab = set()
    for values in reference_df["preferred_tag_ids"]:
        preferred_vocab.update(_ensure_list(values))
    print(
        json.dumps(
            {
                "info": "reference_vocab",
                "preferred_tags": len(preferred_vocab),
                "reference_rows": len(reference_df),
            },
            ensure_ascii=False,
        )
    )

    encoder = UserFeatureSpace(preferred_vocab, numeric_fields=["like_count_30d", "positive_ratio_30d", "signup_days"])

    live_df = _load_live_user_features(db_url, args.min_interactions)
    if args.limit_users:
        live_df = live_df.head(args.limit_users)
    if live_df.empty:
        print(json.dumps({"event": "no_users_found"}, ensure_ascii=False))
        return

    user_vectors = build_user_vectors(encoder, live_df, expected_dim=user_feature_dim)
    if not user_vectors:
        print(json.dumps({"event": "no_valid_users"}, ensure_ascii=False))
        return

    if encoder.base_dim + 3 != user_feature_dim:
        print(
            json.dumps(
                {
                    "info": "user_feature_dim_adjusted",
                    "expected": user_feature_dim,
                    "computed": encoder.base_dim + 3,
                },
                ensure_ascii=False,
            )
        )

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

    embeddings_df = encode_embeddings(model, user_vectors, device)
    print(json.dumps({"event": "user_embeddings_encoded", "count": len(embeddings_df)}, ensure_ascii=False))

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    embeddings_df.sort_values("reviewer_id").to_parquet(output_path, index=False)
    print(json.dumps({"event": "user_embeddings_saved", "path": str(output_path)}, ensure_ascii=False))

    if args.copy_video_embeddings and args.video_embeddings_src.exists():
        target = args.output_dir / args.video_embeddings_src.name
        if target.resolve() != args.video_embeddings_src.resolve():
            target.write_bytes(args.video_embeddings_src.read_bytes())
            print(json.dumps({"event": "video_embeddings_copied", "path": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
