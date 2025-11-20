#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from os import path as _path
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import psycopg


DEFAULT_PROCESSED_ROOT = Path("ml/data/processed/two_tower")
DEFAULT_LATEST_DIR = DEFAULT_PROCESSED_ROOT / "latest"
DEFAULT_SNAPSHOT_ROOT = DEFAULT_PROCESSED_ROOT / "runs"
DEFAULT_MAKER_DISTRIBUTION_LATEST = DEFAULT_LATEST_DIR / "maker_distribution.csv"


JST = timezone(timedelta(hours=9))


def _generate_run_id() -> str:
    return datetime.now(JST).strftime("%Y%m%d_%H%M%S")


@dataclass
class PrepConfig:
    mode: str  # 'explicit' or 'reviews'
    input_csv: Path
    decisions_csv: Path | None
    db_url: str | None
    join_on: str  # 'auto' | 'product_url' | 'external_id'
    min_positive_stars: int
    negatives_per_positive: int
    max_negative_stars: int | None
    val_ratio: float
    out_train: Path
    out_val: Path
    out_items: Path
    out_user_features: Path
    run_id: str | None
    snapshot_root: Path | None
    snapshot_inputs: bool
    summary_path: Path | None
    keep_legacy_outputs: bool


def build_interactions_from_reviews(
    df: pd.DataFrame,
    min_positive_stars: int,
    k_neg: int,
    max_negative_stars: int | None,
) -> pd.DataFrame:
    # Normalize columns
    for c in ("product_url", "reviewer_id", "stars"):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in reviews CSV")

    if max_negative_stars is None:
        max_negative_stars = min_positive_stars - 1

    df = df[["product_url", "reviewer_id", "stars"]].copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    df["stars"] = df["stars"].astype(int)

    # Aggregate by reviewer/video to avoid conflicting labels
    grouped = df.groupby(["reviewer_id", "product_url"], as_index=False)["stars"].max()

    # Positive interactions
    pos = grouped[grouped["stars"] >= min_positive_stars][["reviewer_id", "product_url"]].copy()
    pos["label"] = 1

    # Explicit negative interactions (e.g., stars <= 3)
    neg_explicit = pd.DataFrame(columns=["reviewer_id", "product_url"])
    if max_negative_stars is not None:
        neg_explicit = grouped[grouped["stars"] <= max_negative_stars][["reviewer_id", "product_url"]].copy()
        neg_explicit["label"] = 0

    # Build candidate item universe (all reviewed items)
    items = grouped["product_url"].unique()
    items_set = set(items)

    # For each user, sample negatives from items they have not positively interacted with nor explicitly rated negative
    neg_explicit_map = (
        neg_explicit.groupby("reviewer_id")["product_url"].apply(set).to_dict() if not neg_explicit.empty else {}
    )

    user_pos = pos.groupby("reviewer_id")["product_url"].apply(set)

    rng = np.random.default_rng(42)
    neg_rows = []
    for user, liked in user_pos.items():
        already_disliked = neg_explicit_map.get(user, set())
        disliked_pool = list(items_set - liked - already_disliked)
        if not disliked_pool:
            continue
        n_pos = len(liked)
        n_neg = min(len(disliked_pool), n_pos * k_neg)
        if n_neg <= 0:
            continue
        sampled = rng.choice(disliked_pool, size=n_neg, replace=False)
        for it in sampled:
            neg_rows.append((user, it, 0))

    neg = (
        pd.DataFrame(neg_rows, columns=["reviewer_id", "product_url", "label"])
        if neg_rows
        else pd.DataFrame(columns=["reviewer_id", "product_url", "label"])
    )

    interactions = pd.concat([pos, neg_explicit, neg], ignore_index=True)
    interactions = (
        interactions.sort_values("label", ascending=False)
        .drop_duplicates(subset=["reviewer_id", "product_url"], keep="first")
        .reset_index(drop=True)
    )
    return interactions


def build_interactions_from_explicit(df: pd.DataFrame) -> pd.DataFrame:
    # Support flexible column names for user & item and decision/label
    user_col = "reviewer_id" if "reviewer_id" in df.columns else ("user_id" if "user_id" in df.columns else None)
    item_col = "product_url" if "product_url" in df.columns else ("item_id" if "item_id" in df.columns else None)
    if user_col is None or item_col is None:
        raise ValueError("Expected user column (reviewer_id|user_id) and item column (product_url|item_id)")

    if "label" in df.columns:
        out = df[[user_col, item_col, "label"]].copy()
        out.rename(columns={user_col: "reviewer_id", item_col: "product_url"}, inplace=True)
        out["label"] = out["label"].astype(int).clip(0, 1)
    elif "decision" in df.columns:
        tmp = df[[user_col, item_col, "decision"]].copy()
        tmp.rename(columns={user_col: "reviewer_id", item_col: "product_url"}, inplace=True)
        mapping = {"LIKE": 1, "DISLIKE": 0, "SKIP": None, "UNKNOWN": None}
        lab = tmp["decision"].astype(str).str.upper().map(mapping)
        out = tmp[["reviewer_id", "product_url"]].copy()
        out["label"] = lab
        out = out.dropna(subset=["label"]).astype({"label": int})
    else:
        raise ValueError("Expected either 'label' or 'decision' column in explicit data")

    out = out.drop_duplicates()
    return out


def train_val_split(df: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions per user so that each user appears in both train and val where possible.
    This avoids the cold-start issue of holding out entire users (which makes Two-Tower ID embeddings
    unable to generalize).
    """
    if df is None or len(df) == 0:
        # Return empty splits with the original columns
        cols = df.columns if df is not None else ["reviewer_id", "product_url", "label"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    rng = np.random.default_rng(123)
    chunks_train = []
    chunks_val = []
    for user, g in df.groupby("reviewer_id", sort=False):
        n = len(g)
        if n <= 1:
            chunks_train.append(g)
            continue
        k = max(1, int(n * val_ratio))
        idx = np.arange(n)
        val_idx = set(rng.choice(idx, size=k, replace=False))
        mask = np.array([i in val_idx for i in range(n)])
        gv = g.iloc[mask]
        gt = g.iloc[~mask]
        if gt.empty:  # ensure at least one train sample
            gt = gv.iloc[:1]
            gv = gv.iloc[1:]
        chunks_train.append(gt)
        if not gv.empty:
            chunks_val.append(gv)
    train = pd.concat(chunks_train, ignore_index=True) if chunks_train else pd.DataFrame(columns=df.columns)
    val = pd.concat(chunks_val, ignore_index=True) if chunks_val else pd.DataFrame(columns=df.columns)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _extract_cid_from_url(url: str) -> str | None:
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(str(url))
        qs = parse_qs(parsed.query)
        if 'cid' in qs and len(qs['cid']) > 0:
            return str(qs['cid'][0]).strip().lower()
        # fallback: look for 'cid=' pattern anywhere (handles path style "=/cid=...")
        import re
        m = re.search(r"cid=([^&/#]+)", str(url), flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
    except Exception:
        pass
    return None


def _ensure_cid_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'cid' not in df.columns and 'product_url' in df.columns:
        df = df.copy()
        df['cid'] = df['product_url'].apply(_extract_cid_from_url)
    return df


def _load_videos_df(db_url: str | None) -> pd.DataFrame:
    if not db_url:
        raise ValueError("db_url is required to load videos metadata")
    try:
        with psycopg.connect(db_url) as conn:
            sql = (
                "select v.id, v.product_url, v.external_id, v.title, v.series, v.maker, v.label, "
                "v.source, v.price, "
                "coalesce(array_agg(distinct vt.tag_id) filter (where vt.tag_id is not null and coalesce(tg.use_for_training, true)), array[]::uuid[]) as tag_ids, "
                "coalesce(array_agg(distinct t.name) filter (where t.name is not null and coalesce(tg.use_for_training, true)), array[]::text[]) as tag_names, "
                "coalesce(array_agg(distinct vp.performer_id) filter (where vp.performer_id is not null), array[]::uuid[]) as performer_ids "
                "from public.videos v "
                "left join public.video_tags vt on vt.video_id = v.id "
                "left join public.tags t on t.id = vt.tag_id "
                "left join public.tag_groups tg on tg.id = t.tag_group_id "
                "left join public.video_performers vp on vp.video_id = v.id "
                "group by v.id, v.product_url, v.external_id, v.title, v.series, v.maker, v.label, v.source, v.price"
            )
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [
                    "id", "product_url", "external_id", "title", "series", "maker", "label",
                    "source", "price", "tag_ids", "tag_names", "performer_ids"
                ]
                vids = pd.DataFrame(rows, columns=cols)
                if "tag_names" in vids.columns:
                    vids["tags"] = vids["tag_names"].apply(
                        lambda names: ",".join(sorted({str(n) for n in names if n is not None})) if isinstance(names, (list, tuple)) else ("" if pd.isna(names) else str(names))
                    )
                return vids
    except Exception as e:
        raise RuntimeError(f"Failed to load videos from DB: {e}") from e


def maybe_join_videos(interactions: pd.DataFrame, join_on: str, db_url: str | None) -> tuple[pd.DataFrame, pd.DataFrame | None, int, bool, str]:
    """
    Join interactions with the videos table fetched from Postgres to attach canonical video_id
    and item attributes. Returns (interactions_with_video_id, items_df_or_none).
    """
    vids = _load_videos_df(db_url)
    src = 'db'
    # Expect at least id plus one of (product_url, external_id)
    if "id" not in vids.columns:
        raise ValueError("videos table must contain column: id")
    # Normalize candidate join keys on videos side
    vids = vids.copy()
    if 'external_id' in vids.columns:
        vids['external_id'] = vids['external_id'].astype(str).str.strip().str.lower()
    if 'product_url' in vids.columns:
        vids['product_url'] = vids['product_url'].astype(str)

    join_used = 'product_url'
    if (join_on == 'external_id' or (join_on == 'auto' and 'external_id' in vids.columns)):
        # Build CID from interactions.product_url and join to videos.external_id
        interactions = _ensure_cid_column(interactions)
        if 'cid' not in interactions.columns:
            raise ValueError("Could not derive cid from product_url; cannot join on external_id")
        join_used = 'external_id'
        joined = interactions.merge(vids[['external_id', 'id']], left_on='cid', right_on='external_id', how='left')
    else:
        # Fallback to product_url join
        if 'product_url' not in vids.columns:
            raise ValueError("videos table must contain product_url when join_on != external_id")
        joined = interactions.merge(vids[["product_url", "id"]], on="product_url", how="left")
    missing = int(joined["id"].isna().sum())
    item_cols = [c for c in [
        "id", "product_url", "external_id", "title", "series", "maker", "label",
        "source", "price", "tags", "tag_ids", "performer_ids"
    ] if c in vids.columns]
    items = vids[item_cols].rename(columns={"id": "video_id"}).copy()
    # Standardize tags to string (comma-separated) if present as array
    if 'tags' in items.columns and not pd.api.types.is_string_dtype(items['tags']):
        items['tags'] = items['tags'].apply(lambda v: ','.join(v) if isinstance(v, (list, tuple)) else ('' if pd.isna(v) else str(v)))
    return joined.rename(columns={"id": "video_id"}), items, missing, True, join_used


def _resolve_snapshot_paths(cfg: PrepConfig) -> tuple[Path | None, Dict[str, Path]]:
    if cfg.run_id is None or cfg.snapshot_root is None:
        return None, {}
    run_dir = cfg.snapshot_root / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {
        "train": run_dir / "interactions_train.parquet",
        "val": run_dir / "interactions_val.parquet",
        "items": run_dir / "item_features.parquet",
        "user_features": run_dir / "user_features.parquet",
        "summary": run_dir / "summary.json",
        "maker_distribution": run_dir / "maker_distribution.csv",
    }
    return run_dir, files


def _copy_if_exists(src: Path | None, dest: Path) -> bool:
    if src is None:
        return False
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _snapshot_inputs(cfg: PrepConfig, run_dir: Path, copied_files: Dict[str, str]) -> None:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    copied_any = False

    if cfg.input_csv.exists():
        dest = inputs_dir / cfg.input_csv.name
        shutil.copy2(cfg.input_csv, dest)
        copied_files["input_csv_copy"] = str(dest)
        copied_any = True
    if cfg.decisions_csv is not None and cfg.decisions_csv.exists():
        dest = inputs_dir / cfg.decisions_csv.name
        shutil.copy2(cfg.decisions_csv, dest)
        copied_files["decisions_csv_copy"] = str(dest)
        copied_any = True
    if not copied_any:
        copied_files["inputs_copied"] = "none"


def _write_summary(summary: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_maker_distribution(interactions: pd.DataFrame, items_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if items_df is None or "maker" not in items_df.columns or interactions.empty:
        return None
    if "video_id" not in interactions.columns:
        return None
    interactions = interactions.copy()
    items_df = items_df.copy()
    interactions["video_id"] = interactions["video_id"].astype(str)
    if "video_id" in items_df.columns:
        items_df["video_id"] = items_df["video_id"].astype(str)
    merged = interactions[["reviewer_id", "video_id", "label"]].merge(
        items_df[["video_id", "maker"]], on="video_id", how="left"
    )
    merged["maker"] = merged["maker"].fillna("unknown")
    positives = merged[merged["label"] == 1]
    if positives.empty:
        return None

    stats = (
        positives.groupby("maker")
        .agg(
            recommendations=("label", "size"),
            unique_items=("video_id", "nunique"),
            user_coverage=("reviewer_id", "nunique"),
        )
        .reset_index()
    )
    total_recs = int(stats["recommendations"].sum())
    if total_recs > 0:
        stats["share"] = stats["recommendations"] / total_recs
    else:
        stats["share"] = 0.0
    return stats.sort_values("recommendations", ascending=False)


def _write_maker_distribution(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _normalize_to_str_list(value, *, sort: bool = False) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if v is not None]
    elif value is None or pd.isna(value):
        items = []
    else:
        items = [str(value)]
    return sorted(items) if sort else items


def _build_user_features_from_interactions(interactions: pd.DataFrame, items_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if interactions is None or interactions.empty:
        return None
    df = interactions.copy()
    video_col = "video_id" if "video_id" in df.columns else "product_url"
    df["reviewer_id"] = df["reviewer_id"].astype(str)
    df[video_col] = df[video_col].astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    positives = df[df.get("label", 0) == 1]
    total_counts = df.groupby("reviewer_id").size()
    like_counts = positives.groupby("reviewer_id").size()
    recent_map = positives.groupby("reviewer_id")[video_col].apply(lambda s: s.tolist()[-20:])

    tag_lookup: dict[str, list[str]] = {}
    if items_df is not None and "video_id" in items_df.columns and "tag_ids" in items_df.columns:
        tag_lookup = {
            str(row.video_id): _normalize_to_str_list(row.tag_ids, sort=False)
            for row in items_df.itertuples(index=False)
        }

    rows = []
    for reviewer, total in total_counts.items():
        like_count = int(like_counts.get(reviewer, 0))
        ratio = (like_count / total) if total > 0 else None
        recent = recent_map.get(reviewer, [])
        preferred_tags: list[str] = []
        if tag_lookup and recent:
            tag_counter = Counter()
            for vid in recent:
                tag_counter.update(tag_lookup.get(vid, []))
            preferred_tags = [tag for tag, _ in tag_counter.most_common(10)]
        rows.append(
            {
                "reviewer_id": reviewer,
                "recent_positive_video_ids": recent,
                "like_count_30d": like_count,
                "positive_ratio_30d": float(ratio) if ratio is not None else None,
                "signup_days": 0,
                "preferred_tag_ids": preferred_tags,
            }
        )
    return pd.DataFrame(rows)


def _load_user_features(db_url: str | None) -> pd.DataFrame | None:
    if not db_url:
        return None
    sql = (
        "with base as ("
        "  select"
        "    uvd.user_id,"
        "    (array_agg(uvd.video_id order by uvd.created_at desc)"
        "      filter (where uvd.decision_type = 'like'))[:20] as recent_positive_video_ids,"
        "    count(*) filter (where uvd.decision_type = 'like' and uvd.created_at >= now() - interval '30 days') as like_count_30d,"
        "    count(*) filter (where uvd.created_at >= now() - interval '30 days') as total_count_30d"
        "  from public.user_video_decisions uvd"
        "  group by uvd.user_id"
        "), tag_pref as ("
        "  select"
        "    ranked.user_id,"
        "    (array_agg(ranked.tag_id order by ranked.like_count desc nulls last))[:10] as preferred_tag_ids"
        "  from ("
        "    select"
        "      uvd.user_id,"
        "      vt.tag_id,"
        "      count(*) filter (where uvd.decision_type = 'like') as like_count"
        "    from public.user_video_decisions uvd"
        "    join public.video_tags vt on vt.video_id = uvd.video_id"
        "    join public.tags t on t.id = vt.tag_id"
        "    left join public.tag_groups tg on tg.id = t.tag_group_id"
        "    where coalesce(tg.use_for_training, true)"
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
        "  case when coalesce(b.total_count_30d, 0) > 0"
        "       then b.like_count_30d::double precision / b.total_count_30d"
        "       else null end as positive_ratio_30d,"
        "  p.signup_days,"
        "  coalesce(tp.preferred_tag_ids, array[]::uuid[]) as preferred_tag_ids"
        " from base b"
        " left join profiles p on p.user_id = b.user_id"
        " left join tag_pref tp on tp.user_id = b.user_id"
    )
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [
                    "reviewer_id",
                    "recent_positive_video_ids",
                    "like_count_30d",
                    "positive_ratio_30d",
                    "signup_days",
                    "preferred_tag_ids",
                ]
                df = pd.DataFrame(rows, columns=cols)
    except Exception as e:
        sys.stderr.write(f"Warning: failed to load user features from DB: {e}\n")
        return None
    if df.empty:
        return df
    df["reviewer_id"] = df["reviewer_id"].astype(str)
    if "recent_positive_video_ids" in df.columns:
        df["recent_positive_video_ids"] = df["recent_positive_video_ids"].apply(_normalize_to_str_list)
    if "preferred_tag_ids" in df.columns:
        df["preferred_tag_ids"] = df["preferred_tag_ids"].apply(lambda v: _normalize_to_str_list(v, sort=True))
    if "like_count_30d" in df.columns:
        df["like_count_30d"] = pd.to_numeric(df["like_count_30d"], errors="coerce").fillna(0).astype("int64")
    if "signup_days" in df.columns:
        df["signup_days"] = pd.to_numeric(df["signup_days"], errors="coerce").round().astype("Int64")
    if "positive_ratio_30d" in df.columns:
        df["positive_ratio_30d"] = pd.to_numeric(df["positive_ratio_30d"], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser(description="Prepare Two-Tower interactions dataset.")
    ap.add_argument("--mode", choices=["explicit", "reviews"], default="explicit", help="Input format: explicit (LIKE/DISLIKE) or reviews (stars). Default: explicit")
    ap.add_argument("--input", required=True, type=Path, help="Path to primary input CSV. explicit: (reviewer_id|user_id,product_url|item_id,label|decision). reviews: (product_url,reviewer_id,stars)")
    ap.add_argument("--decisions-csv", type=Path, default=None, help="Optional: user_video_decisions CSV to merge (same columns as explicit mode). Default: not used")
    ap.add_argument("--db-url", type=str, required=True, help="Postgres connection string to fetch videos and user metadata.")
    ap.add_argument("--join-on", choices=["auto", "product_url", "external_id"], default="auto", help="Join key to match interactions to videos (default: auto)")
    ap.add_argument("--min-stars", type=int, default=4, help="Minimum stars to treat as positive (default: 4)")
    ap.add_argument("--neg-per-pos", type=int, default=3, help="Number of negatives per positive (default: 3)")
    ap.add_argument(
        "--max-negative-stars",
        type=int,
        default=None,
        help="Maximum stars to treat as explicit negative (default: min-stars-1). Use a negative value to disable.",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation user ratio (default: 0.2)")
    ap.add_argument("--out-train", type=Path, default=DEFAULT_LATEST_DIR / "interactions_train.parquet")
    ap.add_argument("--out-val", type=Path, default=DEFAULT_LATEST_DIR / "interactions_val.parquet")
    ap.add_argument("--out-items", type=Path, default=DEFAULT_LATEST_DIR / "item_features.parquet", help="Output path for per-item attributes derived from the videos table")
    ap.add_argument("--out-user-features", type=Path, default=DEFAULT_LATEST_DIR / "user_features.parquet", help="Output path for aggregated user features (requires --db-url)")
    ap.add_argument("--run-id", type=str, default=None, help="Optional identifier to snapshot this run under --snapshot-root/<run-id>. Use 'auto' to generate a JST timestamp (YYYY-MM-DD_HH-MM-SS).")
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT, help="Base directory for prep run snapshots when --run-id is provided.")
    ap.add_argument("--snapshot-inputs", action="store_true", help="When used with --run-id, copy input/auxiliary CSVs into the snapshot directory.")
    ap.add_argument("--summary-out", type=Path, default=None, help="Optional path to write the prep summary JSON.")
    ap.add_argument("--skip-legacy-output", action="store_true", help="If set, do not write legacy output files (out-train/out-val/out-items) and rely on snapshots only.")
    args = ap.parse_args()

    run_id = args.run_id
    snapshot_root: Path | None = None
    if run_id:
        if run_id.lower() == "auto":
            run_id = _generate_run_id()
        snapshot_root = args.snapshot_root
    if args.skip_legacy_output and run_id is None:
        raise ValueError("--skip-legacy-output requires --run-id (or --run-id auto) to specify snapshot destination.")
    snapshot_inputs = bool(args.snapshot_inputs and run_id)

    summary_path: Path | None = args.summary_out
    if summary_path is None:
        if run_id is not None and snapshot_root is not None:
            summary_path = snapshot_root / run_id / "summary.json"
        elif not args.skip_legacy_output:
            summary_path = DEFAULT_LATEST_DIR / "summary.json"

    cfg = PrepConfig(
        mode=args.mode,
        input_csv=args.input,
        decisions_csv=args.decisions_csv,
        db_url=args.db_url,
        join_on=args.join_on,
        min_positive_stars=args.min_stars,
        negatives_per_positive=args.neg_per_pos,
        max_negative_stars=args.max_negative_stars,
        val_ratio=args.val_ratio,
        out_train=args.out_train,
        out_val=args.out_val,
        out_items=args.out_items,
        out_user_features=args.out_user_features,
        run_id=run_id,
        snapshot_root=snapshot_root,
        snapshot_inputs=snapshot_inputs,
        summary_path=summary_path,
        keep_legacy_outputs=not args.skip_legacy_output,
    )

    if cfg.db_url is None:
        raise ValueError("A Postgres source is required. Provide --db-url.")

    df = pd.read_csv(cfg.input_csv)
    if cfg.mode == "explicit":
        interactions = build_interactions_from_explicit(df)
    else:
        interactions = build_interactions_from_reviews(
            df,
            cfg.min_positive_stars,
            cfg.negatives_per_positive,
            (cfg.max_negative_stars if cfg.max_negative_stars is not None and cfg.max_negative_stars >= 0 else None),
        )

    # Optional extra source: decisions CSV (e.g., user_video_decisions export)
    if cfg.decisions_csv is not None:
        df_dec = pd.read_csv(cfg.decisions_csv)
        extra = build_interactions_from_explicit(df_dec)
        # Prefer explicit decisions on duplicates
        interactions = (
            pd.concat([interactions, extra], ignore_index=True)
              .drop_duplicates(subset=["reviewer_id", "product_url"], keep="last")
        )

    # Optional: join videos to attach video_id and output item attributes parquet
    interactions, items_df, missing_count, joined_ok, join_used = maybe_join_videos(interactions, cfg.join_on, cfg.db_url)
    if not joined_ok or items_df is None:
        raise ValueError("Failed to join videos metadata. Ensure --db-url provides the required tables.")
    dropped_no_video_id = 0
    before = int(len(interactions))
    # Drop rows that failed to map to a video_id
    interactions = interactions.dropna(subset=["video_id"]).reset_index(drop=True)
    after = int(len(interactions))
    dropped_no_video_id = before - after
    # Normalize dtypes before split/write to avoid UUID dtype issues in parquet
    for col in ["reviewer_id", "product_url"]:
        if col in interactions.columns:
            interactions[col] = interactions[col].astype(str)
    if "video_id" in interactions.columns:
        interactions["video_id"] = interactions["video_id"].astype(str)
    if "label" in interactions.columns:
        interactions["label"] = interactions["label"].astype(int)

    train, val = train_val_split(interactions, cfg.val_ratio)

    summary = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "users_train": int(train['reviewer_id'].nunique()),
        "users_val": int(val['reviewer_id'].nunique()),
        "items": int(interactions['product_url'].nunique()),
        "merged_decisions": bool(cfg.decisions_csv is not None),
        "joined_videos": bool(items_df is not None),
        "mode": cfg.mode,
        "missing_video_id_before_drop": int(missing_count),
    }
    if joined_ok:
        cid_null = int(interactions['cid'].isna().sum()) if ('cid' in interactions.columns and join_used == 'external_id') else 0
        summary.update({
            "interactions_before_join": int(len(train) + len(val) + dropped_no_video_id),
            "dropped_no_video_id": int(dropped_no_video_id),
            "items_matched": int(interactions['video_id'].nunique()),
            "join_used": join_used,
            "cid_missing_in_input": cid_null,
        })

    snapshot_dir, snapshot_files = _resolve_snapshot_paths(cfg)
    paths_info: Dict[str, str] = {}
    items_to_write = None
    if items_df is not None:
        items_to_write = items_df.copy()
        if "video_id" in items_to_write.columns:
            items_to_write["video_id"] = items_to_write["video_id"].astype(str)
        if "external_id" in items_to_write.columns:
            items_to_write["external_id"] = items_to_write["external_id"].astype(str)
        if "source" in items_to_write.columns:
            items_to_write["source"] = items_to_write["source"].fillna("").astype(str)
        if "price" in items_to_write.columns:
            items_to_write["price"] = pd.to_numeric(items_to_write["price"], errors="coerce").astype(float)
        if "tag_ids" in items_to_write.columns:
            items_to_write["tag_ids"] = items_to_write["tag_ids"].apply(lambda v: _normalize_to_str_list(v, sort=True))
        if "performer_ids" in items_to_write.columns:
            items_to_write["performer_ids"] = items_to_write["performer_ids"].apply(lambda v: _normalize_to_str_list(v, sort=True))

    user_features_df: pd.DataFrame | None = None
    if cfg.mode != "reviews" and cfg.db_url:
        user_features_df = _load_user_features(cfg.db_url)
    if user_features_df is None:
        user_features_df = _build_user_features_from_interactions(interactions, items_to_write)
        if user_features_df is not None:
            sys.stderr.write("Info: generated synthetic user_features from interactions (reviews-mode fallback).\n")
    if user_features_df is None:
        sys.stderr.write("Note: user_features could not be generated (insufficient interactions).\n")

    summary_paths: list[Path] = []
    if cfg.summary_path is not None:
        summary_paths.append(cfg.summary_path)
        paths_info["summary_path"] = str(cfg.summary_path)

    if cfg.keep_legacy_outputs:
        cfg.out_train.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_val.parent.mkdir(parents=True, exist_ok=True)
        train.to_parquet(cfg.out_train, index=False)
        val.to_parquet(cfg.out_val, index=False)
        paths_info["train_path"] = str(cfg.out_train)
        paths_info["val_path"] = str(cfg.out_val)
        if items_to_write is not None:
            cfg.out_items.parent.mkdir(parents=True, exist_ok=True)
            items_to_write.to_parquet(cfg.out_items, index=False)
            paths_info["items_path"] = str(cfg.out_items)
        else:
            paths_info["items_path"] = "not_generated"
        if user_features_df is not None:
            cfg.out_user_features.parent.mkdir(parents=True, exist_ok=True)
            user_features_df.sort_values("reviewer_id").to_parquet(cfg.out_user_features, index=False)
            paths_info["user_features_path"] = str(cfg.out_user_features)
        else:
            paths_info["user_features_path"] = "not_generated"
    elif snapshot_dir is None:
        raise ValueError("No legacy outputs and no snapshot destination. Provide --run-id or remove --skip-legacy-output.")
    
    if snapshot_dir is not None:
        if cfg.keep_legacy_outputs:
            _copy_if_exists(cfg.out_train, snapshot_files["train"])
            _copy_if_exists(cfg.out_val, snapshot_files["val"])
            if items_to_write is not None and cfg.out_items.exists():
                _copy_if_exists(cfg.out_items, snapshot_files["items"])
            if user_features_df is not None and cfg.out_user_features.exists():
                _copy_if_exists(cfg.out_user_features, snapshot_files["user_features"])
        else:
            train.to_parquet(snapshot_files["train"], index=False)
            val.to_parquet(snapshot_files["val"], index=False)
            if items_to_write is not None:
                items_to_write.to_parquet(snapshot_files["items"], index=False)
            if user_features_df is not None:
                user_features_df.sort_values("reviewer_id").to_parquet(snapshot_files["user_features"], index=False)
        paths_info["snapshot_dir"] = str(snapshot_dir)
        paths_info["snapshot_train"] = str(snapshot_files["train"])
        paths_info["snapshot_val"] = str(snapshot_files["val"])
        if items_to_write is not None:
            paths_info["snapshot_items"] = str(snapshot_files["items"])
        if user_features_df is not None:
            paths_info["snapshot_user_features"] = str(snapshot_files["user_features"])
        if cfg.snapshot_inputs:
            _snapshot_inputs(cfg, snapshot_dir, paths_info)
        summary["run_id"] = cfg.run_id
        summary["snapshot_dir"] = str(snapshot_dir)
        snapshot_summary_path = snapshot_files["summary"]
        summary_paths.append(snapshot_summary_path)
        paths_info["snapshot_summary"] = str(snapshot_summary_path)

    maker_distribution_df = _build_maker_distribution(interactions, items_df)

    if maker_distribution_df is not None and not maker_distribution_df.empty:
        _write_maker_distribution(maker_distribution_df, DEFAULT_MAKER_DISTRIBUTION_LATEST)
        paths_info["maker_distribution_path"] = str(DEFAULT_MAKER_DISTRIBUTION_LATEST)
        if snapshot_dir is not None:
            snapshot_maker_path = snapshot_files["maker_distribution"]
            _write_maker_distribution(maker_distribution_df, snapshot_maker_path)
            paths_info["snapshot_maker_distribution"] = str(snapshot_maker_path)

    unique_summary_paths: list[Path] = []
    seen = set()
    for p in summary_paths:
        if p is None:
            continue
        p = Path(p)
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_summary_paths.append(p)

    if user_features_df is not None and "user_features_path" not in paths_info:
        if snapshot_dir is not None:
            paths_info["user_features_path"] = str(snapshot_files["user_features"])
        else:
            paths_info["user_features_path"] = "generated"
    elif user_features_df is None and "user_features_path" not in paths_info:
        paths_info["user_features_path"] = "not_generated"

    summary.update(
        {
            "user_features_rows": int(len(user_features_df)) if user_features_df is not None else 0,
            "user_features_generated": bool(user_features_df is not None),
        }
    )

    if maker_distribution_df is not None and not maker_distribution_df.empty:
        summary["maker_distribution_top"] = [
            {
                "maker": row["maker"],
                "recommendations": int(row["recommendations"]),
                "unique_items": int(row["unique_items"]),
                "user_coverage": int(row["user_coverage"]),
                "share": float(row["share"]),
            }
            for _, row in maker_distribution_df.head(10).iterrows()
        ]
    else:
        summary["maker_distribution_top"] = []

    summary_payload = {**summary, "paths": paths_info, "generated_at": datetime.now(timezone.utc).isoformat()}
    for path in unique_summary_paths:
        _write_summary(summary_payload, path)

    print(json.dumps({**summary, "paths": paths_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
