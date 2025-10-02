from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass
class DatasetPaths:
    profiles: Path
    videos: Path
    decisions: Path


def load_profiles(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    df = pd.DataFrame(data)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def load_videos(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    df = pd.DataFrame(data)
    for col in ("product_released_at", "published_at", "created_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    if "performers" in df.columns:
        df["performers"] = df["performers"].apply(lambda x: x if isinstance(x, list) else [])
    else:
        df["performers"] = [[] for _ in range(len(df))]
    return df


def load_decisions(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    df = pd.DataFrame(data)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def _sample_negatives(
    user_id: str,
    liked_video_ids: Iterable[str],
    candidate_video_ids: List[str],
    num_samples: int,
    rng: random.Random,
) -> List[str]:
    liked_set = set(liked_video_ids)
    eligible = [vid for vid in candidate_video_ids if vid not in liked_set]
    if not eligible:
        return []
    # sample with replacement if needed
    choices = []
    for _ in range(num_samples):
        choices.append(rng.choice(eligible))
    return choices


def build_training_samples(
    profiles: pd.DataFrame,
    videos: pd.DataFrame,
    decisions: pd.DataFrame,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)

    # limit to users present in profiles and decisions
    valid_user_ids = set(profiles["user_id"].astype(str))
    decisions = decisions[decisions["user_id"].astype(str).isin(valid_user_ids)].copy()

    # map video id -> metadata
    videos = videos.copy()
    videos["id"] = videos["id"].astype(str)
    video_lookup: Dict[str, Dict] = videos.set_index("id").to_dict(orient="index")
    all_video_ids = list(video_lookup.keys())

    samples: List[Dict] = []

    grouped = decisions.groupby("user_id")
    for user_id, user_df in grouped:
        user_likes = user_df[user_df["decision_type"] == "like"]
        user_nopes = user_df[user_df["decision_type"] == "nope"]

        liked_ids = user_likes["video_id"].astype(str).tolist()
        nope_ids = user_nopes["video_id"].astype(str).tolist()

        # positive samples
        for _, row in user_likes.iterrows():
            video_id = str(row["video_id"])
            if video_id not in video_lookup:
                continue
            samples.append(
                {
                    "user_id": str(user_id),
                    "video_id": video_id,
                    "label": 1,
                    "decision_type": "like",
                    "decision_ts": row.get("created_at"),
                }
            )

        # explicit negatives from NOPE
        for _, row in user_nopes.iterrows():
            video_id = str(row["video_id"])
            if video_id not in video_lookup:
                continue
            samples.append(
                {
                    "user_id": str(user_id),
                    "video_id": video_id,
                    "label": 0,
                    "decision_type": "nope",
                    "decision_ts": row.get("created_at"),
                }
            )

        # sampled negatives
        num_extra_neg = max(0, negative_ratio * max(1, len(liked_ids)) - len(nope_ids))
        negative_candidates = _sample_negatives(
            str(user_id),
            liked_ids + nope_ids,
            all_video_ids,
            num_extra_neg,
            rng,
        )
        for video_id in negative_candidates:
            samples.append(
                {
                    "user_id": str(user_id),
                    "video_id": video_id,
                    "label": 0,
                    "decision_type": "sampled_negative",
                    "decision_ts": None,
                }
            )

    df = pd.DataFrame(samples)
    if not df.empty:
        df["decision_ts"] = pd.to_datetime(df["decision_ts"], errors="coerce")
    return df


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or val_ratio <= 0:
        return df, df.iloc[0:0]
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(df) * val_ratio)
    if val_size == 0:
        return df, df.iloc[0:0]
    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()
    return train_df, val_df
