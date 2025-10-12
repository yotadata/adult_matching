from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg
from sklearn.preprocessing import StandardScaler
from psycopg.rows import dict_row


@dataclass
class Vocabulary:
    tokens: List[str]
    index: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {token: idx for idx, token in enumerate(self.tokens)}

    def encode(self, values: Iterable[str]) -> np.ndarray:
        vec = np.zeros(len(self.tokens), dtype=np.float32)
        for value in values:
            idx = self.index.get(value)
            if idx is not None:
                vec[idx] += 1.0
        if vec.sum() > 0:
            vec /= np.linalg.norm(vec) + 1e-8
        return vec


@dataclass
class NumericNormalizer:
    feature_names: List[str]
    scaler: StandardScaler = field(init=False)

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, matrix: np.ndarray) -> None:
        self.scaler.fit(matrix)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return self.scaler.transform(matrix)

    def stats(self) -> Dict[str, Dict[str, float]]:
        means = self.scaler.mean_.tolist()
        scales = self.scaler.scale_.tolist()
        stats = {}
        for name, mean, std in zip(self.feature_names, means, scales):
            stats[name] = {
                "mean": float(mean),
                "std": float(std if std != 0 else 1.0),
            }
        return stats


def serialize_normalizer(normalizer: NumericNormalizer) -> Dict[str, object]:
    return {
        "feature_names": normalizer.feature_names,
        "mean": normalizer.scaler.mean_.tolist(),
        "scale": normalizer.scaler.scale_.tolist(),
    }


def deserialize_normalizer(payload: Dict[str, object]) -> NumericNormalizer:
    feature_names = list(payload["feature_names"])
    normalizer = NumericNormalizer(feature_names)
    normalizer.scaler.mean_ = np.array(payload["mean"], dtype=np.float64)
    normalizer.scaler.scale_ = np.array(payload["scale"], dtype=np.float64)
    normalizer.scaler.var_ = normalizer.scaler.scale_ ** 2
    normalizer.scaler.n_features_in_ = len(feature_names)
    normalizer.scaler.n_samples_seen_ = 1
    return normalizer


def build_vocab_from_videos(series: pd.Series, max_size: int, min_freq: int) -> Vocabulary:
    freq: Dict[str, int] = {}
    for values in series.dropna():
        for value in values:
            freq[value] = freq.get(value, 0) + 1
    sorted_tokens = [token for token, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if count >= min_freq]
    if max_size > 0:
        sorted_tokens = sorted_tokens[:max_size]
    return Vocabulary(tokens=sorted_tokens)


def compute_reference_datetime(decisions: pd.DataFrame) -> datetime:
    ts = decisions["decision_ts"].dropna()
    if ts.empty:
        return datetime.now(timezone.utc)
    return ts.max().to_pydatetime(warn=False)


def _safe_days_between(later: datetime, earlier: datetime | None) -> float:
    if earlier is None:
        return 0.0
    delta = later - earlier
    return delta.total_seconds() / 86400.0


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
        return ts.to_pydatetime()
    return None


class UserFeatureStore:
    def __init__(
        self,
        profiles: pd.DataFrame,
        videos: pd.DataFrame,
        decisions: pd.DataFrame,
        tag_vocab: Vocabulary,
        actress_vocab: Vocabulary,
        *,
        use_rpc: bool = False,
        pg_dsn: str | None = None,
    ) -> None:
        self.profiles = profiles
        self.videos = videos
        self.decisions = decisions
        self.tag_vocab = tag_vocab
        self.actress_vocab = actress_vocab
        self.use_rpc = use_rpc
        self.pg_dsn = pg_dsn
        self._rpc_conn: psycopg.Connection | None = None
        self._rpc_payload_cache: Dict[str, Dict[str, Any]] = {}
        self.video_lookup = videos.set_index("id").to_dict(orient="index")
        self.profile_lookup = profiles.set_index("user_id").to_dict(orient="index")
        self.reference_dt = datetime.now(timezone.utc) if use_rpc else compute_reference_datetime(decisions)
        self._cache: Dict[str, Dict[str, np.ndarray | float]] = {}

        if self.use_rpc and not self.pg_dsn:
            raise ValueError("pg_dsn must be provided when use_rpc is True")

    def _get_rpc_conn(self) -> psycopg.Connection:
        assert self.pg_dsn is not None
        if self._rpc_conn is None:
            self._rpc_conn = psycopg.connect(self.pg_dsn, row_factory=dict_row)
        return self._rpc_conn

    def _fetch_user_payload(self, user_id: str) -> Dict[str, Any]:
        if user_id in self._rpc_payload_cache:
            return self._rpc_payload_cache[user_id]
        conn = self._get_rpc_conn()
        row = conn.execute(
            "select public.get_user_embedding_features(%s) as payload",
            (user_id,),
        ).fetchone()
        payload = (row or {}).get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        self._rpc_payload_cache[user_id] = payload
        return payload

    def _build_features_from_rpc(self, user_id: str) -> Dict[str, np.ndarray | float]:
        payload = self._fetch_user_payload(user_id)
        decision_stats = payload.get("decision_stats") or {}
        price_stats = payload.get("price_stats") or {}

        like_count = float(decision_stats.get("like_count") or 0.0)
        nope_count = float(decision_stats.get("nope_count") or 0.0)
        decision_count = float(decision_stats.get("decision_count") or (like_count + nope_count))
        like_ratio = like_count / decision_count if decision_count > 0 else 0.5
        like_ratio = float(np.clip(like_ratio, 0.0, 1.0))

        mean_price = float(price_stats.get("mean") or 0.0)
        median_price = float(price_stats.get("median") or 0.0)

        recent_like_at = _parse_datetime(decision_stats.get("recent_like_at"))
        avg_hour = float(decision_stats.get("average_like_hour") or 12.0)
        avg_hour = float(np.clip(avg_hour, 0.0, 23.0))

        profile = payload.get("profile") or {}
        profile_created = _parse_datetime(profile.get("created_at"))

        tag_values = [str(v) for v in payload.get("tags") or [] if isinstance(v, str)]
        actress_values = [str(v) for v in payload.get("performers") or [] if isinstance(v, str)]

        numeric_features = np.array(
            [
                _safe_days_between(self.reference_dt, profile_created),
                mean_price,
                median_price,
                like_ratio,
                like_count,
                nope_count,
                decision_count,
                _safe_days_between(self.reference_dt, recent_like_at),
            ],
            dtype=np.float32,
        )

        return {
            "numeric": numeric_features,
            "hour_of_day": avg_hour,
            "tag_vector": self.tag_vocab.encode(tag_values),
            "actress_vector": self.actress_vocab.encode(actress_values),
        }

    def build_features(self, user_id: str) -> Dict[str, np.ndarray | float]:
        if user_id in self._cache:
            return self._cache[user_id]

        if self.use_rpc:
            features = self._build_features_from_rpc(user_id)
            self._cache[user_id] = features
            return features

        profile = self.profile_lookup.get(user_id, {})
        user_decisions = self.decisions[self.decisions["user_id"] == user_id]
        likes = user_decisions[user_decisions["label"] == 1]
        nopes = user_decisions[user_decisions["label"] == 0]

        like_count = int(likes.shape[0])
        nope_count = int(nopes.shape[0])
        decision_count = int(user_decisions.shape[0])
        like_ratio = float(like_count / decision_count) if decision_count > 0 else 0.5

        liked_video_ids = likes["video_id"].tolist()
        liked_videos = [self.video_lookup.get(video_id) for video_id in liked_video_ids]

        tag_values: List[str] = []
        actress_values: List[str] = []
        prices: List[float] = []
        like_timestamps: List[datetime] = []

        for row, ts in zip(liked_videos, likes["decision_ts"].tolist()):
            if not row:
                continue
            tag_values.extend(row.get("tags", []))
            actress_values.extend(row.get("performers", []))
            price = row.get("price")
            if price is not None:
                prices.append(float(price))
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                like_timestamps.append(ts.to_pydatetime(warn=False))

        tag_vector = self.tag_vocab.encode(tag_values)
        actress_vector = self.actress_vocab.encode(actress_values)

        mean_price = float(np.mean(prices)) if prices else 0.0
        median_price = float(np.median(prices)) if prices else 0.0

        profile_created = profile.get("created_at")
        account_age_days = _safe_days_between(self.reference_dt, profile_created) if isinstance(profile_created, datetime) else 0.0

        recent_like_days = _safe_days_between(self.reference_dt, max(like_timestamps) if like_timestamps else None)

        hours = [ts.hour for ts in like_timestamps if isinstance(ts, datetime)]
        hour_of_day = float(np.mean(hours)) if hours else 12.0

        features = {
            "numeric": np.array([
                account_age_days,
                mean_price,
                median_price,
                like_ratio,
                like_count,
                nope_count,
                decision_count,
                recent_like_days,
            ], dtype=np.float32),
            "hour_of_day": hour_of_day,
            "tag_vector": tag_vector,
            "actress_vector": actress_vector,
        }

        self._cache[user_id] = features
        return features


class ItemFeatureStore:
    def __init__(
        self,
        videos: pd.DataFrame,
        tag_vocab: Vocabulary,
        actress_vocab: Vocabulary,
        *,
        use_rpc: bool = False,
        pg_dsn: str | None = None,
    ) -> None:
        self.videos = videos
        self.tag_vocab = tag_vocab
        self.actress_vocab = actress_vocab
        self.reference_release = videos["product_released_at"].max()
        self.use_rpc = use_rpc
        self.pg_dsn = pg_dsn
        self._rpc_conn: psycopg.Connection | None = None
        self._rpc_payload_cache: Dict[str, Dict[str, Any]] = {}

        if self.use_rpc and not self.pg_dsn:
            raise ValueError("pg_dsn must be provided when use_rpc is True")

    def _get_rpc_conn(self) -> psycopg.Connection:
        assert self.pg_dsn is not None
        if self._rpc_conn is None:
            self._rpc_conn = psycopg.connect(self.pg_dsn, row_factory=dict_row)
        return self._rpc_conn

    def _fetch_item_payload(self, video_id: str) -> Dict[str, Any]:
        if video_id in self._rpc_payload_cache:
            return self._rpc_payload_cache[video_id]
        conn = self._get_rpc_conn()
        row = conn.execute(
            "select public.get_item_embedding_features(%s) as payload",
            (video_id,),
        ).fetchone()
        payload = (row or {}).get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        self._rpc_payload_cache[video_id] = payload
        return payload

    def build_features(self, video_id: str) -> Dict[str, np.ndarray | float]:
        if self.use_rpc:
            payload = self._fetch_item_payload(video_id)
            tags = [str(v) for v in payload.get("tags") or [] if isinstance(v, str)]
            actresses = [str(v) for v in payload.get("performers") or [] if isinstance(v, str)]
            price = float(payload.get("price") or 0.0)
            duration = float(payload.get("duration_seconds") or 0.0)
            release_at = _parse_datetime(payload.get("product_released_at"))
            if isinstance(self.reference_release, pd.Timestamp) and not pd.isna(self.reference_release):
                ref_dt = self.reference_release.to_pydatetime()
                if ref_dt.tzinfo is None:
                    ref_dt = ref_dt.replace(tzinfo=timezone.utc)
            else:
                ref_dt = datetime.now(timezone.utc)
            recency_days = _safe_days_between(ref_dt, release_at)
            return {
                "numeric": np.array([price, recency_days, duration], dtype=np.float32),
                "tag_vector": self.tag_vocab.encode(tags),
                "actress_vector": self.actress_vocab.encode(actresses),
            }

        row = self.videos[self.videos["id"] == video_id]
        if row.empty:
            return {
                "numeric": np.zeros(3, dtype=np.float32),
                "tag_vector": np.zeros(len(self.tag_vocab.tokens), dtype=np.float32),
                "actress_vector": np.zeros(len(self.actress_vocab.tokens), dtype=np.float32),
            }
        record = row.iloc[0]
        tags = record.get("tags", [])
        actresses = record.get("performers", [])
        price = float(record.get("price") or 0.0)
        release_at = record.get("product_released_at")
        recency_days = 0.0
        if isinstance(release_at, pd.Timestamp) and not pd.isna(release_at):
            reference = self.reference_release if isinstance(self.reference_release, pd.Timestamp) else release_at
            recency_days = (reference - release_at).days
        duration = float(record.get("duration_seconds") or 0.0)
        return {
            "numeric": np.array([price, recency_days, duration], dtype=np.float32),
            "tag_vector": self.tag_vocab.encode(tags),
            "actress_vector": self.actress_vocab.encode(actresses),
        }


def assemble_user_feature_vector(
    user_features: Dict[str, np.ndarray | float],
    normalizer: NumericNormalizer,
) -> np.ndarray:
    numeric = user_features["numeric"].reshape(1, -1)
    normalized_numeric = normalizer.transform(numeric)[0].astype(np.float32)
    hour_norm = float(user_features.get("hour_of_day", 12.0)) / 23.0
    components = [normalized_numeric, np.array([hour_norm], dtype=np.float32)]
    components.append(user_features.get("tag_vector", np.zeros(0, dtype=np.float32)).astype(np.float32))
    components.append(user_features.get("actress_vector", np.zeros(0, dtype=np.float32)).astype(np.float32))
    return np.concatenate(components, axis=0)


def assemble_item_feature_vector(
    item_features: Dict[str, np.ndarray | float],
    normalizer: NumericNormalizer,
) -> np.ndarray:
    numeric = item_features["numeric"].reshape(1, -1)
    normalized_numeric = normalizer.transform(numeric)[0].astype(np.float32)
    components = [normalized_numeric]
    components.append(item_features.get("tag_vector", np.zeros(0, dtype=np.float32)).astype(np.float32))
    components.append(item_features.get("actress_vector", np.zeros(0, dtype=np.float32)).astype(np.float32))
    return np.concatenate(components, axis=0)


@dataclass
class FeaturePipeline:
    tag_vocab: Vocabulary
    actress_vocab: Vocabulary
    user_numeric_normalizer: NumericNormalizer
    item_numeric_normalizer: NumericNormalizer
    user_feature_dim: int
    item_feature_dim: int

    def export_metadata(self) -> Dict[str, object]:
        return {
            "tag_vocab": self.tag_vocab.tokens,
            "actress_vocab": self.actress_vocab.tokens,
            "user_numeric_normalizer": serialize_normalizer(self.user_numeric_normalizer),
            "item_numeric_normalizer": serialize_normalizer(self.item_numeric_normalizer),
            "user_feature_dim": self.user_feature_dim,
            "item_feature_dim": self.item_feature_dim,
        }


def restore_pipeline(metadata: Dict[str, object]) -> FeaturePipeline:
    tag_vocab = Vocabulary(list(metadata["tag_vocab"]))
    actress_vocab = Vocabulary(list(metadata["actress_vocab"]))
    user_normalizer = deserialize_normalizer(metadata["user_numeric_normalizer"])
    item_normalizer = deserialize_normalizer(metadata["item_numeric_normalizer"])
    return FeaturePipeline(
        tag_vocab=tag_vocab,
        actress_vocab=actress_vocab,
        user_numeric_normalizer=user_normalizer,
        item_numeric_normalizer=item_normalizer,
        user_feature_dim=int(metadata["user_feature_dim"]),
        item_feature_dim=int(metadata["item_feature_dim"]),
    )


def build_feature_pipeline(
    profiles: pd.DataFrame,
    videos: pd.DataFrame,
    decisions: pd.DataFrame,
    max_tag_vocab: int,
    min_tag_freq: int,
    max_actress_vocab: int,
    min_actress_freq: int,
    *,
    use_rpc_features: bool = False,
    pg_dsn: str | None = None,
) -> Tuple[FeaturePipeline, UserFeatureStore, ItemFeatureStore]:
    tag_vocab = build_vocab_from_videos(videos["tags"], max_tag_vocab, min_tag_freq)
    actress_vocab = build_vocab_from_videos(videos["performers"], max_actress_vocab, min_actress_freq)

    user_store = UserFeatureStore(
        profiles,
        videos,
        decisions,
        tag_vocab,
        actress_vocab,
        use_rpc=use_rpc_features,
        pg_dsn=pg_dsn,
    )
    item_store = ItemFeatureStore(
        videos,
        tag_vocab,
        actress_vocab,
        use_rpc=use_rpc_features,
        pg_dsn=pg_dsn,
    )

    user_numeric_samples: List[np.ndarray] = []
    for user_id in decisions["user_id"].unique():
        feats = user_store.build_features(user_id)
        user_numeric_samples.append(feats["numeric"])
    user_numeric_matrix = np.stack(user_numeric_samples) if user_numeric_samples else np.zeros((1, 8), dtype=np.float32)
    user_normalizer = NumericNormalizer([
        "account_age_days",
        "mean_price",
        "median_price",
        "like_ratio",
        "like_count",
        "nope_count",
        "decision_count",
        "recent_like_days",
    ])
    user_normalizer.fit(user_numeric_matrix)

    item_numeric_samples: List[np.ndarray] = []
    for video_id in decisions["video_id"].unique():
        feats = item_store.build_features(video_id)
        item_numeric_samples.append(feats["numeric"])
    item_numeric_matrix = np.stack(item_numeric_samples) if item_numeric_samples else np.zeros((1, 3), dtype=np.float32)
    item_normalizer = NumericNormalizer([
        "price",
        "recency_days",
        "duration_seconds",
    ])
    item_normalizer.fit(item_numeric_matrix)

    if decisions.empty:
        dummy_user = assemble_user_feature_vector({"numeric": np.zeros(8, dtype=np.float32)}, user_normalizer)
        dummy_item = assemble_item_feature_vector({"numeric": np.zeros(3, dtype=np.float32)}, item_normalizer)
    else:
        first_user_id = decisions.iloc[0]["user_id"]
        first_video_id = decisions.iloc[0]["video_id"]
        dummy_user = assemble_user_feature_vector(user_store.build_features(first_user_id), user_normalizer)
        dummy_item = assemble_item_feature_vector(item_store.build_features(first_video_id), item_normalizer)

    pipeline = FeaturePipeline(
        tag_vocab=tag_vocab,
        actress_vocab=actress_vocab,
        user_numeric_normalizer=user_normalizer,
        item_numeric_normalizer=item_normalizer,
        user_feature_dim=int(dummy_user.shape[0]),
        item_feature_dim=int(dummy_item.shape[0]),
    )
    return pipeline, user_store, item_store
