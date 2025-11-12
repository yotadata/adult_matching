#!/usr/bin/env python3
"""Interactive qualitative evaluation for Two-Tower recommender models."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

st.set_page_config(page_title="Two-Tower Qualitative Review", layout="wide")

DEFAULT_MODEL_DIR = Path("ml/artifacts/latest")
DEFAULT_ITEM_FEATURES = Path("ml/data/processed/two_tower/latest/item_features.parquet")
DEFAULT_TRAIN = Path("ml/data/processed/two_tower/latest/interactions_train.parquet")
DEFAULT_VAL = Path("ml/data/processed/two_tower/latest/interactions_val.parquet")
ATTRIBUTE_CHOICES = ["maker", "source", "label", "series", "tags", "performer_ids"]


@dataclass
class ModelData:
    base_path: Path
    metrics: Dict[str, object]
    user_df: pd.DataFrame
    user_matrix: np.ndarray
    user_index: Dict[str, int]
    item_df: pd.DataFrame
    item_matrix: np.ndarray
    item_ids: List[str]
    item_index: Dict[str, int]


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_metrics_json(path_str: str) -> Dict[str, object]:
    path = Path(path_str)
    if not path.exists():
        return {}
    import json

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=True)
def load_model_data(base_path_str: str) -> Optional[ModelData]:
    base_path = Path(base_path_str)
    user_embeddings = base_path / "user_embeddings.parquet"
    item_embeddings = base_path / "video_embeddings.parquet"
    metrics_path = base_path / "metrics.json"

    if not user_embeddings.exists() or not item_embeddings.exists():
        return None

    user_df = pd.read_parquet(user_embeddings)
    user_df["reviewer_id"] = user_df["reviewer_id"].astype(str)
    user_matrix = np.vstack(user_df["embedding"].tolist())
    user_index = {rid: idx for idx, rid in enumerate(user_df["reviewer_id"].tolist())}

    item_df = pd.read_parquet(item_embeddings)
    item_df["video_id"] = item_df["video_id"].astype(str)
    item_matrix = np.vstack(item_df["embedding"].tolist())
    item_ids = item_df["video_id"].tolist()
    item_index = {vid: idx for idx, vid in enumerate(item_ids)}

    metrics = load_metrics_json(str(metrics_path))

    return ModelData(
        base_path=base_path,
        metrics=metrics,
        user_df=user_df,
        user_matrix=user_matrix,
        user_index=user_index,
        item_df=item_df,
        item_matrix=item_matrix,
        item_ids=item_ids,
        item_index=item_index,
    )


@st.cache_data(show_spinner=False)
def load_model_meta(base_path_str: str) -> Dict[str, object]:
    path = Path(base_path_str) / "model_meta.json"
    if not path.exists():
        return {}
    import json

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def format_trained_at(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return value


@st.cache_data(show_spinner=False)
def load_item_features(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "video_id" in df.columns:
        df["video_id"] = df["video_id"].astype(str)
        df = df.set_index("video_id")
    return df


@st.cache_data(show_spinner=False)
def load_interactions(train_path_str: str, val_path_str: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name, path_str in [("train", train_path_str), ("val", val_path_str)]:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_parquet(path).copy()
        df["reviewer_id"] = df["reviewer_id"].astype(str)
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].astype(str)
        df["dataset"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["reviewer_id", "video_id", "label", "dataset"])
    return pd.concat(frames, ignore_index=True)


def build_history_map(interactions: pd.DataFrame) -> Dict[str, set[str]]:
    history: Dict[str, set[str]] = {}
    for rid, group in interactions.groupby("reviewer_id"):
        history[rid] = set(group["video_id"].tolist())
    return history


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _normalize_tokens(value: str) -> List[str]:
    normalized = value.replace("／", ",").replace("、", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def get_attribute_values(row: pd.Series, attribute: str) -> List[str]:
    if attribute not in row.index:
        return []
    value = row[attribute]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                tokens = _normalize_tokens(item)
                if tokens:
                    result.extend(tokens)
                elif item.strip():
                    result.append(item.strip())
            else:
                result.append(str(item))
        return [token for token in result if token]
    if isinstance(value, str):
        tokens = _normalize_tokens(value)
        if tokens:
            return tokens
        value = value.strip()
        return [value] if value else []
    return [str(value)] if str(value).strip() else []


def compute_recommendations_for_user(
    model: ModelData,
    user_id: str,
    top_k: int,
    history_map: Dict[str, set[str]],
    exclude_seen: bool,
) -> List[str]:
    if user_id not in model.user_index:
        return []
    idx = model.user_index[user_id]
    user_vec = model.user_matrix[idx]
    scores = model.item_matrix @ user_vec
    if exclude_seen:
        seen_ids = history_map.get(user_id, set())
        if seen_ids:
            seen_indices = [model.item_index[item_id] for item_id in seen_ids if item_id in model.item_index]
            scores[seen_indices] = -np.inf
    if top_k >= len(scores):
        top_indices = np.argsort(-scores)
    else:
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [model.item_ids[i] for i in top_indices]


def build_recommendation_df(
    model: ModelData,
    item_features: pd.DataFrame,
    item_ids: Sequence[str],
    scores: Sequence[float],
    seen_info: Dict[str, Dict[str, object]],
    other_items: Optional[set[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rank, (item_id, score) in enumerate(zip(item_ids, scores), start=1):
        meta = item_features.loc[item_id] if item_id in item_features.index else None
        seen = seen_info.get(item_id)
        rows.append(
            {
                "rank": rank,
                "score": score,
                "video_id": item_id,
                "title": None if meta is None else meta.get("title"),
                "maker": None if meta is None else meta.get("maker"),
                "series": None if meta is None else meta.get("series"),
                "label": None if meta is None else meta.get("label"),
                "source": None if meta is None else meta.get("source"),
                "tags": "" if meta is None else ", ".join(get_attribute_values(meta, "tags")),
                "seen_dataset": None if seen is None else seen.get("dataset"),
                "seen_label": None if seen is None else seen.get("label"),
                "product_url": None if meta is None else meta.get("product_url"),
                "also_in_other_model": other_items is not None and item_id in other_items,
            }
        )
    return pd.DataFrame(rows)


def compute_dataset_attribute_counts(
    item_features: pd.DataFrame,
    interactions: pd.DataFrame,
    attribute: str,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if attribute not in item_features.columns:
        return pd.DataFrame(columns=[attribute, "count"]), pd.DataFrame(columns=[attribute, "positive_count"])

    def flatten(series: pd.Series) -> List[str]:
        values: List[str] = []
        for val in series:
            values.extend(get_attribute_values(pd.Series({attribute: val}), attribute))
        return values

    all_values = flatten(item_features[attribute])
    freq = (
        pd.Series(all_values)
        .value_counts()
        .head(top_n)
        .rename_axis(attribute)
        .reset_index(name="count")
    )

    positives = interactions[interactions.get("label", 0) > 0][["video_id"]].merge(
        item_features.reset_index()[["video_id", attribute]], on="video_id", how="left"
    )
    pos_values = flatten(positives[attribute])
    pos_freq = (
        pd.Series(pos_values)
        .value_counts()
        .head(top_n)
        .rename_axis(attribute)
        .reset_index(name="positive_count")
    )
    return freq, pos_freq


def compute_recommendation_distribution(
    model: ModelData,
    item_features: pd.DataFrame,
    history_map: Dict[str, set[str]],
    attribute: str,
    top_k: int,
    exclude_seen: bool,
    top_n: int,
) -> pd.DataFrame:
    if attribute not in item_features.columns:
        return pd.DataFrame(columns=[attribute, "recommendations", "unique_items", "user_coverage"])

    counter: Dict[str, int] = {}
    unique_items: Dict[str, set[str]] = {}
    user_coverage: Dict[str, set[str]] = {}

    for user_id in model.user_df["reviewer_id"]:
        top_items = compute_recommendations_for_user(model, user_id, top_k, history_map, exclude_seen)
        for item_id in top_items:
            if item_id not in item_features.index:
                continue
            row = item_features.loc[item_id]
            values = get_attribute_values(row, attribute)
            for value in values:
                counter[value] = counter.get(value, 0) + 1
                unique_items.setdefault(value, set()).add(item_id)
                user_coverage.setdefault(value, set()).add(user_id)

    if not counter:
        return pd.DataFrame(columns=[attribute, "recommendations", "unique_items", "user_coverage"])

    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    records = []
    for value, count in items:
        records.append(
            {
                attribute: value,
                "recommendations": count,
                "unique_items": len(unique_items.get(value, set())),
                "user_coverage": len(user_coverage.get(value, set())),
            }
        )
    return pd.DataFrame(records)


def compute_attribute_label_breakdown(
    item_features: pd.DataFrame,
    interactions: pd.DataFrame,
    attribute: str,
    top_n: int,
) -> pd.DataFrame:
    if attribute not in item_features.columns:
        return pd.DataFrame(columns=[attribute, "positive_count", "negative_count", "positive_ratio", "negative_ratio", "total_count"])

    base = item_features.reset_index()[["video_id", attribute]]
    merged = interactions[["video_id", "label"]].merge(base, on="video_id", how="left")
    counter: Dict[str, Dict[str, int]] = {}

    for row in merged.itertuples(index=False):
        value_series = pd.Series({attribute: getattr(row, attribute, None)})
        values = get_attribute_values(value_series, attribute)
        if not values:
            continue
        label = float(getattr(row, "label", 0))
        for val in values:
            bucket = counter.setdefault(val, {"positive": 0, "negative": 0, "total": 0})
            bucket["total"] += 1
            if label > 0:
                bucket["positive"] += 1
            else:
                bucket["negative"] += 1

    if not counter:
        return pd.DataFrame(columns=[attribute, "positive_count", "negative_count", "positive_ratio", "negative_ratio", "total_count"])

    rows = []
    for val, counts in counter.items():
        total = counts["total"]
        pos = counts["positive"]
        neg = counts["negative"]
        rows.append(
            {
                attribute: val,
                "positive_count": pos,
                "negative_count": neg,
                "total_count": total,
                "positive_ratio": pos / total if total else 0.0,
                "negative_ratio": neg / total if total else 0.0,
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values("total_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df


def render_metrics_card(title: str, metrics: Dict[str, object]) -> None:
    col1, col2, col3 = st.columns(3)
    roc = metrics.get("metrics", {}).get("roc_auc") if metrics else None
    recall = metrics.get("metrics", {}).get("recall_at_k") if metrics else None
    val_loss = metrics.get("metrics", {}).get("bce_with_logits") if metrics else None
    col1.metric(f"{title} ROC-AUC", f"{roc:.3f}" if roc is not None else "-")
    col2.metric(f"{title} Recall@K", f"{recall:.3f}" if recall is not None else "-")
    col3.metric(f"{title} BCE", f"{val_loss:.3f}" if val_loss is not None else "-")


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------


def main() -> None:
    st.sidebar.header("Artifacts & Paths")
    item_features_path = Path(st.sidebar.text_input("Item features parquet", str(DEFAULT_ITEM_FEATURES)))
    train_path = Path(st.sidebar.text_input("Train interactions parquet", str(DEFAULT_TRAIN)))
    val_path = Path(st.sidebar.text_input("Validation interactions parquet", str(DEFAULT_VAL)))

    model_a_dir = Path(st.sidebar.text_input("Model A artifacts directory", str(DEFAULT_MODEL_DIR)))
    compare_models = st.sidebar.checkbox("Compare with Model B", value=False)
    model_b_dir: Optional[Path] = None
    if compare_models:
        model_b_dir = Path(st.sidebar.text_input("Model B artifacts directory", str(DEFAULT_MODEL_DIR)))

    model_a_meta = load_model_meta(str(model_a_dir))
    model_b_meta = load_model_meta(str(model_b_dir)) if compare_models and model_b_dir else {}

    top_k = st.sidebar.slider("Top K", min_value=5, max_value=100, value=20, step=5)
    exclude_seen = st.sidebar.checkbox("Exclude known items", value=True)

    st.sidebar.markdown("**Model timestamps**")
    if model_a_meta:
        st.sidebar.caption(f"Model A trained at: {format_trained_at(model_a_meta.get('trained_at'))}")
    else:
        st.sidebar.caption("Model A trained at: unknown")
    if compare_models:
        if model_b_meta:
            st.sidebar.caption(f"Model B trained at: {format_trained_at(model_b_meta.get('trained_at'))}")
        else:
            st.sidebar.caption("Model B trained at: unknown")

    item_features_df = load_item_features(str(item_features_path))
    interactions_df = load_interactions(str(train_path), str(val_path))
    history_map = build_history_map(interactions_df)

    model_a = load_model_data(str(model_a_dir))
    model_b = load_model_data(str(model_b_dir)) if compare_models and model_b_dir else None

    if model_a is None:
        st.error("Model A artifactsが見つかりません。user_embeddings.parquet と video_embeddings.parquet を含むディレクトリを指定してください。")
        st.stop()
    if compare_models and model_b is None:
        st.warning("Model B artifacts が読み込めませんでした。比較を無効化するか、正しいパスを指定してください。")
        model_b = None

    tab_user, tab_bias, tab_distribution = st.tabs(["User-level recommendations", "Dataset bias", "Recommendation distribution"])

    with tab_user:
        st.subheader("User-level recommendations")
        render_metrics_card("Model A", model_a.metrics)
        if model_b is not None:
            render_metrics_card("Model B", model_b.metrics)
        st.caption(
            f"Model A users/items: {len(model_a.user_df):,} / {len(model_a.item_df):,}"
            + ("" if model_b is None else f" | Model B users/items: {len(model_b.user_df):,} / {len(model_b.item_df):,}")
        )

        user_ids = model_a.user_df["reviewer_id"].tolist()
        if model_b is not None:
            user_ids = sorted(set(user_ids) & set(model_b.user_df["reviewer_id"].tolist()))
        if not user_ids:
            st.warning("共通のユーザーが存在しません。")
            st.stop()
        selected_user = st.selectbox("Reviewer ID", user_ids)

        user_history = interactions_df[interactions_df["reviewer_id"] == selected_user].copy()
        history_map_selected = {
            vid: {"label": float(getattr(row, "label", 0.0)), "dataset": getattr(row, "dataset", "")}
            for row in user_history.itertuples(index=False)
            for vid in [str(row.video_id)]
        }

        top_items_a = compute_recommendations_for_user(model_a, selected_user, top_k, history_map, exclude_seen)
        scores_a = []
        if top_items_a:
            idx_a = model_a.user_index[selected_user]
            scores_full_a = model_a.item_matrix @ model_a.user_matrix[idx_a]
            scores_a = [float(scores_full_a[model_a.item_index[item_id]]) for item_id in top_items_a]

        top_items_b: List[str] = []
        scores_b: List[float] = []
        if model_b is not None:
            top_items_b = compute_recommendations_for_user(model_b, selected_user, top_k, history_map, exclude_seen)
            if top_items_b:
                idx_b = model_b.user_index[selected_user]
                scores_full_b = model_b.item_matrix @ model_b.user_matrix[idx_b]
                scores_b = [float(scores_full_b[model_b.item_index[item_id]]) for item_id in top_items_b]

        col_a, col_b = st.columns(2) if model_b is not None else (st.container(), None)

        with col_a:
            st.markdown("#### Model A recommendations")
            df_a = build_recommendation_df(
                model_a,
                item_features_df,
                top_items_a,
                scores_a,
                history_map_selected,
                other_items=set(top_items_b) if top_items_b else None,
            )
            st.dataframe(df_a, use_container_width=True)

        if model_b is not None and col_b is not None:
            with col_b:
                st.markdown("#### Model B recommendations")
                df_b = build_recommendation_df(
                    model_b,
                    item_features_df,
                    top_items_b,
                    scores_b,
                    history_map_selected,
                    other_items=set(top_items_a) if top_items_a else None,
                )
                st.dataframe(df_b, use_container_width=True)

        with st.expander("既知アイテム"):
            if user_history.empty:
                st.write("既知のインタラクションがありません。")
            else:
                hist_rows = []
                for row in user_history.itertuples(index=False):
                    item_id = str(row.video_id)
                    meta = item_features_df.loc[item_id] if item_id in item_features_df.index else None
                    hist_rows.append(
                        {
                            "video_id": item_id,
                            "label": float(getattr(row, "label", 0.0)),
                            "dataset": getattr(row, "dataset", ""),
                            "maker": None if meta is None else meta.get("maker"),
                            "title": None if meta is None else meta.get("title"),
                            "product_url": None if meta is None else meta.get("product_url"),
                        }
                    )
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)

    with tab_bias:
        st.subheader("Dataset bias overview")
        available_attrs = [attr for attr in ATTRIBUTE_CHOICES if attr in item_features_df.columns]
        if not available_attrs:
            st.write("選択可能な属性がありません。item_features.parquet を確認してください。")
        else:
            attribute = st.selectbox("Attribute", available_attrs)
            bias_top_n = st.slider("Top N", min_value=5, max_value=40, value=10, step=5, key="bias_topn")
            freq_df, pos_df = compute_dataset_attribute_counts(item_features_df.reset_index(), interactions_df, attribute, bias_top_n)
            col1, col2 = st.columns(2)
            label_angle = -45 if attribute in {"tags", "performer_ids"} else 0
            with col1:
                st.write(f"Item features: {attribute} distribution")
                st.dataframe(freq_df, use_container_width=True)
                chart = (
                    alt.Chart(freq_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            f"{attribute}:N",
                            sort="-y",
                            title=attribute.title(),
                            axis=alt.Axis(labelAngle=label_angle, labelLimit=200),
                        ),
                        y=alt.Y("count:Q", title="Count"),
                        tooltip=[attribute, "count"],
                    )
                    .properties(height=300, title=f"Item features: {attribute} count (top {bias_top_n})")
                )
                st.altair_chart(chart, use_container_width=True)
            with col2:
                st.write(f"Positive interactions: {attribute} distribution")
                st.dataframe(pos_df, use_container_width=True)
                chart_pos = (
                    alt.Chart(pos_df)
                    .mark_bar(color="#FF7F0E")
                    .encode(
                        x=alt.X(
                            f"{attribute}:N",
                            sort="-y",
                            title=attribute.title(),
                            axis=alt.Axis(labelAngle=label_angle, labelLimit=200),
                        ),
                        y=alt.Y("positive_count:Q", title="Positive count"),
                        tooltip=[attribute, "positive_count"],
                    )
                    .properties(height=300, title=f"Positive interactions: {attribute} count (top {bias_top_n})")
                    )
                st.altair_chart(chart_pos, use_container_width=True)

        breakdown_df = compute_attribute_label_breakdown(
            item_features_df.reset_index(), interactions_df, attribute, bias_top_n
        )
        st.markdown("#### Attribute label breakdown (positive vs negative)")
        st.dataframe(
            breakdown_df.rename(
                columns={
                    "positive_count": "Positive count",
                    "negative_count": "Negative count",
                    "total_count": "Total count",
                    "positive_ratio": "Positive ratio",
                    "negative_ratio": "Negative ratio",
                }
            ),
            use_container_width=True,
        )

    with tab_distribution:
        st.subheader("Recommendation distribution")
        if model_b is not None:
            model_choice = st.selectbox("Model", ["Model A", "Model B"], key="dist_model")
            chosen_model = model_a if model_choice == "Model A" else model_b
        else:
            model_choice = "Model A"
            chosen_model = model_a

        available_attrs = [attr for attr in ATTRIBUTE_CHOICES if attr in item_features_df.columns]
        if not available_attrs:
            st.write("選択可能な属性がありません。")
        else:
            attribute = st.selectbox("Recommendation attribute", available_attrs, key="dist_attribute")
            dist_top_n = st.slider("Top N", min_value=5, max_value=40, value=10, step=5, key="dist_topn")
            with st.spinner("Computing distribution..."):
                dist_df = compute_recommendation_distribution(
                    chosen_model,
                    item_features_df,
                    history_map,
                    attribute,
                    top_k,
                    exclude_seen,
                    dist_top_n,
                )
            if dist_df.empty:
                st.write("推論結果から分布を算出できませんでした。")
            else:
                st.caption(f"Total users: {len(chosen_model.user_df):,} | user_coverage = 推薦が提示されたユニークユーザー数")
                label_angle = -45 if attribute in {"tags", "performer_ids"} else 0
                st.dataframe(dist_df, use_container_width=True)
                chart = (
                    alt.Chart(dist_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            f"{attribute}:N",
                            sort="-y",
                            title=attribute.title(),
                            axis=alt.Axis(labelAngle=label_angle, labelLimit=200),
                        ),
                        y=alt.Y("recommendations:Q", title="Recommendation count"),
                        tooltip=[attribute, "recommendations", "unique_items", "user_coverage"],
                    )
                    .properties(height=300, title=f"Recommendations by {attribute} (top {dist_top_n})")
                )
                st.altair_chart(chart, use_container_width=True)

                coverage_chart = (
                    alt.Chart(dist_df)
                    .mark_bar(color="#2ca02c")
                    .encode(
                        x=alt.X(
                            f"{attribute}:N",
                            sort="-y",
                            title=attribute.title(),
                            axis=alt.Axis(labelAngle=label_angle, labelLimit=200),
                        ),
                        y=alt.Y("unique_items:Q", title="Unique items"),
                        tooltip=[attribute, "unique_items"],
                    )
                    .properties(height=300, title=f"Unique items per {attribute} (top {dist_top_n})")
                )
                st.altair_chart(coverage_chart, use_container_width=True)

                user_chart = (
                    alt.Chart(dist_df)
                    .mark_bar(color="#17becf")
                    .encode(
                        x=alt.X(
                            f"{attribute}:N",
                            sort="-y",
                            title=attribute.title(),
                            axis=alt.Axis(labelAngle=label_angle, labelLimit=200),
                        ),
                        y=alt.Y("user_coverage:Q", title="Users reached"),
                        tooltip=[attribute, "user_coverage"],
                    )
                    .properties(height=300, title=f"Users per {attribute} (top {dist_top_n})")
                )
                st.altair_chart(user_chart, use_container_width=True)


if __name__ == "__main__":
    main()
