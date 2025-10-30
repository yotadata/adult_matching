#!/usr/bin/env python3
"""
Evaluate Two-Tower model artifacts on a validation dataset.

Metrics:
  - Binary cross entropy with logits
  - ROC-AUC
  - Recall@K (averaged over users with at least one positive item)
The script relies on precomputed embeddings exported during training
(`user_embeddings.parquet`, `video_embeddings.parquet`) so ID マップ無し
でも推論可能な構造を維持できる。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _load_embeddings(path: Path, id_column: str) -> Dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    embeddings: Dict[str, np.ndarray] = {}
    for row in df.itertuples(index=False):
        entity_id = str(getattr(row, id_column))
        vec = np.asarray(getattr(row, "embedding"), dtype=np.float32)
        embeddings[entity_id] = vec
    return embeddings


def _bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    # Same as torch.nn.functional.binary_cross_entropy_with_logits
    return float(
        np.mean(
            np.clip(logits, 0, None)
            - logits * labels
            + np.log1p(np.exp(-np.abs(logits)))
        )
    )


def evaluate(
    val_path: Path,
    user_embeddings_path: Path,
    item_embeddings_path: Path,
    item_key: str,
    recall_k: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    val_df = pd.read_parquet(val_path)
    user_emb = _load_embeddings(user_embeddings_path, "reviewer_id")
    item_emb = _load_embeddings(item_embeddings_path, item_key)

    logits: List[float] = []
    labels: List[float] = []
    per_user_scores: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    missing_user = missing_item = 0

    for row in val_df.itertuples(index=False):
        reviewer_id = str(row.reviewer_id)
        item_id = str(getattr(row, item_key))
        user_vec = user_emb.get(reviewer_id)
        item_vec = item_emb.get(item_id)
        if user_vec is None:
            missing_user += 1
            continue
        if item_vec is None:
            missing_item += 1
            continue
        score = float(np.dot(user_vec, item_vec))
        label = float(getattr(row, "label"))
        logits.append(score)
        labels.append(label)
        per_user_scores[reviewer_id].append((score, label))

    if not logits:
        raise RuntimeError("No validation rows could be scored (all missing embeddings).")

    logits_arr = np.asarray(logits, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)

    metrics: Dict[str, float] = {
        "bce_with_logits": _bce_with_logits(logits_arr, labels_arr),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(labels_arr, logits_arr))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    recall_values: List[float] = []
    for _, scores in per_user_scores.items():
        positives = [label for _, label in scores if label > 0.0]
        if not positives:
            continue  # Skip users without positive examples
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        top_k = sorted_scores[: recall_k]
        hits = sum(1.0 for _, label in top_k if label > 0.0)
        recall_values.append(hits / len(positives))

    metrics["recall_at_k"] = float(np.mean(recall_values)) if recall_values else float("nan")

    stats = {
        "scored_rows": len(logits),
        "total_val_rows": len(val_df),
        "missing_user_embeddings": missing_user,
        "missing_item_embeddings": missing_item,
        "users_evaluated": len(recall_values),
    }

    return metrics, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Two-Tower model artifacts.")
    ap.add_argument(
        "--val",
        type=Path,
        default=Path("ml/data/processed/two_tower/latest/interactions_val.parquet"),
        help="Validation interactions parquet.",
    )
    ap.add_argument(
        "--user-embeddings",
        type=Path,
        default=Path("ml/artifacts/user_embeddings.parquet"),
        help="User embeddings parquet exported during training.",
    )
    ap.add_argument(
        "--item-embeddings",
        type=Path,
        default=Path("ml/artifacts/video_embeddings.parquet"),
        help="Item embeddings parquet exported during training.",
    )
    ap.add_argument(
        "--item-key",
        choices=["video_id", "product_url"],
        default="video_id",
        help="Column to use as item identifier in the validation data.",
    )
    ap.add_argument(
        "--recall-k",
        type=int,
        default=20,
        help="K value for Recall@K.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("ml/artifacts/metrics.json"),
        help="Path to write evaluation metrics as JSON.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    metrics, stats = evaluate(
        val_path=args.val,
        user_embeddings_path=args.user_embeddings,
        item_embeddings_path=args.item_embeddings,
        item_key=args.item_key,
        recall_k=args.recall_k,
    )

    result = {
        "metrics": metrics,
        "stats": stats,
        "recall_k": args.recall_k,
        "val_path": str(args.val),
        "user_embeddings": str(args.user_embeddings),
        "item_embeddings": str(args.item_embeddings),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
