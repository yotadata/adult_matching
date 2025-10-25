#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class TrainConfig:
    train_path: Path
    val_path: Path
    user_features_path: Path
    item_features_path: Path
    embedding_dim: int
    hidden_dim: int
    epochs: int
    batch_size: int
    lr: float
    out_dir: Path


def _ensure_list(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v) != ""]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(value)]


def _log1p_safe(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0).astype(float))


class FeatureSpace:
    """Build dense feature vectors for users/items from categorical & numeric fields."""

    def __init__(
        self,
        *,
        categorical_fields: Dict[str, Iterable[str]],
        multi_fields: Dict[str, Iterable[str]],
    ):
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self.multi_offsets: Dict[str, Tuple[int, int]] = {}
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


class InteractionsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_vectors: Dict[str, np.ndarray],
        item_vectors: Dict[str, np.ndarray],
        *,
        item_key: str,
    ):
        rows: List[Tuple[np.ndarray, np.ndarray, float]] = []
        dropped = 0
        for record in df.itertuples(index=False):
            user_vec = user_vectors.get(str(record.reviewer_id))
            item_id = getattr(record, item_key)
            item_vec = item_vectors.get(str(item_id))
            if user_vec is None or item_vec is None:
                dropped += 1
                continue
            rows.append((user_vec, item_vec, float(record.label)))
        if not rows:
            raise ValueError("No interactions remain after aligning features. Check user/item feature coverage.")

        self.user = torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.float32)
        self.item = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32)
        self.y = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        self.dropped = dropped

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.user[idx], self.item[idx], self.y[idx]


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class TwoTower(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.user_encoder = MLPEncoder(user_dim, hidden_dim, embedding_dim)
        self.item_encoder = MLPEncoder(item_dim, hidden_dim, embedding_dim)

    def forward(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        u = self.user_encoder(user_vec)
        i = self.item_encoder(item_vec)
        return (u * i).sum(dim=-1)

    def encode_user(self, user_vec: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(user_vec)

    def encode_item(self, item_vec: torch.Tensor) -> torch.Tensor:
        return self.item_encoder(item_vec)


def evaluate(model: TwoTower, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    with torch.no_grad():
        for user_vec, item_vec, y in loader:
            user_vec, item_vec, y = user_vec.to(device), item_vec.to(device), y.to(device)
            logits = model(user_vec, item_vec)
            loss = loss_fn(logits, y)
            total += loss.item() * y.size(0)
            n += y.size(0)
    return total / max(1, n)


def export_onnx(model: TwoTower, user_dim: int, item_dim: int, out_path: Path) -> None:
    model.eval()
    dummy_user = torch.zeros(1, user_dim, dtype=torch.float32)
    dummy_item = torch.zeros(1, item_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_user, dummy_item),
        out_path.as_posix(),
        input_names=["user_features", "item_features"],
        output_names=["score"],
        opset_version=17,
        dynamic_axes={
            "user_features": {0: "batch"},
            "item_features": {0: "batch"},
            "score": {0: "batch"},
        },
    )
    onnx.load(out_path.as_posix())


def build_feature_vectors(
    *,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    item_key: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, int, Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    # Prepare vocabularies
    tag_values = set()
    for values in item_df.get("tag_ids", []):
        tag_values.update(_ensure_list(values))
    for values in user_df.get("preferred_tag_ids", []):
        tag_values.update(_ensure_list(values))
    performer_values = set()
    for values in item_df.get("performer_ids", []):
        performer_values.update(_ensure_list(values))

    user_space = FeatureSpace(
        categorical_fields={},
        multi_fields={
            "preferred_tag_ids": tag_values,
        },
    )
    item_space = FeatureSpace(
        categorical_fields={
            "source": (str(v) for v in item_df.get("source", [])),
            "maker": (str(v) for v in item_df.get("maker", [])),
            "label": (str(v) for v in item_df.get("label", [])),
            "series": (str(v) for v in item_df.get("series", [])),
        },
        multi_fields={
            "tag_ids": tag_values,
            "performer_ids": performer_values,
        },
    )

    user_numeric_fields = ["like_count_30d", "positive_ratio_30d", "signup_days"]
    item_numeric_fields = ["price"]

    for col in user_numeric_fields:
        if col not in user_df.columns:
            user_df[col] = 0
    if "preferred_tag_ids" not in user_df.columns:
        user_df["preferred_tag_ids"] = [[] for _ in range(len(user_df))]

    user_vectors: Dict[str, np.ndarray] = {}
    if not user_df.empty:
        u_numeric = user_df[user_numeric_fields].copy()
        u_numeric["like_count_30d"] = _log1p_safe(u_numeric["like_count_30d"].fillna(0))
        u_numeric["signup_days"] = _log1p_safe(u_numeric["signup_days"].fillna(0))
        u_numeric["positive_ratio_30d"] = u_numeric["positive_ratio_30d"].fillna(0).clip(0, 1)
        user_numeric_array = u_numeric.to_numpy(dtype=np.float32)

        for idx, row in enumerate(user_df.itertuples(index=False)):
            vec = np.zeros(user_space.base_dim + user_numeric_array.shape[1], dtype=np.float32)
            user_space.encode_multi(vec, "preferred_tag_ids", _ensure_list(getattr(row, "preferred_tag_ids", [])))
            vec[user_space.base_dim :] = user_numeric_array[idx]
            user_vectors[str(row.reviewer_id)] = vec
    user_dim = user_space.base_dim + len(user_numeric_fields)

    item_vectors: Dict[str, np.ndarray] = {}
    for field in ["source", "maker", "label", "series"]:
        if field not in item_df.columns:
            item_df[field] = ""
    if "tag_ids" not in item_df.columns:
        item_df["tag_ids"] = [[] for _ in range(len(item_df))]
    if "performer_ids" not in item_df.columns:
        item_df["performer_ids"] = [[] for _ in range(len(item_df))]
    if "price" not in item_df.columns:
        item_df["price"] = 0.0
    if not item_df.empty:
        i_numeric = item_df[item_numeric_fields].copy()
        i_numeric["price"] = _log1p_safe(i_numeric["price"].fillna(0))
        item_numeric_array = i_numeric.to_numpy(dtype=np.float32)

        for idx, row in enumerate(item_df.itertuples(index=False)):
            vec = np.zeros(item_space.base_dim + item_numeric_array.shape[1], dtype=np.float32)
            item_space.encode_categorical(vec, "source", str(getattr(row, "source", "")))
            item_space.encode_categorical(vec, "maker", str(getattr(row, "maker", "")))
            item_space.encode_categorical(vec, "label", str(getattr(row, "label", "")))
            item_space.encode_categorical(vec, "series", str(getattr(row, "series", "")))
            item_space.encode_multi(vec, "tag_ids", _ensure_list(getattr(row, "tag_ids", [])))
            item_space.encode_multi(vec, "performer_ids", _ensure_list(getattr(row, "performer_ids", [])))
            vec[item_space.base_dim :] = item_numeric_array[idx]
            item_vectors[str(getattr(row, item_key))] = vec
    item_dim = item_space.base_dim + len(item_numeric_fields)

    return user_vectors, item_vectors, user_dim, item_dim, user_space.multi_vocab, item_space.multi_vocab


def compute_embeddings(model: TwoTower, vectors: Dict[str, np.ndarray], encode_fn, device: torch.device, id_key: str) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for entity_id, vec in vectors.items():
            tens = torch.from_numpy(vec).unsqueeze(0).to(device)
            emb = encode_fn(tens).cpu().numpy()[0]
            rows.append({id_key: entity_id, "embedding": emb.tolist()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a feature-based Two-Tower model (PyTorch)")
    ap.add_argument("--train", type=Path, default=Path("ml/data/processed/two_tower/latest/interactions_train.parquet"))
    ap.add_argument("--val", type=Path, default=Path("ml/data/processed/two_tower/latest/interactions_val.parquet"))
    ap.add_argument("--user-features", type=Path, default=Path("ml/data/processed/two_tower/latest/user_features.parquet"))
    ap.add_argument("--item-features", type=Path, default=Path("ml/data/processed/two_tower/latest/item_features.parquet"))
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("ml/artifacts"))
    ap.add_argument("--item-key", choices=["product_url", "video_id"], default="video_id", help="Column to treat as item identifier.")
    args = ap.parse_args()

    cfg = TrainConfig(
        train_path=args.train,
        val_path=args.val,
        user_features_path=args.user_features,
        item_features_path=args.item_features,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
    )

    if not cfg.user_features_path.exists():
        raise FileNotFoundError(f"user features not found: {cfg.user_features_path}")
    if not cfg.item_features_path.exists():
        raise FileNotFoundError(f"item features not found: {cfg.item_features_path}")

    train_df = pd.read_parquet(cfg.train_path)
    val_df = pd.read_parquet(cfg.val_path)
    user_df = pd.read_parquet(cfg.user_features_path)
    item_df = pd.read_parquet(cfg.item_features_path)

    item_key = args.item_key if args.item_key in item_df.columns else "video_id"

    user_vectors, item_vectors, user_dim, item_dim, user_multi_vocab, item_multi_vocab = build_feature_vectors(
        user_df=user_df,
        item_df=item_df,
        item_key=item_key,
    )

    train_ds = InteractionsDataset(train_df, user_vectors, item_vectors, item_key=item_key)
    val_ds = InteractionsDataset(val_df, user_vectors, item_vectors, item_key=item_key)
    if train_ds.dropped:
        print(json.dumps({"info": "dropped_train_rows_missing_features", "count": train_ds.dropped}))
    if val_ds.dropped:
        print(json.dumps({"info": "dropped_val_rows_missing_features", "count": val_ds.dropped}))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(user_dim=user_dim, item_dim=item_dim, hidden_dim=cfg.hidden_dim, embedding_dim=cfg.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for user_vec, item_vec, y in pbar:
            user_vec, item_vec, y = user_vec.to(device), item_vec.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(user_vec, item_vec)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        val_loss = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "val_loss": val_loss}))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.out_dir / "two_tower_latest.pt")

    model.load_state_dict(torch.load(cfg.out_dir / "two_tower_latest.pt", map_location=device))
    model.eval()

    export_onnx(model.cpu(), user_dim, item_dim, cfg.out_dir / "two_tower_latest.onnx")

    # Compute embeddings
    model = model.to(device)
    user_embeddings_df = compute_embeddings(model, user_vectors, model.encode_user, device, "reviewer_id")
    item_embeddings_df = compute_embeddings(model, item_vectors, model.encode_item, device, item_key)
    user_embeddings_df.to_parquet(cfg.out_dir / "user_embeddings.parquet", index=False)
    item_embeddings_df.to_parquet(cfg.out_dir / "video_embeddings.parquet", index=False)

    meta = {
        "embedding_dim": cfg.embedding_dim,
        "hidden_dim": cfg.hidden_dim,
        "user_feature_dim": int(user_dim),
        "item_feature_dim": int(item_dim),
        "num_users_training": int(len(train_df["reviewer_id"].unique())),
        "num_items_training": int(len(train_df[item_key].unique())),
        "tag_vocab_size": int(len(item_multi_vocab.get("tag_ids", {}))),
        "performer_vocab_size": int(len(item_multi_vocab.get("performer_ids", {}))),
        "preferred_tag_vocab_size": int(len(user_multi_vocab.get("preferred_tag_ids", {}))),
        "input_schema_version": 1,
        "format": "two_tower.feature_mlp",
    }
    with (cfg.out_dir / "model_meta.json").open("w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved artifacts to", cfg.out_dir)


if __name__ == "__main__":
    main()
