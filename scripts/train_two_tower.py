#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm


@dataclass
class TrainConfig:
    train_path: Path
    val_path: Path
    embedding_dim: int
    epochs: int
    batch_size: int
    lr: float
    out_dir: Path


class InteractionsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user_map: Dict[str, int], item_map: Dict[str, int], item_key: str):
        # Cast to str before mapping to align with mapping keys
        self.u = df["reviewer_id"].astype(str).map(user_map).astype(np.int64).to_numpy()
        self.i = df[item_key].astype(str).map(item_map).astype(np.int64).to_numpy()
        self.y = df["label"].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.u[idx], dtype=torch.long),
            torch.tensor(self.i[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(u_idx)  # [B, D]
        i = self.item_emb(i_idx)  # [B, D]
        # dot product
        return (u * i).sum(dim=-1)


def build_id_maps(train: pd.DataFrame, val: pd.DataFrame, item_key: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    users = pd.concat([train["reviewer_id"].astype(str), val["reviewer_id"].astype(str)]).unique()
    items = pd.concat([train[item_key].astype(str), val[item_key].astype(str)]).unique()
    user_map = {str(u): idx for idx, u in enumerate(users)}
    item_map = {str(it): idx for idx, it in enumerate(items)}
    return user_map, item_map


def evaluate(model: TwoTower, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    with torch.no_grad():
        for u, i, y in loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            logits = model(u, i)
            loss = loss_fn(logits, y)
            total += loss.item() * y.size(0)
            n += y.size(0)
    return total / max(1, n)


def export_onnx(model: TwoTower, out_path: Path):
    model.eval()
    dummy_u = torch.tensor([0], dtype=torch.long)
    dummy_i = torch.tensor([0], dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_u, dummy_i),
        out_path.as_posix(),
        input_names=["user_idx", "item_idx"],
        output_names=["score"],
        opset_version=17,
        dynamic_axes={
            "user_idx": {0: "batch"},
            "item_idx": {0: "batch"},
            "score": {0: "batch"},
        },
    )
    # Validate ONNX
    onnx.load(out_path.as_posix())


def main():
    ap = argparse.ArgumentParser(description="Train a simple Two-Tower model (PyTorch)")
    ap.add_argument("--train", type=Path, default=Path("data/interactions_train.parquet"))
    ap.add_argument("--val", type=Path, default=Path("data/interactions_val.parquet"))
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--item-key", choices=["product_url", "video_id"], default="product_url", help="Which column to use as item key if present in parquet")
    args = ap.parse_args()

    cfg = TrainConfig(
        train_path=args.train,
        val_path=args.val,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
    )

    train_df = pd.read_parquet(cfg.train_path)
    val_df = pd.read_parquet(cfg.val_path)
    item_key = args.item_key if args.item_key in train_df.columns else "product_url"

    user_map, item_map = build_id_maps(train_df, val_df, item_key)
    train_ds = InteractionsDataset(train_df, user_map, item_map, item_key)
    val_ds = InteractionsDataset(val_df, user_map, item_map, item_key)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(num_users=len(user_map), num_items=len(item_map), dim=cfg.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for u, i, y in pbar:
            u, i, y = u.to(device), i.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(u, i)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        val_loss = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "val_loss": val_loss}))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.out_dir / "two_tower_latest.pt")

    # Final exports
    # Save mappings
    (cfg.out_dir / "mappings").mkdir(exist_ok=True)
    with (cfg.out_dir / "mappings" / "user_id_map.json").open("w") as f:
        json.dump(user_map, f)
    with (cfg.out_dir / "mappings" / "item_id_map.json").open("w") as f:
        json.dump(item_map, f)

    # Export ONNX for cross-language inference
    export_onnx(model.cpu(), cfg.out_dir / "two_tower_latest.onnx")

    # Export embeddings for DB upsert or offline use
    user_emb = model.user_emb.weight.detach().cpu().numpy()
    item_emb = model.item_emb.weight.detach().cpu().numpy()
    # Save as parquet with columns [id, embedding]
    pd.DataFrame({
        "reviewer_id": list(user_map.keys()),
        "user_idx": list(user_map.values()),
        "embedding": list(map(lambda v: v.tolist(), user_emb)),
    }).to_parquet(cfg.out_dir / "user_embeddings.parquet", index=False)
    pd.DataFrame({
        item_key: list(item_map.keys()),
        "item_idx": list(item_map.values()),
        "embedding": list(map(lambda v: v.tolist(), item_emb)),
    }).to_parquet(cfg.out_dir / "video_embeddings.parquet", index=False)

    # Save meta
    meta = {
        "embedding_dim": cfg.embedding_dim,
        "num_users": int(user_emb.shape[0]),
        "num_items": int(item_emb.shape[0]),
        "format": "two_tower.dot",
    }
    with (cfg.out_dir / "model_meta.json").open("w") as f:
        json.dump(meta, f)

    print("Saved artifacts to", cfg.out_dir)


if __name__ == "__main__":
    main()
