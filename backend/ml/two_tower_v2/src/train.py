from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.table import Table

from data import build_training_samples, load_decisions, load_profiles, load_videos, split_train_val
from features import (
    FeaturePipeline,
    UserFeatureStore,
    ItemFeatureStore,
    assemble_item_feature_vector,
    assemble_user_feature_vector,
    build_feature_pipeline,
)
from model import TwoTowerModel
from utils import ensure_dir, load_yaml, set_global_seed

console = Console()


class InteractionDataset(Dataset):
    def __init__(
        self,
        df,
        user_store: UserFeatureStore,
        item_store: ItemFeatureStore,
        pipeline: FeaturePipeline,
    ) -> None:
        user_vectors = []
        item_vectors = []
        labels = []
        user_ids = []
        video_ids = []
        for row in df.itertuples():
            user_id = str(row.user_id)
            video_id = str(row.video_id)
            user_feat = user_store.build_features(user_id)
            item_feat = item_store.build_features(video_id)
            user_vector = assemble_user_feature_vector(user_feat, pipeline.user_numeric_normalizer)
            item_vector = assemble_item_feature_vector(item_feat, pipeline.item_numeric_normalizer)
            user_vectors.append(user_vector)
            item_vectors.append(item_vector)
            labels.append(float(row.label))
            user_ids.append(user_id)
            video_ids.append(video_id)
        self.user_vectors = torch.tensor(np.stack(user_vectors), dtype=torch.float32)
        self.item_vectors = torch.tensor(np.stack(item_vectors), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        self.user_ids = user_ids
        self.video_ids = video_ids

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.user_vectors[idx], self.item_vectors[idx], self.labels[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Tower model v2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def create_dataloaders(
    train_df,
    val_df,
    user_store: UserFeatureStore,
    item_store: ItemFeatureStore,
    pipeline: FeaturePipeline,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader | None]:
    train_dataset = InteractionDataset(train_df, user_store, item_store, pipeline)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = None
    if len(val_df) > 0:
        val_dataset = InteractionDataset(val_df, user_store, item_store, pipeline)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def evaluate(model: TwoTowerModel, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for user_vec, item_vec, labels in data_loader:
            user_vec = user_vec.to(device)
            item_vec = item_vec.to(device)
            labels = labels.to(device)
            logits = model(user_vec, item_vec)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_global_seed(config.get("seed", 42))

    console.rule("データ読込")
    profiles_df = load_profiles(args.profiles)
    videos_df = load_videos(args.videos)
    decisions_df = load_decisions(args.decisions)

    samples_df = build_training_samples(
        profiles_df,
        videos_df,
        decisions_df,
        negative_ratio=config["training"].get("negative_sampling_ratio", 3),
        seed=config.get("seed", 42),
    )
    console.print(f"サンプル数: {len(samples_df)} (正例: {(samples_df['label']==1).sum()}, 負例: {(samples_df['label']==0).sum()})")

    pipeline, user_store, item_store = build_feature_pipeline(
        profiles_df,
        videos_df,
        samples_df,
        max_tag_vocab=config["artifacts"].get("max_tag_vocab", 512),
        min_tag_freq=config["artifacts"].get("min_tag_freq", 3),
        max_actress_vocab=config["artifacts"].get("max_actress_vocab", 256),
        min_actress_freq=config["artifacts"].get("min_actress_freq", 2),
    )
    console.print(f"ユーザー入力次元: {pipeline.user_feature_dim}, アイテム入力次元: {pipeline.item_feature_dim}")

    train_df, val_df = split_train_val(
        samples_df,
        val_ratio=config["training"].get("train_val_split", 0.1),
        seed=config.get("seed", 42),
    )

    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        user_store,
        item_store,
        pipeline,
        batch_size=config["training"].get("batch_size", 256),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(
        user_input_dim=pipeline.user_feature_dim,
        item_input_dim=pipeline.item_feature_dim,
        embedding_dim=config["artifacts"].get("embedding_dim", 256),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("lr", 0.001),
        weight_decay=config["training"].get("weight_decay", 1e-5),
    )

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = config["training"].get("early_stopping_patience", 5)
    best_state = {
        "model_state": model.state_dict(),
        "pipeline": pipeline.export_metadata(),
        "config": config,
    }

    console.rule("学習開始")
    for epoch in range(1, config["training"].get("epochs", 25) + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for step, (user_vec, item_vec, labels) in enumerate(train_loader, start=1):
            user_vec = user_vec.to(device)
            item_vec = item_vec.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(user_vec, item_vec)
            loss = criterion(logits, labels)
            loss.backward()
            grad_clip = config["training"].get("gradient_clip_norm")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_samples += labels.size(0)

        epoch_loss = running_loss / max(1, running_samples)
        console.print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}")

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            val_loss = metrics["loss"]
            console.print(f"           val_loss={val_loss:.4f}, val_acc={metrics['accuracy']:.4f}")
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    "model_state": model.state_dict(),
                    "pipeline": pipeline.export_metadata(),
                    "config": config,
                }
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    console.print("Early stopping triggered")
                    break
        else:
            best_state = {
                "model_state": model.state_dict(),
                "pipeline": pipeline.export_metadata(),
                "config": config,
            }

    ensure_dir(args.output_dir)
    checkpoint_path = args.output_dir / "latest.pt"
    torch.save(best_state, checkpoint_path)
    console.print(f"Checkpoint saved to {checkpoint_path}")

    # Export training summary
    summary = Table("metric", "value")
    summary.add_row("train_samples", str(len(train_df)))
    summary.add_row("val_samples", str(len(val_df)))
    summary.add_row("user_feature_dim", str(pipeline.user_feature_dim))
    summary.add_row("item_feature_dim", str(pipeline.item_feature_dim))
    summary.add_row("tag_vocab_size", str(len(pipeline.tag_vocab.tokens)))
    summary.add_row("actress_vocab_size", str(len(pipeline.actress_vocab.tokens)))
    console.print(summary)


if __name__ == "__main__":
    main()
