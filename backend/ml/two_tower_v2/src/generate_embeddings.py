from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_videos
from features import ItemFeatureStore, assemble_item_feature_vector, restore_pipeline
from model import TwoTowerModel
from utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate item embeddings")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to output Parquet file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    pipeline = restore_pipeline(checkpoint["pipeline"])

    embedding_dim = config["artifacts"].get("embedding_dim", 256)
    model = TwoTowerModel(
        user_input_dim=pipeline.user_feature_dim,
        item_input_dim=pipeline.item_feature_dim,
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    videos_df = load_videos(args.videos)
    item_store = ItemFeatureStore(videos_df, pipeline.tag_vocab, pipeline.actress_vocab)

    embeddings = []
    video_ids = []
    with torch.no_grad():
        for row in videos_df.itertuples():
            video_id = str(row.id)
            item_features = item_store.build_features(video_id)
            item_vector = assemble_item_feature_vector(item_features, pipeline.item_numeric_normalizer)
            tensor = torch.tensor(item_vector, dtype=torch.float32).unsqueeze(0)
            embedding = model.encode_item(tensor).squeeze(0).numpy().astype(np.float32)
            video_ids.append(video_id)
            embeddings.append(embedding)

    df = pd.DataFrame({
        "video_id": video_ids,
        "embedding": embeddings,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
