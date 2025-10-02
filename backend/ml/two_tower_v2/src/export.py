from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from features import restore_pipeline
from model import TwoTowerModel
from utils import ensure_dir, load_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Two-Tower artifacts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    pipeline_meta = checkpoint["pipeline"]
    pipeline = restore_pipeline(pipeline_meta)

    embedding_dim = config["artifacts"].get("embedding_dim", 256)

    model = TwoTowerModel(
        user_input_dim=pipeline.user_feature_dim,
        item_input_dim=pipeline.item_feature_dim,
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ensure_dir(args.output_dir)

    # Export ONNX (User Tower)
    onnx_path = args.output_dir / "user_tower.onnx"
    dummy_input = torch.randn(1, pipeline.user_feature_dim)
    input_name = config["export"].get("onnx_user_input_name", "user_features")
    torch.onnx.export(
        model.user_tower,
        dummy_input,
        onnx_path,
        input_names=[input_name],
        output_names=["user_embedding"],
        dynamic_axes={input_name: {0: "batch"}, "user_embedding": {0: "batch"}},
        opset_version=config["export"].get("onnx_opset", 17),
    )

    # Feature schema
    feature_schema = {
        "version": "v2",
        "input_name": input_name,
        "input_dim": pipeline.user_feature_dim,
        "embedding_dim": embedding_dim,
        "segments": [
            {
                "name": "numeric",
                "size": len(pipeline.user_numeric_normalizer.feature_names),
                "features": pipeline.user_numeric_normalizer.feature_names,
                "type": "standard_score",
            },
            {
                "name": "hour_of_day",
                "size": 1,
                "type": "scaled_bucket",
                "range": [0, 1],
            },
            {
                "name": "tag_vector",
                "size": len(pipeline.tag_vocab.tokens),
                "type": "l2_normalized_bow",
            },
            {
                "name": "actress_vector",
                "size": len(pipeline.actress_vocab.tokens),
                "type": "l2_normalized_bow",
            },
        ],
    }
    write_json(args.output_dir / "feature_schema.json", feature_schema)

    # Normalizer stats
    normalizer_payload = {
        "user_numeric": pipeline.user_numeric_normalizer.stats(),
        "item_numeric": pipeline.item_numeric_normalizer.stats(),
    }
    write_json(args.output_dir / "normalizer.json", normalizer_payload)

    # Vocabularies
    write_json(args.output_dir / "vocab_tag.json", {"tokens": pipeline.tag_vocab.tokens})
    write_json(args.output_dir / "vocab_actress.json", {"tokens": pipeline.actress_vocab.tokens})

    # Model metadata
    model_meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "embedding_dim": embedding_dim,
        "user_feature_dim": pipeline.user_feature_dim,
        "item_feature_dim": pipeline.item_feature_dim,
        "tag_vocab_size": len(pipeline.tag_vocab.tokens),
        "actress_vocab_size": len(pipeline.actress_vocab.tokens),
        "config": config,
    }
    write_json(args.output_dir / "model_meta.json", model_meta)

    print(f"Artifacts exported to {args.output_dir}")


if __name__ == "__main__":
    main()
