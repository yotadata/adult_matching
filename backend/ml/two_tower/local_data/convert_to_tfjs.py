#!/usr/bin/env python3
"""
Two-TowerモデルをTensorFlow.js形式に変換
Edge Function展開用の軽量化モデル作成
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path

def convert_model_to_tfjs():
    """訓練済みTwo-TowerモデルをTensorFlow.js形式に変換"""

    # モデルファイルパス
    model_dir = Path(__file__).parent / "saved_models"
    user_model_path = model_dir / "user_tower.keras"
    item_model_path = model_dir / "item_tower.keras"

    # 出力ディレクトリ
    output_dir = Path(__file__).parent / "tfjs_models"
    output_dir.mkdir(exist_ok=True)

    print("Loading trained models...")

    # モデル読み込み
    user_tower = tf.keras.models.load_model(user_model_path)
    item_tower = tf.keras.models.load_model(item_model_path)

    print(f"User Tower: {user_tower.input_shape} -> {user_tower.output_shape}")
    print(f"Item Tower: {item_tower.input_shape} -> {item_tower.output_shape}")

    # TensorFlow.js形式で保存
    print("Converting User Tower to TensorFlow.js...")
    tfjs.converters.save_keras_model(
        user_tower,
        str(output_dir / "user_tower"),
        quantization_bytes=2  # 16-bit量子化で軽量化
    )

    print("Converting Item Tower to TensorFlow.js...")
    tfjs.converters.save_keras_model(
        item_tower,
        str(output_dir / "item_tower"),
        quantization_bytes=2  # 16-bit量子化で軽量化
    )

    # 特徴量情報も保存（推論時の前処理用）
    features_info = {
        "user_features": {
            "feature_count": user_tower.input_shape[1],
            "embedding_dim": user_tower.output_shape[1]
        },
        "item_features": {
            "feature_count": item_tower.input_shape[1],
            "embedding_dim": item_tower.output_shape[1]
        },
        "model_info": {
            "version": "v1.0",
            "training_samples": 5332,
            "accuracy": 0.9786,
            "created_at": "2025-09-27T18:29:43.623885"
        }
    }

    with open(output_dir / "features_info.json", "w") as f:
        json.dump(features_info, f, indent=2)

    print(f"Models converted successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    for file in output_dir.rglob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.relative_to(output_dir)}: {size_mb:.2f} MB")

if __name__ == "__main__":
    convert_model_to_tfjs()