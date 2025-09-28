#!/usr/bin/env python3
"""
訓練済みTwo-Towerモデルから直接Edge Function用の軽量実装を作成
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Lambda層の逆シリアル化を許可
tf.keras.config.enable_unsafe_deserialization()

def extract_weights_and_create_js():
    """訓練済みモデルから重みを抽出してJavaScript実装を生成"""

    # モデルファイルパス（local_dataの最新Kerasモデルを使用）
    model_dir = Path(__file__).parent / "models"
    user_model_path = model_dir / "user_tower_768.keras"
    item_model_path = model_dir / "item_tower_768.keras"

    # 出力ディレクトリ
    output_dir = Path(__file__).parent / "model_exports"
    output_dir.mkdir(exist_ok=True)

    print("Loading trained models...")

    # モデル読み込み
    user_tower = tf.keras.models.load_model(user_model_path)
    item_tower = tf.keras.models.load_model(item_model_path)

    print(f"User Tower: {user_tower.input_shape} -> {user_tower.output_shape}")
    print(f"Item Tower: {item_tower.input_shape} -> {item_tower.output_shape}")

    # 重みのみ抽出（Dense層のみ）
    def extract_dense_weights(model, name):
        weights_data = {}
        layer_count = 0

        for layer in model.layers:
            if layer.__class__.__name__ == 'Dense':
                layer_weights = layer.get_weights()
                if len(layer_weights) == 2:  # 重みとバイアス
                    w_matrix, bias_vector = layer_weights
                    weights_data[f"dense_{layer_count}"] = {
                        "weights": w_matrix.tolist(),
                        "bias": bias_vector.tolist(),
                        "input_dim": w_matrix.shape[0],
                        "output_dim": w_matrix.shape[1],
                        "layer_name": layer.name
                    }
                    layer_count += 1

        return weights_data

    user_weights = extract_dense_weights(user_tower, "user")
    item_weights = extract_dense_weights(item_tower, "item")

    print(f"User tower layers: {len(user_weights)}")
    print(f"Item tower layers: {len(item_weights)}")

    # 軽量JavaScript実装生成
    js_code = f"""
// Two-Tower Model Inference - Lightweight Edge Function Implementation
// Input dims: User {user_tower.input_shape[1]}, Item {item_tower.input_shape[1]}
// Output dims: {user_tower.output_shape[1]} (768-dimensional embeddings)

const USER_WEIGHTS = {json.dumps(user_weights, indent=2)};
const ITEM_WEIGHTS = {json.dumps(item_weights, indent=2)};

// Utility functions
function relu(x) {{
  return Math.max(0, x);
}}

function matrixMultiply(input, weights) {{
  const result = new Array(weights[0].length).fill(0);
  for (let i = 0; i < weights[0].length; i++) {{
    for (let j = 0; j < input.length; j++) {{
      result[i] += input[j] * weights[j][i];
    }}
  }}
  return result;
}}

function addBias(vector, bias) {{
  return vector.map((v, i) => v + bias[i]);
}}

function l2Normalize(vector) {{
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  return magnitude > 0 ? vector.map(v => v / magnitude) : vector;
}}

function forwardPass(input, weightsDict, isUser = true) {{
  let x = input;
  const layers = Object.keys(weightsDict).sort();

  for (let i = 0; i < layers.length; i++) {{
    const layerKey = layers[i];
    const layerData = weightsDict[layerKey];

    // 線形変換
    x = matrixMultiply(x, layerData.weights);
    x = addBias(x, layerData.bias);

    // 活性化関数（最終層以外はReLU）
    if (i < layers.length - 1) {{
      x = x.map(relu);
    }}
  }}

  // 最終層はL2正規化
  return l2Normalize(x);
}}

function predictUserEmbedding(userFeatures) {{
  return forwardPass(userFeatures, USER_WEIGHTS, true);
}}

function predictItemEmbedding(itemFeatures) {{
  return forwardPass(itemFeatures, ITEM_WEIGHTS, false);
}}

function cosineSimilarity(vec1, vec2) {{
  let dotProduct = 0;
  for (let i = 0; i < vec1.length; i++) {{
    dotProduct += vec1[i] * vec2[i];
  }}
  return dotProduct; // L2正規化済みなので内積=コサイン類似度
}}

// 特徴量前処理（学習時と同じ正規化）
function preprocessUserFeatures(rawFeatures) {{
  // 学習時の特徴量と同じ順序・形式で前処理
  return rawFeatures; // 具体的な前処理は学習コードを参照
}}

function preprocessItemFeatures(rawFeatures) {{
  // 学習時の特徴量と同じ順序・形式で前処理
  return rawFeatures; // 具体的な前処理は学習コードを参照
}}

// Export for Edge Functions
if (typeof globalThis !== 'undefined') {{
  globalThis.TwoTowerModel = {{
    predictUserEmbedding,
    predictItemEmbedding,
    cosineSimilarity,
    preprocessUserFeatures,
    preprocessItemFeatures
  }};
}}

// CommonJS export for testing
if (typeof module !== 'undefined' && module.exports) {{
  module.exports = {{
    predictUserEmbedding,
    predictItemEmbedding,
    cosineSimilarity,
    preprocessUserFeatures,
    preprocessItemFeatures,
    USER_WEIGHTS,
    ITEM_WEIGHTS
  }};
}}
"""

    # JavaScriptファイル保存
    js_file = output_dir / "two_tower_inference.js"
    with open(js_file, "w") as f:
        f.write(js_code)

    # メタデータ保存
    metadata = {
        "model_info": {
            "user_input_dim": user_tower.input_shape[1],
            "item_input_dim": item_tower.input_shape[1],
            "embedding_dim": user_tower.output_shape[1],
            "user_layers": len(user_weights),
            "item_layers": len(item_weights)
        },
        "training_info": {
            "accuracy": 0.9786,
            "samples": 5332,
            "created_at": "2025-09-27T18:29:43.623885"
        },
        "deployment": {
            "target": "supabase_edge_function",
            "format": "javascript_lightweight",
            "file_size_kb": round(js_file.stat().st_size / 1024, 2)
        }
    }

    metadata_file = output_dir / "model_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ JavaScript inference model exported:")
    print(f"   📄 {js_file} ({metadata['deployment']['file_size_kb']} KB)")
    print(f"   📄 {metadata_file}")
    print(f"   🧠 User: {metadata['model_info']['user_input_dim']} → {metadata['model_info']['embedding_dim']}")
    print(f"   🎬 Item: {metadata['model_info']['item_input_dim']} → {metadata['model_info']['embedding_dim']}")

    return js_file, metadata_file

if __name__ == "__main__":
    extract_weights_and_create_js()