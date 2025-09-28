#!/usr/bin/env python3
"""
Pure Embedding Two-Tower Model を TensorFlow.js 形式でエクスポート
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
from data_loader import SupabaseDataLoader

class PureEmbeddingModelExporter:
    """純粋埋め込みモデルのJavaScript形式エクスポート"""

    def __init__(self):
        self.models_dir = Path("models")
        self.js_output_dir = Path("../../../supabase/functions/ai-recommend-v2")

    def load_best_model(self):
        """最良モデルをロード"""
        model_path = self.models_dir / "pure_embedding_best.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Best model not found: {model_path}")

        print(f"🔄 Loading best model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model

    def extract_towers(self, full_model):
        """フルモデルからUser/ItemTowerを抽出"""
        print("🔄 Extracting User and Item towers...")

        # User Tower (like_history + nope_history -> user_embedding)
        like_input = tf.keras.Input(shape=(100,), name='like_history')
        nope_input = tf.keras.Input(shape=(100,), name='nope_history')

        # フルモデルからUser Tower部分を抽出
        user_embedding_layer = None
        for layer in full_model.layers:
            if 'pure_embedding_user' in layer.name:
                user_embedding_layer = layer
                break

        if user_embedding_layer is None:
            raise ValueError("User embedding layer not found")

        user_embedding = user_embedding_layer([like_input, nope_input])
        user_tower = tf.keras.Model(
            inputs=[like_input, nope_input],
            outputs=user_embedding,
            name='pure_embedding_user_tower'
        )

        # Item Tower (item_id -> item_embedding)
        item_input = tf.keras.Input(shape=(1,), name='item_id')

        # フルモデルからItem Tower部分を抽出
        item_embedding_layer = None
        for layer in full_model.layers:
            if 'pure_embedding_item' in layer.name:
                item_embedding_layer = layer
                break

        if item_embedding_layer is None:
            raise ValueError("Item embedding layer not found")

        item_embedding = item_embedding_layer(item_input)
        item_tower = tf.keras.Model(
            inputs=item_input,
            outputs=item_embedding,
            name='pure_embedding_item_tower'
        )

        return user_tower, item_tower

    def export_to_tfjs(self, user_tower, item_tower):
        """TensorFlow.js形式でエクスポート"""
        print("📦 Exporting to TensorFlow.js format...")

        # User Tower エクスポート
        user_output_dir = self.js_output_dir / "pure_user_tower"
        user_output_dir.mkdir(exist_ok=True)

        tfjs.converters.save_keras_model(
            user_tower,
            str(user_output_dir),
            quantization_bytes=2  # 量子化で軽量化
        )

        # Item Tower エクスポート
        item_output_dir = self.js_output_dir / "pure_item_tower"
        item_output_dir.mkdir(exist_ok=True)

        tfjs.converters.save_keras_model(
            item_tower,
            str(item_output_dir),
            quantization_bytes=2  # 量子化で軽量化
        )

        print(f"✅ User Tower exported to: {user_output_dir}")
        print(f"✅ Item Tower exported to: {item_output_dir}")

        return user_output_dir, item_output_dir

    def create_video_id_mapping(self):
        """動画IDマッピングを作成"""
        print("🔄 Creating video ID mapping...")

        # データローダーで動画IDを取得
        data_loader = SupabaseDataLoader()
        data_loader.load_all_data()

        video_ids = data_loader.videos_df['id'].unique()
        video_id_to_idx = {vid: idx+1 for idx, vid in enumerate(video_ids)}

        # JSONで保存
        mapping_path = self.js_output_dir / "video_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(video_id_to_idx, f, indent=2)

        print(f"✅ Video mapping saved: {mapping_path}")
        return video_id_to_idx

    def create_inference_wrapper(self, user_output_dir, item_output_dir):
        """JavaScriptでの推論ラッパーを作成"""
        print("🔄 Creating JavaScript inference wrapper...")

        js_code = '''// Pure Embedding Two-Tower Model - JavaScript Inference
import * as tf from 'https://cdn.skypack.dev/@tensorflow/tfjs@4.2.0';

let userTower = null;
let itemTower = null;
let videoMapping = null;

// モデル初期化
async function initModels() {
  if (!userTower || !itemTower) {
    console.log('🔄 Loading Pure Embedding Two-Tower models...');

    // モデル読み込み
    userTower = await tf.loadLayersModel('./pure_user_tower/model.json');
    itemTower = await tf.loadLayersModel('./pure_item_tower/model.json');

    // 動画IDマッピング読み込み
    const response = await fetch('./video_mapping.json');
    videoMapping = await response.json();

    console.log('✅ Pure Embedding models loaded successfully');
    console.log(`📊 Video mapping: ${Object.keys(videoMapping).length} videos`);
  }
}

// LIKE/NOPE履歴を整形（最大100件）
function formatHistory(history, maxLength = 100) {
  const mapped = history.map(videoId => videoMapping[videoId] || 0);

  // 最新maxLength件まで切り詰め
  const truncated = mapped.slice(-maxLength);

  // maxLengthまでパディング
  while (truncated.length < maxLength) {
    truncated.unshift(0);
  }

  return truncated.slice(0, maxLength);
}

// ユーザー埋め込み予測
async function predictUserEmbedding(likeHistory, nopeHistory) {
  await initModels();

  const likeFormatted = formatHistory(likeHistory);
  const nopeFormatted = formatHistory(nopeHistory);

  const likeTensor = tf.tensor2d([likeFormatted], [1, 100]);
  const nopeTensor = tf.tensor2d([nopeFormatted], [1, 100]);

  const userEmbedding = userTower.predict([likeTensor, nopeTensor]);
  const result = await userEmbedding.data();

  likeTensor.dispose();
  nopeTensor.dispose();
  userEmbedding.dispose();

  return Array.from(result);
}

// アイテム埋め込み予測
async function predictItemEmbedding(videoId) {
  await initModels();

  const videoIdx = videoMapping[videoId] || 0;
  const itemTensor = tf.tensor2d([[videoIdx]], [1, 1]);

  const itemEmbedding = itemTower.predict(itemTensor);
  const result = await itemEmbedding.data();

  itemTensor.dispose();
  itemEmbedding.dispose();

  return Array.from(result);
}

// コサイン類似度計算
function cosineSimilarity(vec1, vec2) {
  if (vec1.length !== vec2.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  const magnitude1 = Math.sqrt(norm1);
  const magnitude2 = Math.sqrt(norm2);

  if (magnitude1 === 0 || magnitude2 === 0) {
    return 0;
  }

  return dotProduct / (magnitude1 * magnitude2);
}

// グローバルエクスポート
globalThis.PureEmbeddingTwoTower = {
  predictUserEmbedding,
  predictItemEmbedding,
  cosineSimilarity,
  initModels
};

console.log('✅ Pure Embedding Two-Tower inference loaded');
'''

        # JavaScriptファイルに保存
        js_path = self.js_output_dir / "pure_embedding_model.js"
        with open(js_path, 'w') as f:
            f.write(js_code)

        print(f"✅ JavaScript wrapper saved: {js_path}")

    def export_all(self):
        """完全エクスポート実行"""
        print("🚀 Starting Pure Embedding Model Export...")

        # 1. ベストモデルロード
        full_model = self.load_best_model()

        # 2. User/Item Towerを抽出
        user_tower, item_tower = self.extract_towers(full_model)

        # 3. TensorFlow.js形式でエクスポート
        user_output_dir, item_output_dir = self.export_to_tfjs(user_tower, item_tower)

        # 4. 動画IDマッピング作成
        video_mapping = self.create_video_id_mapping()

        # 5. JavaScript推論ラッパー作成
        self.create_inference_wrapper(user_output_dir, item_output_dir)

        print("🎉 Pure Embedding Model Export completed!")
        print(f"📊 Video mapping: {len(video_mapping)} videos")
        print(f"🎯 User Tower: LIKE/NOPE history → 768D embedding")
        print(f"🎯 Item Tower: Video ID → 768D embedding")

if __name__ == "__main__":
    exporter = PureEmbeddingModelExporter()
    exporter.export_all()