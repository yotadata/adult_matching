#!/usr/bin/env python3
"""
Enhanced Two-Tower Model with Rich Features
豊富な特徴量を活用した改良版Two-Towerモデル
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
from pathlib import Path
from data_loader import SupabaseDataLoader

class TwoTowerModel:
    """豊富な特徴量を活用するTwo-Towerモデル"""

    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)

        # モデル
        self.user_tower = None
        self.item_tower = None
        self.full_model = None

        # データローダー
        self.data_loader = SupabaseDataLoader()

    def build_enhanced_user_tower(self, input_dim: int) -> tf.keras.Model:
        """拡張ユーザータワー（13次元入力）"""
        inputs = layers.Input(shape=(input_dim,), name='user_features')

        # 第1層: より大きな隠れ層（13次元→512次元）
        x = layers.Dense(512, activation='relu', name='user_dense1')(inputs)
        x = layers.Dropout(0.3, name='user_dropout1')(x)

        # 第2層: 中間層（512次元→256次元）
        x = layers.Dense(256, activation='relu', name='user_dense2')(x)
        x = layers.Dropout(0.2, name='user_dropout2')(x)

        # 第3層: さらなる中間層（256次元→128次元）
        x = layers.Dense(128, activation='relu', name='user_dense3')(x)
        x = layers.Dropout(0.1, name='user_dropout3')(x)

        # 最終層: 768次元埋め込み
        user_dense = layers.Dense(self.embedding_dim, name='user_embedding_dense')(x)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                                      output_shape=(self.embedding_dim,), name='user_embedding')(user_dense)

        model = models.Model(inputs=inputs, outputs=user_embedding, name='enhanced_user_tower')
        return model

    def build_enhanced_item_tower(self, input_dim: int) -> tf.keras.Model:
        """拡張アイテムタワー（229次元入力）"""
        inputs = layers.Input(shape=(input_dim,), name='item_features')

        # 第1層: 大きな隠れ層（229次元→1024次元）
        x = layers.Dense(1024, activation='relu', name='item_dense1')(inputs)
        x = layers.Dropout(0.4, name='item_dropout1')(x)

        # 第2層: 中間層（1024次元→512次元）
        x = layers.Dense(512, activation='relu', name='item_dense2')(x)
        x = layers.Dropout(0.3, name='item_dropout2')(x)

        # 第3層: 中間層（512次元→256次元）
        x = layers.Dense(256, activation='relu', name='item_dense3')(x)
        x = layers.Dropout(0.2, name='item_dropout3')(x)

        # 第4層: さらなる中間層（256次元→128次元）
        x = layers.Dense(128, activation='relu', name='item_dense4')(x)
        x = layers.Dropout(0.1, name='item_dropout4')(x)

        # 最終層: 768次元埋め込み
        item_dense = layers.Dense(self.embedding_dim, name='item_embedding_dense')(x)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                                      output_shape=(self.embedding_dim,), name='item_embedding')(item_dense)

        model = models.Model(inputs=inputs, outputs=item_embedding, name='enhanced_item_tower')
        return model

    def build_full_model(self, user_input_dim: int, item_input_dim: int) -> tf.keras.Model:
        """完全なTwo-Towerモデル"""
        # タワー構築
        self.user_tower = self.build_enhanced_user_tower(user_input_dim)
        self.item_tower = self.build_enhanced_item_tower(item_input_dim)

        # 入力
        user_input = layers.Input(shape=(user_input_dim,), name='user_features')
        item_input = layers.Input(shape=(item_input_dim,), name='item_features')

        # 埋め込み計算
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)

        # コサイン類似度計算
        similarity = layers.Dot(axes=1, normalize=False, name='similarity')([user_embedding, item_embedding])

        # 最終予測（シグモイド）
        prediction = layers.Dense(1, activation='sigmoid', name='prediction')(similarity)

        # モデル構築
        model = models.Model(
            inputs=[user_input, item_input],
            outputs=prediction,
            name='enhanced_two_tower_full'
        )

        return model

    def train(self, data_loader: EnhancedSupabaseDataLoader):
        """豊富な特徴量でモデル学習"""
        print("🚀 Starting Enhanced Two-Tower training...")

        # データ読み込み
        data_loader.load_all_data()
        training_data, labels = data_loader.get_enhanced_training_data()

        # 特徴量生成（各サンプルごと）
        user_features = []
        item_features = []

        for _, row in training_data.iterrows():
            # ユーザー特徴量を生成
            user_feat = data_loader.get_single_user_features(row)
            user_features.append(user_feat)

            # アイテム特徴量を生成
            item_feat = data_loader.get_single_video_features(row)
            item_features.append(item_feat)

        user_features = np.array(user_features)
        item_features = np.array(item_features)

        print(f"📊 Enhanced features:")
        print(f"   User features shape: {user_features.shape}")
        print(f"   Item features shape: {item_features.shape}")
        print(f"   Labels shape: {labels.shape}")

        # モデル構築
        self.full_model = self.build_full_model(user_features.shape[1], item_features.shape[1])

        print(f"\n📋 Enhanced Model Architecture:")
        self.full_model.summary()

        # コンパイル
        self.full_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        # データ分割
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

        train_user_features = user_features[train_idx]
        train_item_features = item_features[train_idx]
        train_labels = labels[train_idx]

        val_user_features = user_features[val_idx]
        val_item_features = item_features[val_idx]
        val_labels = labels[val_idx]

        print(f"🏋️ Training enhanced model...")
        print(f"   Training samples: {len(train_labels)}")
        print(f"   Validation samples: {len(val_labels)}")

        # 学習実行
        history = self.full_model.fit(
            [train_user_features, train_item_features],
            train_labels,
            epochs=10,
            batch_size=32,
            validation_data=([val_user_features, val_item_features], val_labels),
            verbose=1
        )

        # 評価
        val_predictions = self.full_model.predict([val_user_features, val_item_features])
        val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(val_labels, val_predictions_binary)
        precision = precision_score(val_labels, val_predictions_binary)
        recall = recall_score(val_labels, val_predictions_binary)
        f1 = f1_score(val_labels, val_predictions_binary)

        print(f"\n✅ Enhanced Validation Results:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")

        return history

    def save_models(self):
        """モデルとメタデータ保存"""
        print("💾 Saving models...")

        # Keras形式で保存
        self.user_tower.save(self.output_dir / "user_tower_768.keras")
        self.item_tower.save(self.output_dir / "item_tower_768.keras")

        # エンコーダー保存
        encoders_data = {
            'tag_binarizer': self.data_loader.tag_binarizer,
            'frequent_tags': list(self.data_loader.frequent_tags) if hasattr(self.data_loader, 'frequent_tags') else [],
            'tag_classes': list(self.data_loader.tag_binarizer.classes)
        }

        with open(self.output_dir / "preprocessors.pkl", 'wb') as f:
            pickle.dump(encoders_data, f)

        # メタデータ更新
        metadata = {
            "embedding_dim": 768,
            "model_version": "2.0",
            "created_at": "2025-09-27T19:50:00.000000",
            "supabase_compatible": True,
            "edge_function_ready": True,
            "enhanced_features": True,
            "user_features": {
                "dimensions": 13,
                "features": ["display_name_length", "account_age_days", "total_decisions", "like_ratio",
                           "avg_liked_price", "price_preference_high", "like_count", "nope_count"] +
                           [f"top_tag_pref_{i}" for i in range(5)]
            },
            "video_features": {
                "dimensions": 229,
                "basic_features": ["title_length", "price", "release_year", "has_thumbnail", "total_decisions", "like_ratio"],
                "metadata_features": ["maker_length", "director_length", "series_length", "has_sample", "has_preview", "image_count", "duration"],
                "tag_features": f"{len(self.data_loader.tag_binarizer.classes_)} tags one-hot encoded"
            },
            "data_stats": {
                "total_tags": len(self.data_loader.all_tags),
                "frequent_tags": len(self.data_loader.frequent_tags),
                "unique_makers": 214,
                "unique_directors": 370,
                "unique_series": 929
            }
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Models saved to: {self.output_dir}")
        print(f"   📁 Keras models: user_tower, item_tower")
        print(f"   📁 Encoders: tag_binarizer and metadata")
        print(f"   📁 Metadata: comprehensive feature information")

if __name__ == "__main__":
    print(f"🚀 Two-Tower training starting...")

    # Two-Tower学習実行
    model = TwoTowerModel()
    loader = SupabaseDataLoader()

    try:
        # 学習実行
        history = model.train(loader)

        # モデル保存
        if model.user_tower is not None and model.item_tower is not None:
            model.save_models()
            print(f"\n🎉 Two-Tower training completed!")
            print(f"🚀 Ready for Edge Function deployment with rich features!")
        else:
            print("❌ Models not trained properly")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()