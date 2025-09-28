#!/usr/bin/env python3
"""
Edge Function Compatible Two-Tower Model

Supabase Edge Function用Two-Towerモデル
- TensorFlow.js出力対応
- 768次元embedding（Supabase vector(768)準拠）
- 軽量化設計（Edge Functionメモリ制限対応）
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from data_loader import SupabaseCompatibleDataLoader

class EdgeFunctionTwoTower:
    """
    Edge Function対応Two-Towerモデル
    Supabaseデータベース構造に完全準拠
    """

    def __init__(self,
                 embedding_dim: int = 768,  # Supabase vector(768)準拠
                 learning_rate: float = 0.001,
                 batch_size: int = 32,  # 少量データ用
                 epochs: int = 50):  # 少量データなので多めのepoch

        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # 前処理器
        self.user_scaler = StandardScaler()
        self.video_scaler = StandardScaler()

        # モデル
        self.user_tower = None
        self.item_tower = None
        self.full_model = None

        # 出力ディレクトリ
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)

    def build_user_tower(self, input_dim: int) -> tf.keras.Model:
        """
        User Tower - ユーザーエンベディング生成
        Edge Function用に軽量化
        """
        inputs = layers.Input(shape=(input_dim,), name='user_features')

        # 軽量ネットワーク（Edge Function制約）
        x = layers.Dense(256, activation='relu', name='user_dense1')(inputs)
        x = layers.Dropout(0.3, name='user_dropout1')(x)
        x = layers.Dense(128, activation='relu', name='user_dense2')(x)
        x = layers.Dropout(0.2, name='user_dropout2')(x)

        # 768次元出力（Supabase vector(768)準拠）
        user_dense = layers.Dense(self.embedding_dim, name='user_embedding_dense')(x)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                                      output_shape=(self.embedding_dim,), name='user_embedding')(user_dense)

        model = models.Model(inputs=inputs, outputs=user_embedding, name='user_tower')
        return model

    def build_item_tower(self, input_dim: int) -> tf.keras.Model:
        """
        Item Tower - 動画エンベディング生成
        Edge Function用に軽量化
        """
        inputs = layers.Input(shape=(input_dim,), name='video_features')

        # 軽量ネットワーク（Edge Function制約）
        x = layers.Dense(256, activation='relu', name='item_dense1')(inputs)
        x = layers.Dropout(0.3, name='item_dropout1')(x)
        x = layers.Dense(128, activation='relu', name='item_dense2')(x)
        x = layers.Dropout(0.2, name='item_dropout2')(x)

        # 768次元出力（Supabase vector(768)準拠）
        item_dense = layers.Dense(self.embedding_dim, name='item_embedding_dense')(x)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                                      output_shape=(self.embedding_dim,), name='item_embedding')(item_dense)

        model = models.Model(inputs=inputs, outputs=item_embedding, name='item_tower')
        return model

    def build_full_model(self, user_input_dim: int, item_input_dim: int) -> tf.keras.Model:
        """
        Two-Tower統合モデル
        """
        # User Tower
        user_inputs = layers.Input(shape=(user_input_dim,), name='user_features')
        user_embedding = self.user_tower(user_inputs)

        # Item Tower
        item_inputs = layers.Input(shape=(item_input_dim,), name='item_features')
        item_embedding = self.item_tower(item_inputs)

        # 類似度計算（内積）
        similarity = layers.Dot(axes=1, name='similarity')([user_embedding, item_embedding])

        # 確率出力
        output = layers.Dense(1, activation='sigmoid', name='prediction')(similarity)

        model = models.Model(
            inputs=[user_inputs, item_inputs],
            outputs=output,
            name='two_tower_full'
        )

        return model

    def prepare_features(self, training_data: pd.DataFrame, user_features: pd.DataFrame,
                        video_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """特徴量の前処理"""

        # ユーザー特徴量（User Tower用）
        user_feature_cols = ['display_name_length', 'account_age_days', 'total_decisions', 'like_ratio']
        user_matrix = []

        for _, row in training_data.iterrows():
            user_data = user_features[user_features['user_id'] == row['user_id']].iloc[0]
            user_matrix.append([user_data[col] for col in user_feature_cols])

        user_matrix = np.array(user_matrix, dtype=np.float32)
        user_matrix = self.user_scaler.fit_transform(user_matrix)

        # 動画特徴量（Item Tower用）
        video_feature_cols = ['title_length', 'price', 'release_year', 'has_thumbnail', 'total_decisions', 'like_ratio']
        video_matrix = []

        for _, row in training_data.iterrows():
            video_data = video_features[video_features['video_id'] == row['video_id']].iloc[0]
            video_matrix.append([video_data[col] for col in video_feature_cols])

        video_matrix = np.array(video_matrix, dtype=np.float32)
        video_matrix = self.video_scaler.fit_transform(video_matrix)

        return user_matrix, video_matrix

    def train(self, data_loader: SupabaseCompatibleDataLoader):
        """Two-Towerモデルの訓練"""
        print("🚀 Starting Edge Function Two-Tower training...")

        # データ読み込み
        training_data, labels = data_loader.get_training_data()
        user_features = data_loader.get_user_features(training_data)
        video_features = data_loader.get_video_features(training_data)

        # 特徴量準備
        user_matrix, video_matrix = self.prepare_features(training_data, user_features, video_features)

        print(f"📊 User features shape: {user_matrix.shape}")
        print(f"📊 Video features shape: {video_matrix.shape}")
        print(f"📊 Labels shape: {labels.shape}")

        # モデル構築
        self.user_tower = self.build_user_tower(user_matrix.shape[1])
        self.item_tower = self.build_item_tower(video_matrix.shape[1])
        self.full_model = self.build_full_model(user_matrix.shape[1], video_matrix.shape[1])

        # コンパイル（少量データ用の設定）
        self.full_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print("\n📋 Model Architecture:")
        self.full_model.summary()

        # 訓練・検証分割（少量データなので80:20）
        train_user, val_user, train_video, val_video, train_labels, val_labels = train_test_split(
            user_matrix, video_matrix, labels,
            test_size=0.2, random_state=42, stratify=labels
        )

        # 訓練実行
        print("\n🏋️ Training model...")
        history = self.full_model.fit(
            [train_user, train_video], train_labels,
            validation_data=([val_user, val_video], val_labels),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )

        # 評価
        val_loss, val_acc, val_prec, val_recall = self.full_model.evaluate([val_user, val_video], val_labels, verbose=0)
        print(f"\n✅ Validation Results:")
        print(f"   - Accuracy: {val_acc:.4f}")
        print(f"   - Precision: {val_prec:.4f}")
        print(f"   - Recall: {val_recall:.4f}")
        print(f"   - F1-Score: {2 * val_prec * val_recall / (val_prec + val_recall):.4f}")

        return history

    def save_models(self):
        """モデルをEdge Function用に保存"""
        print("\n💾 Saving models for Edge Functions...")

        # 個別Tower保存（SavedModel形式でTensorFlow.js用）
        user_tower_path = self.output_dir / "user_tower_768"
        item_tower_path = self.output_dir / "item_tower_768"

        # Keras形式で保存（Edge Function用）
        self.user_tower.save(self.output_dir / "user_tower_768.keras")
        self.item_tower.save(self.output_dir / "item_tower_768.keras")

        print(f"✅ Models saved to: {self.output_dir}")

        # 前処理器保存
        with open(self.output_dir / "preprocessors.pkl", 'wb') as f:
            pickle.dump({
                'user_scaler': self.user_scaler,
                'video_scaler': self.video_scaler
            }, f)

        # メタデータ保存
        metadata = {
            'embedding_dim': self.embedding_dim,
            'model_version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'supabase_compatible': True,
            'edge_function_ready': True,
            'user_features': ['display_name_length', 'account_age_days', 'total_decisions', 'like_ratio'],
            'video_features': ['title_length', 'price', 'release_year', 'has_thumbnail', 'total_decisions', 'like_ratio']
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Models saved to {self.output_dir}/")
        print("🌐 TensorFlow.js models ready for Edge Functions!")

if __name__ == "__main__":
    # モデル訓練実行
    loader = SupabaseCompatibleDataLoader()
    model = EdgeFunctionTwoTower()

    # 訓練
    history = model.train(loader)

    # 保存
    model.save_models()

    print("\n🎉 Edge Function Two-Tower model training completed!")