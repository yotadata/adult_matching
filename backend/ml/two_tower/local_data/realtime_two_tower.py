#!/usr/bin/env python3
"""
リアルタイム型Two-Towerモデル
LIKE/NOPE履歴から動的にユーザー表現を生成
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

class RealtimeTwoTowerModel:
    """LIKE/NOPE履歴ベースのリアルタイム型Two-Towerモデル"""

    def __init__(self, num_videos=3742, video_embed_dim=64, final_embed_dim=768):
        self.num_videos = num_videos
        self.video_embed_dim = video_embed_dim
        self.final_embed_dim = final_embed_dim
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)

        # モデル
        self.video_embedding = None
        self.user_tower = None
        self.item_tower = None
        self.full_model = None

        # データローダー
        self.data_loader = SupabaseDataLoader()

    def build_video_embedding_layer(self):
        """動画埋め込み層を構築"""
        self.video_embedding = layers.Embedding(
            input_dim=self.num_videos + 1,  # +1 for unknown videos
            output_dim=self.video_embed_dim,
            mask_zero=True,
            name='video_embedding'
        )

    def build_user_representation_layer(self, max_history=50):
        """LIKE/NOPE履歴からユーザー表現を生成"""
        # LIKE履歴入力
        like_history_input = layers.Input(shape=(max_history,), name='like_history')
        like_embeds = self.video_embedding(like_history_input)
        like_mask = layers.Masking(mask_value=0.0)(like_embeds)
        like_pooled = layers.GlobalAveragePooling1D()(like_mask)

        # NOPE履歴入力
        nope_history_input = layers.Input(shape=(max_history,), name='nope_history')
        nope_embeds = self.video_embedding(nope_history_input)
        nope_mask = layers.Masking(mask_value=0.0)(nope_embeds)
        nope_pooled = layers.GlobalAveragePooling1D()(nope_mask)

        # 重み付き結合（LIKE: +1.0, NOPE: -0.5）
        weighted_user_repr = layers.Add()([
            layers.Lambda(lambda x: x * 1.0)(like_pooled),
            layers.Lambda(lambda x: x * -0.5)(nope_pooled)
        ])

        # ユーザータワー
        user_dense1 = layers.Dense(256, activation='relu', name='user_dense1')(weighted_user_repr)
        user_dropout1 = layers.Dropout(0.3)(user_dense1)
        user_dense2 = layers.Dense(128, activation='relu', name='user_dense2')(user_dropout1)
        user_dropout2 = layers.Dropout(0.2)(user_dense2)

        # 最終ユーザー埋め込み
        user_embedding = layers.Dense(self.final_embed_dim, name='user_embedding_dense')(user_dropout2)
        user_embedding_norm = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='user_embedding')(user_embedding)

        self.user_tower = models.Model(
            inputs=[like_history_input, nope_history_input],
            outputs=user_embedding_norm,
            name='realtime_user_tower'
        )

    def build_item_tower(self, item_features_dim=229):
        """アイテムタワー（動画特徴量 + 埋め込み）"""
        # 動画ID入力
        item_id_input = layers.Input(shape=(1,), name='item_id')
        item_embed = layers.Flatten()(self.video_embedding(item_id_input))

        # 動画特徴量入力
        item_features_input = layers.Input(shape=(item_features_dim,), name='item_features')

        # 動画埋め込み + 特徴量を結合
        combined_item = layers.Concatenate()([item_embed, item_features_input])

        # アイテムタワー
        item_dense1 = layers.Dense(512, activation='relu', name='item_dense1')(combined_item)
        item_dropout1 = layers.Dropout(0.4)(item_dense1)
        item_dense2 = layers.Dense(256, activation='relu', name='item_dense2')(item_dropout1)
        item_dropout2 = layers.Dropout(0.3)(item_dense2)
        item_dense3 = layers.Dense(128, activation='relu', name='item_dense3')(item_dropout2)
        item_dropout3 = layers.Dropout(0.2)(item_dense3)

        # 最終アイテム埋め込み
        item_embedding = layers.Dense(self.final_embed_dim, name='item_embedding_dense')(item_dense3)
        item_embedding_norm = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='item_embedding')(item_embedding)

        self.item_tower = models.Model(
            inputs=[item_id_input, item_features_input],
            outputs=item_embedding_norm,
            name='realtime_item_tower'
        )

    def build_full_model(self, max_history=50, item_features_dim=229):
        """完全なリアルタイムTwo-Towerモデル"""

        # 動画埋め込み層構築
        self.build_video_embedding_layer()

        # ユーザー・アイテムタワー構築
        self.build_user_representation_layer(max_history)
        self.build_item_tower(item_features_dim)

        # 入力
        like_history_input = layers.Input(shape=(max_history,), name='like_history')
        nope_history_input = layers.Input(shape=(max_history,), name='nope_history')
        item_id_input = layers.Input(shape=(1,), name='item_id')
        item_features_input = layers.Input(shape=(item_features_dim,), name='item_features')

        # 埋め込み計算
        user_embedding = self.user_tower([like_history_input, nope_history_input])
        item_embedding = self.item_tower([item_id_input, item_features_input])

        # コサイン類似度計算
        similarity = layers.Dot(axes=1, normalize=False, name='similarity')([user_embedding, item_embedding])

        # 最終予測
        prediction = layers.Dense(1, activation='sigmoid', name='prediction')(similarity)

        # モデル構築
        self.full_model = models.Model(
            inputs=[like_history_input, nope_history_input, item_id_input, item_features_input],
            outputs=prediction,
            name='realtime_two_tower_full'
        )

        return self.full_model

    def prepare_training_data(self, samples, max_history=50):
        """学習データを準備"""
        like_histories = []
        nope_histories = []
        item_ids = []
        item_features = []
        labels = []

        # 動画IDマッピング作成
        video_ids = self.data_loader.videos_df['id'].unique()
        video_id_to_idx = {vid: idx+1 for idx, vid in enumerate(video_ids)}  # 0は予約

        for sample in samples:
            # LIKE履歴をパディング
            like_hist = [video_id_to_idx.get(vid, 0) for vid in sample['user_like_history']]
            like_hist = like_hist[-max_history:] if len(like_hist) > max_history else like_hist
            like_hist += [0] * (max_history - len(like_hist))

            # NOPE履歴をパディング
            nope_hist = [video_id_to_idx.get(vid, 0) for vid in sample['user_nope_history']]
            nope_hist = nope_hist[-max_history:] if len(nope_hist) > max_history else nope_hist
            nope_hist += [0] * (max_history - len(nope_hist))

            # 動画ID
            item_id = video_id_to_idx.get(sample['video_id'], 0)

            # 動画特徴量
            item_feat = self.data_loader.get_single_video_features(sample['video_features'])

            like_histories.append(like_hist)
            nope_histories.append(nope_hist)
            item_ids.append([item_id])
            item_features.append(item_feat)
            labels.append(sample['label'])

        return {
            'like_histories': np.array(like_histories),
            'nope_histories': np.array(nope_histories),
            'item_ids': np.array(item_ids),
            'item_features': np.array(item_features),
            'labels': np.array(labels)
        }

    def train(self, data_loader: SupabaseDataLoader):
        """リアルタイム型モデル学習"""
        print("🚀 Starting Realtime Two-Tower training...")

        # データ読み込み
        data_loader.load_all_data()
        training_samples = data_loader.get_realtime_training_data()

        # 学習データ準備
        data = self.prepare_training_data(training_samples)

        print(f"📊 Realtime training features:")
        print(f"   Like histories shape: {data['like_histories'].shape}")
        print(f"   Nope histories shape: {data['nope_histories'].shape}")
        print(f"   Item IDs shape: {data['item_ids'].shape}")
        print(f"   Item features shape: {data['item_features'].shape}")
        print(f"   Labels shape: {data['labels'].shape}")

        # モデル構築
        self.full_model = self.build_full_model(
            max_history=50,
            item_features_dim=data['item_features'].shape[1]
        )

        print(f"\n📋 Realtime Model Architecture:")
        self.full_model.summary()

        # コンパイル
        self.full_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        # データ分割
        indices = np.arange(len(data['labels']))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=data['labels'])

        train_data = {
            'like_histories': data['like_histories'][train_idx],
            'nope_histories': data['nope_histories'][train_idx],
            'item_ids': data['item_ids'][train_idx],
            'item_features': data['item_features'][train_idx],
            'labels': data['labels'][train_idx]
        }

        val_data = {
            'like_histories': data['like_histories'][val_idx],
            'nope_histories': data['nope_histories'][val_idx],
            'item_ids': data['item_ids'][val_idx],
            'item_features': data['item_features'][val_idx],
            'labels': data['labels'][val_idx]
        }

        print(f"🏋️ Training realtime model...")
        print(f"   Training samples: {len(train_data['labels'])}")
        print(f"   Validation samples: {len(val_data['labels'])}")

        # 学習実行
        history = self.full_model.fit(
            x=[train_data['like_histories'], train_data['nope_histories'],
               train_data['item_ids'], train_data['item_features']],
            y=train_data['labels'],
            epochs=15,
            batch_size=32,
            validation_data=(
                [val_data['like_histories'], val_data['nope_histories'],
                 val_data['item_ids'], val_data['item_features']],
                val_data['labels']
            ),
            verbose=1
        )

        # 評価
        val_predictions = self.full_model.predict([
            val_data['like_histories'], val_data['nope_histories'],
            val_data['item_ids'], val_data['item_features']
        ])
        val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(val_data['labels'], val_predictions_binary)
        precision = precision_score(val_data['labels'], val_predictions_binary)
        recall = recall_score(val_data['labels'], val_predictions_binary)
        f1 = f1_score(val_data['labels'], val_predictions_binary)

        print(f"\n✅ Realtime Validation Results:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")

        return history

    def save_models(self):
        """リアルタイムモデル保存"""
        print("💾 Saving realtime models...")

        # Keras形式で保存
        self.user_tower.save(self.output_dir / "realtime_user_tower_768.keras")
        self.item_tower.save(self.output_dir / "realtime_item_tower_768.keras")
        self.full_model.save(self.output_dir / "realtime_full_model_768.keras")

        # 動画埋め込み層保存
        video_embedding_model = models.Model(
            inputs=layers.Input(shape=(1,)),
            outputs=self.video_embedding(layers.Input(shape=(1,))),
            name='video_embedding_model'
        )
        video_embedding_model.save(self.output_dir / "video_embedding_768.keras")

        # メタデータ保存
        metadata = {
            "model_type": "realtime_two_tower",
            "embedding_dim": self.final_embed_dim,
            "video_embed_dim": self.video_embed_dim,
            "num_videos": self.num_videos,
            "model_version": "3.0_realtime",
            "created_at": "2025-09-28T00:45:00.000000",
            "supports_like_nope": True,
            "realtime_capable": True,
            "features": {
                "like_history_max": 50,
                "nope_history_max": 50,
                "item_features_dim": 229
            }
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Realtime models saved to: {self.output_dir}")
        print(f"   📁 User tower: LIKE/NOPE履歴ベース")
        print(f"   📁 Item tower: 動画ID + 特徴量")
        print(f"   📁 Video embedding: {self.num_videos}動画 × {self.video_embed_dim}次元")

if __name__ == "__main__":
    print("🚀 Realtime Two-Tower training starting...")

    # リアルタイムTwo-Tower学習実行
    model = RealtimeTwoTowerModel()
    loader = SupabaseDataLoader()

    try:
        # 学習実行
        history = model.train(loader)

        # モデル保存
        if model.user_tower is not None and model.item_tower is not None:
            model.save_models()
            print(f"\n🎉 Realtime Two-Tower training completed!")
            print(f"🚀 Ready for real-time recommendations with LIKE/NOPE history!")
        else:
            print("❌ Models not trained properly")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()