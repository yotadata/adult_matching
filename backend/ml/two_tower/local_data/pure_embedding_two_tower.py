#!/usr/bin/env python3
"""
完全埋め込み学習版Two-Towerモデル
ユーザー・動画ともに手動特徴量なし、純粋な埋め込み学習
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
from pathlib import Path
from data_loader import SupabaseDataLoader

class PureEmbeddingTwoTowerModel:
    """完全埋め込み学習版Two-Towerモデル（手動特徴量なし）"""

    def __init__(self, num_videos=3742, embed_dim=256, final_embed_dim=768):
        self.num_videos = num_videos
        self.embed_dim = embed_dim  # 動画埋め込み次元（大きく設定）
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
        """大容量動画埋め込み層（重たいモデルOK）"""
        self.video_embedding = layers.Embedding(
            input_dim=self.num_videos + 1,  # +1 for unknown videos
            output_dim=self.embed_dim,      # 256次元（大きめ）
            embeddings_regularizer=regularizers.L2(0.001),  # 正則化
            mask_zero=True,
            name='video_embedding'
        )

    def build_user_tower(self, max_history=100):
        """LIKE/NOPE履歴からユーザー埋め込み（重たいモデル）"""
        # LIKE履歴入力（長い履歴に対応）
        like_history_input = layers.Input(shape=(max_history,), name='like_history')
        like_embeds = self.video_embedding(like_history_input)
        like_mask = layers.Masking(mask_value=0.0)(like_embeds)

        # LIKE履歴をLSTMで処理（順序も考慮）
        like_lstm = layers.LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(like_mask)
        like_attention = layers.GlobalAveragePooling1D()(like_lstm)

        # NOPE履歴入力
        nope_history_input = layers.Input(shape=(max_history,), name='nope_history')
        nope_embeds = self.video_embedding(nope_history_input)
        nope_mask = layers.Masking(mask_value=0.0)(nope_embeds)

        # NOPE履歴もLSTMで処理
        nope_lstm = layers.LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(nope_mask)
        nope_attention = layers.GlobalAveragePooling1D()(nope_lstm)

        # LIKE/NOPE重み付き結合（LIKE: +1.0, NOPE: -0.6）
        weighted_user_repr = layers.Add()([
            layers.Lambda(lambda x: x * 1.0)(like_attention),
            layers.Lambda(lambda x: x * -0.6)(nope_attention)
        ])

        # 深いユーザータワー（重たいモデルOK）
        user_dense1 = layers.Dense(1024, activation='relu',
                                  kernel_regularizer=regularizers.L2(0.001))(weighted_user_repr)
        user_dropout1 = layers.Dropout(0.4)(user_dense1)

        user_dense2 = layers.Dense(512, activation='relu',
                                  kernel_regularizer=regularizers.L2(0.001))(user_dropout1)
        user_dropout2 = layers.Dropout(0.3)(user_dense2)

        user_dense3 = layers.Dense(256, activation='relu',
                                  kernel_regularizer=regularizers.L2(0.001))(user_dropout2)
        user_dropout3 = layers.Dropout(0.2)(user_dense3)

        # 最終ユーザー埋め込み
        user_embedding = layers.Dense(self.final_embed_dim, name='user_embedding_dense')(user_dropout3)
        user_embedding_norm = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                                           name='user_embedding')(user_embedding)

        self.user_tower = models.Model(
            inputs=[like_history_input, nope_history_input],
            outputs=user_embedding_norm,
            name='pure_embedding_user_tower'
        )

    def build_item_tower(self):
        """純粋動画埋め込みベースのアイテムタワー（重たいモデル）"""
        # 動画ID入力のみ（特徴量なし）
        item_id_input = layers.Input(shape=(1,), name='item_id')
        item_embed = layers.Flatten()(self.video_embedding(item_id_input))

        # 深いアイテムタワー（動画埋め込みから豊富な表現を学習）
        item_dense1 = layers.Dense(1024, activation='relu',
                                  kernel_regularizer=regularizers.L2(0.001))(item_embed)
        item_dropout1 = layers.Dropout(0.4)(item_dense1)

        item_dense2 = layers.Dense(512, activation='relu',
                                  kernel_regularizer=regularizers.L2(0.001))(item_dropout1)
        item_dropout2 = layers.Dropout(0.3)(item_dense2)

        item_dense3 = layers.Dense(256, activation='relu',
                                  kernel_regularizer=regularizers.L2(0.001))(item_dropout2)
        item_dropout3 = layers.Dropout(0.2)(item_dense3)

        item_dense4 = layers.Dense(128, activation='relu',
                                  kernel_regularizer=regularizers.L2(0.001))(item_dropout3)
        item_dropout4 = layers.Dropout(0.1)(item_dense4)

        # 最終アイテム埋め込み
        item_embedding = layers.Dense(self.final_embed_dim, name='item_embedding_dense')(item_dropout4)
        item_embedding_norm = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                                           name='item_embedding')(item_embedding)

        self.item_tower = models.Model(
            inputs=item_id_input,
            outputs=item_embedding_norm,
            name='pure_embedding_item_tower'
        )

    def build_full_model(self, max_history=100):
        """完全埋め込み学習Two-Towerモデル"""

        # 動画埋め込み層構築
        self.build_video_embedding_layer()

        # ユーザー・アイテムタワー構築
        self.build_user_tower(max_history)
        self.build_item_tower()

        # 入力（動画特徴量なし）
        like_history_input = layers.Input(shape=(max_history,), name='like_history')
        nope_history_input = layers.Input(shape=(max_history,), name='nope_history')
        item_id_input = layers.Input(shape=(1,), name='item_id')

        # 埋め込み計算
        user_embedding = self.user_tower([like_history_input, nope_history_input])
        item_embedding = self.item_tower(item_id_input)

        # コサイン類似度計算
        similarity = layers.Dot(axes=1, normalize=False, name='similarity')([user_embedding, item_embedding])

        # 最終予測（シンプルなシグモイド）
        prediction = layers.Dense(1, activation='sigmoid', name='prediction')(similarity)

        # モデル構築
        self.full_model = models.Model(
            inputs=[like_history_input, nope_history_input, item_id_input],
            outputs=prediction,
            name='pure_embedding_two_tower_full'
        )

        return self.full_model

    def prepare_training_data(self, samples, data_loader, max_history=100):
        """完全埋め込み学習用データ準備（動画特徴量なし）"""
        like_histories = []
        nope_histories = []
        item_ids = []
        labels = []

        # 動画IDマッピング作成
        video_ids = data_loader.videos_df['id'].unique()
        video_id_to_idx = {vid: idx+1 for idx, vid in enumerate(video_ids)}  # 0は予約

        print(f"📊 Video ID mapping: {len(video_id_to_idx)} videos")

        for sample in samples:
            # LIKE履歴を長めに対応（100件まで）
            like_hist = [video_id_to_idx.get(vid, 0) for vid in sample['user_like_history']]
            like_hist = like_hist[-max_history:] if len(like_hist) > max_history else like_hist
            like_hist += [0] * (max_history - len(like_hist))

            # NOPE履歴を長めに対応
            nope_hist = [video_id_to_idx.get(vid, 0) for vid in sample['user_nope_history']]
            nope_hist = nope_hist[-max_history:] if len(nope_hist) > max_history else nope_hist
            nope_hist += [0] * (max_history - len(nope_hist))

            # 動画IDのみ（特徴量なし）
            item_id = video_id_to_idx.get(sample['video_id'], 0)

            like_histories.append(like_hist)
            nope_histories.append(nope_hist)
            item_ids.append([item_id])
            labels.append(sample['label'])

        return {
            'like_histories': np.array(like_histories),
            'nope_histories': np.array(nope_histories),
            'item_ids': np.array(item_ids),
            'labels': np.array(labels),
            'video_id_mapping': video_id_to_idx
        }

    def train(self, data_loader: SupabaseDataLoader):
        """完全埋め込み学習版モデル学習"""
        print("🚀 Starting Pure Embedding Two-Tower training...")

        # データ読み込み
        data_loader.load_all_data()
        training_samples = data_loader.get_realtime_training_data()

        # 学習データ準備（動画特徴量なし）
        data = self.prepare_training_data(training_samples, data_loader, max_history=100)

        print(f"📊 Pure Embedding training features:")
        print(f"   Like histories shape: {data['like_histories'].shape}")
        print(f"   Nope histories shape: {data['nope_histories'].shape}")
        print(f"   Item IDs shape: {data['item_ids'].shape}")
        print(f"   Labels shape: {data['labels'].shape}")
        print(f"   No manual features! Pure embedding learning.")

        # モデル構築（重たいモデル）
        self.full_model = self.build_full_model(max_history=100)

        print(f"\n📋 Pure Embedding Model Architecture:")
        self.full_model.summary()

        # パラメータ数表示
        total_params = self.full_model.count_params()
        print(f"🧠 Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

        # コンパイル
        self.full_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),  # 少し小さめ学習率
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
            'labels': data['labels'][train_idx]
        }

        val_data = {
            'like_histories': data['like_histories'][val_idx],
            'nope_histories': data['nope_histories'][val_idx],
            'item_ids': data['item_ids'][val_idx],
            'labels': data['labels'][val_idx]
        }

        print(f"🏋️ Training pure embedding model...")
        print(f"   Training samples: {len(train_data['labels'])}")
        print(f"   Validation samples: {len(val_data['labels'])}")

        # 学習実行（重いモデルなのでエポック数調整）
        history = self.full_model.fit(
            x=[train_data['like_histories'], train_data['nope_histories'], train_data['item_ids']],
            y=train_data['labels'],
            epochs=30,  # 早期停止があるので多めに設定
            batch_size=16,  # バッチサイズ小さめ
            validation_data=(
                [val_data['like_histories'], val_data['nope_histories'], val_data['item_ids']],
                val_data['labels']
            ),
            verbose=1,
            callbacks=[
                ModelCheckpoint(
                    filepath=str(self.output_dir / "pure_embedding_best.keras"),
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    mode='max',
                    verbose=1,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        )

        # 評価
        val_predictions = self.full_model.predict([
            val_data['like_histories'], val_data['nope_histories'], val_data['item_ids']
        ])
        val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(val_data['labels'], val_predictions_binary)
        precision = precision_score(val_data['labels'], val_predictions_binary)
        recall = recall_score(val_data['labels'], val_predictions_binary)
        f1 = f1_score(val_data['labels'], val_predictions_binary)

        print(f"\n✅ Pure Embedding Validation Results:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")

        # 動画IDマッピング保存
        self.video_id_mapping = data['video_id_mapping']

        return history

    def save_models(self):
        """完全埋め込み学習モデル保存"""
        print("💾 Saving pure embedding models...")

        # Keras形式で保存
        self.user_tower.save(self.output_dir / "user_tower_768.keras")
        self.item_tower.save(self.output_dir / "item_tower_768.keras")
        self.full_model.save(self.output_dir / "full_model_768.keras")

        # 動画埋め込み層を単独で保存
        video_embedding_input = layers.Input(shape=(1,), name='video_input')
        video_embedding_output = layers.Flatten()(self.video_embedding(video_embedding_input))
        video_embedding_model = models.Model(
            inputs=video_embedding_input,
            outputs=video_embedding_output,
            name='video_embedding_model'
        )
        video_embedding_model.save(self.output_dir / "video_embedding_768.keras")

        # 動画IDマッピング保存
        with open(self.output_dir / "video_id_mapping.pkl", 'wb') as f:
            pickle.dump(self.video_id_mapping, f)

        # メタデータ保存
        metadata = {
            "model_type": "pure_embedding_two_tower",
            "embedding_dim": self.final_embed_dim,
            "video_embed_dim": self.embed_dim,
            "num_videos": self.num_videos,
            "model_version": "4.0_pure_embedding",
            "created_at": "2025-09-28T01:00:00.000000",
            "features": {
                "manual_features": False,
                "pure_embedding": True,
                "like_history_max": 100,
                "nope_history_max": 100,
                "lstm_enabled": True,
                "attention_enabled": True
            },
            "performance": {
                "heavy_model": True,
                "backend_optimized": True,
                "realtime_inference": True
            }
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Pure embedding models saved to: {self.output_dir}")
        print(f"   📁 User tower: LIKE/NOPE履歴 → LSTM → 深層NN")
        print(f"   📁 Item tower: 動画ID → 大容量埋め込み → 深層NN")
        print(f"   📁 Video embedding: {self.num_videos}動画 × {self.embed_dim}次元")
        print(f"   📁 No manual features - Pure embedding learning!")

if __name__ == "__main__":
    print("🚀 Pure Embedding Two-Tower training starting...")

    # 完全埋め込み学習Two-Tower実行
    model = PureEmbeddingTwoTowerModel(
        num_videos=3742,
        embed_dim=256,      # 大容量埋め込み
        final_embed_dim=768
    )
    loader = SupabaseDataLoader()

    try:
        # 学習実行
        history = model.train(loader)

        # モデル保存
        if model.user_tower is not None and model.item_tower is not None:
            model.save_models()
            print(f"\n🎉 Pure Embedding Two-Tower training completed!")
            print(f"🚀 Ready for pure embedding-based recommendations!")
            print(f"💪 Heavy model optimized for backend processing!")
        else:
            print("❌ Models not trained properly")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()