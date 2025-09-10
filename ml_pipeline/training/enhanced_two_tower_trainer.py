"""
強化された768次元Two-Tower訓練システム

PostgreSQLのvector(768)テーブルとの完全互換性を実現し、
リアルユーザーデータと疑似ユーザーデータの統合訓練を支援する
高性能Two-Towerモデル訓練器
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pickle

# 既存の特徴処理器をインポート
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.real_user_feature_extractor import RealUserFeatureExtractor
from preprocessing.enhanced_item_feature_processor import EnhancedItemFeatureProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfiguration:
    """訓練設定"""
    embedding_dim: int = 768
    user_tower_hidden_dims: List[int] = None
    item_tower_hidden_dims: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    dropout_rate: float = 0.3
    l2_reg: float = 1e-4
    negative_sampling_ratio: int = 2

@dataclass
class ModelArchitecture:
    """モデルアーキテクチャ設定"""
    user_tower_layers: List[int] = None
    item_tower_layers: List[int] = None
    activation: str = 'relu'
    final_activation: str = 'tanh'
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    
    def __post_init__(self):
        if self.user_tower_layers is None:
            self.user_tower_layers = [1024, 512, 256, 768]
        if self.item_tower_layers is None:
            self.item_tower_layers = [1024, 512, 256, 768]

class EnhancedTwoTowerTrainer:
    """強化されたTwo-Tower訓練器"""
    
    def __init__(self, 
                 db_connection_string: str,
                 config: TrainingConfiguration = None,
                 architecture: ModelArchitecture = None):
        """
        初期化
        
        Args:
            db_connection_string: PostgreSQL接続文字列
            config: 訓練設定
            architecture: モデルアーキテクチャ設定
        """
        self.db_connection_string = db_connection_string
        self.config = config or TrainingConfiguration()
        self.architecture = architecture or ModelArchitecture()
        
        # 特徴処理器
        self.user_feature_extractor = RealUserFeatureExtractor(db_connection_string)
        self.item_feature_processor = EnhancedItemFeatureProcessor(db_connection_string)
        
        # モデル
        self.user_tower = None
        self.item_tower = None
        self.full_model = None
        
        # 訓練履歴と統計
        self.training_history = {}
        self.training_stats = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'architecture': self.architecture.__dict__,
            'data_stats': {},
            'model_stats': {},
            'evaluation_metrics': {},
            'training_time': 0
        }
        
        # GPU設定
        self._setup_gpu()

    def _setup_gpu(self):
        """GPU設定の最適化"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU設定完了: {len(gpus)} GPU検出")
            else:
                logger.info("CPU環境で実行")
        except Exception as e:
            logger.warning(f"GPU設定警告: {e}")

    def connect_db(self) -> psycopg2.extensions.connection:
        """データベース接続"""
        try:
            conn = psycopg2.connect(
                self.db_connection_string,
                cursor_factory=RealDictCursor
            )
            logger.info("PostgreSQL接続成功")
            return conn
        except Exception as e:
            logger.error(f"データベース接続エラー: {e}")
            raise

    def prepare_training_data(self, 
                            user_sample_size: int = 1000,
                            video_sample_size: int = 10000,
                            include_pseudo_users: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """訓練データの準備"""
        logger.info("訓練データ準備開始...")
        
        start_time = datetime.now()
        
        with self.connect_db() as conn:
            # ユーザー特徴の抽出
            logger.info("ユーザー特徴抽出...")
            
            # アクティブユーザーのサンプリング
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT user_id 
                FROM user_video_decisions 
                WHERE created_at >= %s
                ORDER BY user_id
                LIMIT %s
            """, (datetime.now() - timedelta(days=180), user_sample_size))
            
            user_ids = [row['user_id'] for row in cursor.fetchall()]
            logger.info(f"対象ユーザー数: {len(user_ids)}")
            
            # 語彙構築
            self.user_feature_extractor.build_vocabulary(conn)
            
            # ユーザー特徴抽出
            user_feature_vectors, user_stats = self.user_feature_extractor.extract_batch_features(user_ids, conn)
            
            # 疑似ユーザーとの統合（オプション）
            if include_pseudo_users:
                try:
                    pseudo_user_file = "data/processed/rating_based_pseudo_users.json"
                    if os.path.exists(pseudo_user_file):
                        user_features, integrated_user_ids = self.user_feature_extractor.integrate_with_pseudo_users(
                            user_feature_vectors, pseudo_user_file
                        )
                        logger.info(f"疑似ユーザー統合完了: 総ユーザー数 {len(integrated_user_ids)}")
                    else:
                        user_features, integrated_user_ids = self.user_feature_extractor.convert_to_numpy_array(user_feature_vectors)
                        logger.info("疑似ユーザーファイル未検出、リアルユーザーのみ使用")
                except Exception as e:
                    logger.warning(f"疑似ユーザー統合エラー: {e}")
                    user_features, integrated_user_ids = self.user_feature_extractor.convert_to_numpy_array(user_feature_vectors)
            else:
                user_features, integrated_user_ids = self.user_feature_extractor.convert_to_numpy_array(user_feature_vectors)
            
            # ビデオ特徴の抽出
            logger.info("ビデオ特徴抽出...")
            videos = self.item_feature_processor.load_video_metadata(conn, limit=video_sample_size)
            video_feature_vectors = self.item_feature_processor.convert_to_feature_vectors(videos, conn)
            video_features, video_ids = self.item_feature_processor.convert_to_numpy_array(video_feature_vectors)
            
            logger.info(f"ビデオ特徴: {video_features.shape}")
            
            # インタラクションデータの生成
            logger.info("インタラクションデータ生成...")
            user_indices, video_indices, labels = self._create_training_interactions(
                integrated_user_ids, video_ids, conn
            )
        
        # データ統計の更新
        preparation_time = (datetime.now() - start_time).total_seconds()
        self.training_stats['data_stats'] = {
            'user_count': len(integrated_user_ids),
            'video_count': len(video_ids),
            'interaction_count': len(labels),
            'positive_ratio': np.mean(labels),
            'user_feature_dim': user_features.shape[1],
            'video_feature_dim': video_features.shape[1],
            'preparation_time': preparation_time
        }
        
        logger.info(f"訓練データ準備完了: {preparation_time:.2f}秒")
        logger.info(f"ユーザー特徴: {user_features.shape}, ビデオ特徴: {video_features.shape}")
        logger.info(f"インタラクション: {len(labels)} 件 (正例率: {np.mean(labels):.3f})")
        
        return user_features, video_features, (user_indices, video_indices, labels)

    def _create_training_interactions(self, 
                                    user_ids: List[str], 
                                    video_ids: List[str],
                                    conn: psycopg2.extensions.connection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """訓練用インタラクションデータの生成"""
        
        # ユーザー・ビデオIDのマッピング
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        video_id_to_idx = {vid: idx for idx, vid in enumerate(video_ids)}
        
        cursor = conn.cursor()
        
        # 既存の正例インタラクション
        cursor.execute("""
            SELECT user_id, video_id, decision_type
            FROM user_video_decisions
            WHERE user_id = ANY(%s) AND video_id = ANY(%s)
            AND created_at >= %s
        """, (user_ids, video_ids, datetime.now() - timedelta(days=180)))
        
        interactions = cursor.fetchall()
        logger.info(f"既存インタラクション: {len(interactions)} 件")
        
        positive_pairs = []
        negative_pairs = []
        
        # 正例の収集
        for interaction in interactions:
            user_id = interaction['user_id']
            video_id = str(interaction['video_id'])
            decision_type = interaction['decision_type']
            
            if user_id in user_id_to_idx and video_id in video_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                video_idx = video_id_to_idx[video_id]
                
                if decision_type == 'like':
                    positive_pairs.append((user_idx, video_idx, 1))
                elif decision_type == 'nope':
                    negative_pairs.append((user_idx, video_idx, 0))
        
        logger.info(f"正例: {len([p for p in positive_pairs if p[2] == 1])} 件")
        logger.info(f"負例（既存）: {len([p for p in negative_pairs if p[2] == 0])} 件")
        
        # 追加の負例サンプリング
        target_negative_count = len(positive_pairs) * self.config.negative_sampling_ratio
        additional_negatives_needed = max(0, target_negative_count - len(negative_pairs))
        
        if additional_negatives_needed > 0:
            logger.info(f"追加負例サンプリング: {additional_negatives_needed} 件")
            
            # ユーザーごとに未インタラクションビデオをサンプル
            user_interactions = {}
            for user_idx, video_idx, label in positive_pairs + negative_pairs:
                if user_idx not in user_interactions:
                    user_interactions[user_idx] = set()
                user_interactions[user_idx].add(video_idx)
            
            additional_negatives = []
            for user_idx in range(len(user_ids)):
                interacted_videos = user_interactions.get(user_idx, set())
                available_videos = set(range(len(video_ids))) - interacted_videos
                
                if len(available_videos) > 0:
                    # このユーザーの負例数を決定
                    user_positive_count = len([p for p in positive_pairs if p[0] == user_idx])
                    user_negative_needed = user_positive_count * self.config.negative_sampling_ratio
                    user_current_negatives = len([p for p in negative_pairs if p[0] == user_idx])
                    user_additional_needed = max(0, user_negative_needed - user_current_negatives)
                    
                    if user_additional_needed > 0 and len(additional_negatives) < additional_negatives_needed:
                        sample_size = min(user_additional_needed, len(available_videos), 
                                        additional_negatives_needed - len(additional_negatives))
                        sampled_videos = np.random.choice(
                            list(available_videos), size=sample_size, replace=False
                        )
                        
                        for video_idx in sampled_videos:
                            additional_negatives.append((user_idx, video_idx, 0))
            
            negative_pairs.extend(additional_negatives)
            logger.info(f"負例追加完了: {len(additional_negatives)} 件")
        
        # 全インタラクションの結合
        all_interactions = positive_pairs + negative_pairs
        np.random.shuffle(all_interactions)
        
        user_indices = np.array([inter[0] for inter in all_interactions])
        video_indices = np.array([inter[1] for inter in all_interactions])
        labels = np.array([inter[2] for inter in all_interactions], dtype=np.float32)
        
        logger.info(f"最終インタラクション: {len(all_interactions)} 件 (正例率: {np.mean(labels):.3f})")
        
        return user_indices, video_indices, labels

    def build_user_tower(self, input_dim: int) -> tf.keras.Model:
        """ユーザータワーの構築"""
        logger.info(f"ユーザータワー構築: 入力次元 {input_dim} → {self.config.embedding_dim}")
        
        # 入力層
        user_input = layers.Input(shape=(input_dim,), name='user_input')
        
        # 正規化層
        x = layers.LayerNormalization(name='user_input_norm')(user_input) if self.architecture.use_layer_norm else user_input
        
        # 隠れ層の構築
        for i, units in enumerate(self.architecture.user_tower_layers[:-1]):  # 最後の層（埋め込み層）を除く
            x = layers.Dense(
                units, 
                activation=self.architecture.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
                name=f'user_dense_{i+1}'
            )(x)
            
            # 正規化
            if self.architecture.use_batch_norm:
                x = layers.BatchNormalization(name=f'user_bn_{i+1}')(x)
            elif self.architecture.use_layer_norm:
                x = layers.LayerNormalization(name=f'user_ln_{i+1}')(x)
            
            # ドロップアウト
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f'user_dropout_{i+1}')(x)
        
        # 埋め込み層（768次元）
        user_embedding = layers.Dense(
            self.config.embedding_dim,
            activation=self.architecture.final_activation,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name='user_embedding'
        )(x)
        
        # L2正規化（コサイン類似度のため）
        user_normalized = tf.nn.l2_normalize(user_embedding, axis=1, name='user_normalized')
        
        # モデルの構築
        user_tower = models.Model(
            inputs=user_input, 
            outputs=user_normalized, 
            name='user_tower'
        )
        
        return user_tower

    def build_item_tower(self, input_dim: int) -> tf.keras.Model:
        """アイテムタワーの構築（ユーザータワーと対称）"""
        logger.info(f"アイテムタワー構築: 入力次元 {input_dim} → {self.config.embedding_dim}")
        
        # 入力層
        item_input = layers.Input(shape=(input_dim,), name='item_input')
        
        # 正規化層
        x = layers.LayerNormalization(name='item_input_norm')(item_input) if self.architecture.use_layer_norm else item_input
        
        # 隠れ層の構築（ユーザータワーと同じ構造）
        for i, units in enumerate(self.architecture.item_tower_layers[:-1]):  # 最後の層を除く
            x = layers.Dense(
                units, 
                activation=self.architecture.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
                name=f'item_dense_{i+1}'
            )(x)
            
            # 正規化
            if self.architecture.use_batch_norm:
                x = layers.BatchNormalization(name=f'item_bn_{i+1}')(x)
            elif self.architecture.use_layer_norm:
                x = layers.LayerNormalization(name=f'item_ln_{i+1}')(x)
            
            # ドロップアウト
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f'item_dropout_{i+1}')(x)
        
        # 埋め込み層（768次元）
        item_embedding = layers.Dense(
            self.config.embedding_dim,
            activation=self.architecture.final_activation,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name='item_embedding'
        )(x)
        
        # L2正規化
        item_normalized = tf.nn.l2_normalize(item_embedding, axis=1, name='item_normalized')
        
        # モデルの構築
        item_tower = models.Model(
            inputs=item_input, 
            outputs=item_normalized, 
            name='item_tower'
        )
        
        return item_tower

    def build_two_tower_model(self, 
                            user_feature_dim: int, 
                            item_feature_dim: int) -> tf.keras.Model:
        """Two-Towerモデルの構築"""
        logger.info("Two-Towerモデル構築開始...")
        
        # 個別タワーの構築
        self.user_tower = self.build_user_tower(user_feature_dim)
        self.item_tower = self.build_item_tower(item_feature_dim)
        
        # 統合モデル（訓練用）
        user_input = layers.Input(shape=(user_feature_dim,), name='user_input')
        item_input = layers.Input(shape=(item_feature_dim,), name='item_input')
        
        # 各タワーでの埋め込み生成
        user_emb = self.user_tower(user_input)
        item_emb = self.item_tower(item_input)
        
        # コサイン類似度の計算
        similarity = layers.Dot(axes=1, normalize=False, name='cosine_similarity')([user_emb, item_emb])
        
        # シグモイド活性化で確率に変換
        output = layers.Activation('sigmoid', name='probability')(similarity)
        
        # フルモデルの構築
        full_model = models.Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='two_tower_model'
        )
        
        # モデルのコンパイル
        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0  # 勾配クリッピング
        )
        
        full_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc_roc'),
                tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        self.full_model = full_model
        
        # モデル統計の記録
        self.training_stats['model_stats'] = {
            'user_tower_params': self.user_tower.count_params(),
            'item_tower_params': self.item_tower.count_params(),
            'total_params': full_model.count_params(),
            'user_tower_layers': len(self.user_tower.layers),
            'item_tower_layers': len(self.item_tower.layers),
            'embedding_dimension': self.config.embedding_dim
        }
        
        # モデル構造の表示
        logger.info("=== ユーザータワー構造 ===")
        self.user_tower.summary(print_fn=logger.info)
        logger.info("=== アイテムタワー構造 ===")
        self.item_tower.summary(print_fn=logger.info)
        logger.info("=== フルモデル構造 ===")
        full_model.summary(print_fn=logger.info)
        
        return full_model

    def train_model(self, 
                   user_features: np.ndarray, 
                   video_features: np.ndarray,
                   interaction_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict:
        """モデル訓練の実行"""
        logger.info("モデル訓練開始...")
        
        start_time = datetime.now()
        
        user_indices, video_indices, labels = interaction_data
        
        # 訓練用特徴の準備
        X_user = user_features[user_indices]
        X_video = video_features[video_indices]
        y = labels
        
        logger.info(f"訓練データ形状: User {X_user.shape}, Video {X_video.shape}, Labels {y.shape}")
        
        # 訓練・検証分割
        train_user, val_user, train_video, val_video, train_y, val_y = train_test_split(
            X_user, X_video, y, 
            test_size=self.config.validation_split,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"訓練セット: {len(train_y)} 件, 検証セット: {len(val_y)} 件")
        
        # コールバックの設定
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_auc_pr',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='models/best_two_tower_model.h5',
                monitor='val_auc_pr',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # 訓練の実行
        history = self.full_model.fit(
            [train_user, train_video], train_y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=([val_user, val_video], val_y),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 訓練時間の記録
        training_time = (datetime.now() - start_time).total_seconds()
        self.training_stats['training_time'] = training_time
        
        # 訓練履歴の保存
        self.training_history = history.history
        
        # 最終評価
        final_metrics = self._evaluate_model(val_user, val_video, val_y)
        self.training_stats['evaluation_metrics'] = final_metrics
        
        logger.info(f"モデル訓練完了: {training_time:.2f}秒")
        logger.info("=== 最終評価結果 ===")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return history.history

    def _evaluate_model(self, 
                       val_user: np.ndarray, 
                       val_video: np.ndarray, 
                       val_y: np.ndarray) -> Dict[str, float]:
        """モデル評価の実行"""
        
        # 予測の実行
        predictions = self.full_model.predict([val_user, val_video], batch_size=self.config.batch_size)
        predictions = predictions.flatten()
        
        # メトリクスの計算
        auc_roc = roc_auc_score(val_y, predictions)
        precision, recall, _ = precision_recall_curve(val_y, predictions)
        auc_pr = auc(recall, precision)
        
        # 閾値0.5での分類性能
        binary_preds = (predictions > 0.5).astype(int)
        accuracy = np.mean(val_y == binary_preds)
        
        # True Positive, False Positive等の計算
        tp = np.sum((val_y == 1) & (binary_preds == 1))
        fp = np.sum((val_y == 0) & (binary_preds == 1))
        tn = np.sum((val_y == 0) & (binary_preds == 0))
        fn = np.sum((val_y == 1) & (binary_preds == 0))
        
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0.0
        
        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'accuracy': accuracy,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    def generate_embeddings(self, 
                          user_features: np.ndarray, 
                          video_features: np.ndarray,
                          user_ids: List[str],
                          video_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """埋め込みベクターの生成"""
        logger.info("埋め込み生成開始...")
        
        # バッチサイズでの推論
        batch_size = self.config.batch_size * 2  # 推論時は大きめのバッチサイズ
        
        # ユーザー埋め込みの生成
        user_embeddings = []
        for i in range(0, len(user_features), batch_size):
            batch_end = min(i + batch_size, len(user_features))
            batch_embeddings = self.user_tower.predict(
                user_features[i:batch_end], 
                batch_size=batch_size,
                verbose=0
            )
            user_embeddings.append(batch_embeddings)
        
        user_embeddings = np.vstack(user_embeddings)
        
        # ビデオ埋め込みの生成
        video_embeddings = []
        for i in range(0, len(video_features), batch_size):
            batch_end = min(i + batch_size, len(video_features))
            batch_embeddings = self.item_tower.predict(
                video_features[i:batch_end], 
                batch_size=batch_size,
                verbose=0
            )
            video_embeddings.append(batch_embeddings)
        
        video_embeddings = np.vstack(video_embeddings)
        
        logger.info(f"埋め込み生成完了: User {user_embeddings.shape}, Video {video_embeddings.shape}")
        
        # 正規化の確認
        user_norms = np.linalg.norm(user_embeddings, axis=1)
        video_norms = np.linalg.norm(video_embeddings, axis=1)
        logger.info(f"埋め込み正規化確認 - User norm: {user_norms.mean():.4f}±{user_norms.std():.4f}")
        logger.info(f"埋め込み正規化確認 - Video norm: {video_norms.mean():.4f}±{video_norms.std():.4f}")
        
        return user_embeddings, video_embeddings

    def save_models_and_artifacts(self, output_dir: str = "models/enhanced_two_tower"):
        """モデルとアーティファクトの保存"""
        logger.info(f"モデル保存開始: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # TensorFlowモデルの保存
        self.user_tower.save(f"{output_dir}/user_tower")
        self.item_tower.save(f"{output_dir}/item_tower")
        self.full_model.save(f"{output_dir}/full_model")
        
        # 軽量形式での保存（推論用）
        self.user_tower.save_weights(f"{output_dir}/user_tower_weights.h5")
        self.item_tower.save_weights(f"{output_dir}/item_tower_weights.h5")
        
        # モデル構造の保存
        with open(f"{output_dir}/user_tower_architecture.json", 'w') as f:
            f.write(self.user_tower.to_json())
        
        with open(f"{output_dir}/item_tower_architecture.json", 'w') as f:
            f.write(self.item_tower.to_json())
        
        # 設定・統計情報の保存
        with open(f"{output_dir}/training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        with open(f"{output_dir}/model_architecture.json", 'w') as f:
            json.dump(self.architecture.__dict__, f, indent=2)
        
        with open(f"{output_dir}/training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        with open(f"{output_dir}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 前処理器の保存
        preprocessing_artifacts = {
            'user_feature_extractor_stats': self.user_feature_extractor.extraction_stats,
            'item_feature_processor_stats': self.item_feature_processor.processing_stats,
            'user_genre_vocab': self.user_feature_extractor.genre_to_idx,
            'user_maker_vocab': self.user_feature_extractor.maker_to_idx,
        }
        
        with open(f"{output_dir}/preprocessing_artifacts.pkl", 'wb') as f:
            pickle.dump(preprocessing_artifacts, f)
        
        # モデル図の生成（可能な場合）
        try:
            plot_model(self.full_model, to_file=f"{output_dir}/model_architecture.png", 
                      show_shapes=True, show_layer_names=True)
            plot_model(self.user_tower, to_file=f"{output_dir}/user_tower.png", 
                      show_shapes=True, show_layer_names=True)
            plot_model(self.item_tower, to_file=f"{output_dir}/item_tower.png", 
                      show_shapes=True, show_layer_names=True)
        except Exception as e:
            logger.warning(f"モデル図生成エラー: {e}")
        
        logger.info(f"モデル保存完了: {output_dir}")

    def update_database_embeddings(self, 
                                 user_embeddings: np.ndarray,
                                 video_embeddings: np.ndarray,
                                 user_ids: List[str],
                                 video_ids: List[str]):
        """データベースの埋め込みテーブル更新"""
        logger.info("データベース埋め込み更新開始...")
        
        with self.connect_db() as conn:
            cursor = conn.cursor()
            
            # ユーザー埋め込みの更新
            logger.info(f"ユーザー埋め込み更新: {len(user_ids)} 件")
            for i, user_id in enumerate(user_ids):
                embedding_list = user_embeddings[i].tolist()
                cursor.execute("""
                    INSERT INTO user_embeddings (user_id, embedding, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        updated_at = EXCLUDED.updated_at
                """, (user_id, embedding_list, datetime.now()))
            
            # ビデオ埋め込みの更新
            logger.info(f"ビデオ埋め込み更新: {len(video_ids)} 件")
            for i, video_id in enumerate(video_ids):
                embedding_list = video_embeddings[i].tolist()
                cursor.execute("""
                    INSERT INTO video_embeddings (video_id, embedding, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (video_id)
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        updated_at = EXCLUDED.updated_at
                """, (video_id, embedding_list, datetime.now()))
            
            conn.commit()
        
        logger.info("データベース埋め込み更新完了")

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Two-Tower Model Training')
    parser.add_argument('--db-url', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--output-dir', default='models/enhanced_two_tower', help='Output directory')
    parser.add_argument('--user-samples', type=int, default=1000, help='Number of user samples')
    parser.add_argument('--video-samples', type=int, default=10000, help='Number of video samples')
    parser.add_argument('--embedding-dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--update-db', action='store_true', help='Update database embeddings')
    parser.add_argument('--include-pseudo', action='store_true', help='Include pseudo users')
    
    args = parser.parse_args()
    
    # 設定の作成
    config = TrainingConfiguration(
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # 訓練器の初期化
    trainer = EnhancedTwoTowerTrainer(args.db_url, config)
    
    try:
        logger.info("Enhanced Two-Tower Training 開始")
        
        # 訓練データの準備
        user_features, video_features, interaction_data = trainer.prepare_training_data(
            user_sample_size=args.user_samples,
            video_sample_size=args.video_samples,
            include_pseudo_users=args.include_pseudo
        )
        
        # モデルの構築
        trainer.build_two_tower_model(user_features.shape[1], video_features.shape[1])
        
        # 訓練の実行
        history = trainer.train_model(user_features, video_features, interaction_data)
        
        # 埋め込みの生成
        user_ids = list(range(len(user_features)))  # 実際のユーザーIDリストに置き換える
        video_ids = list(range(len(video_features)))  # 実際のビデオIDリストに置き換える
        
        user_embeddings, video_embeddings = trainer.generate_embeddings(
            user_features, video_features, user_ids, video_ids
        )
        
        # モデルの保存
        trainer.save_models_and_artifacts(args.output_dir)
        
        # データベース更新（オプション）
        if args.update_db:
            trainer.update_database_embeddings(
                user_embeddings, video_embeddings, user_ids, video_ids
            )
        
        # 最終結果の表示
        logger.info("=== 訓練完了 ===")
        logger.info(f"最終AUC-ROC: {trainer.training_stats['evaluation_metrics']['auc_roc']:.4f}")
        logger.info(f"最終AUC-PR: {trainer.training_stats['evaluation_metrics']['auc_pr']:.4f}")
        logger.info(f"最終精度: {trainer.training_stats['evaluation_metrics']['accuracy']:.4f}")
        logger.info(f"訓練時間: {trainer.training_stats['training_time']:.2f}秒")
        logger.info(f"モデル保存先: {args.output_dir}")
        
        logger.info("Enhanced Two-Tower Training 完了")
        
    except Exception as e:
        logger.error(f"訓練エラー: {e}")
        raise

if __name__ == '__main__':
    main()