"""
Unified Two-Tower Trainer

統一されたTwo-Tower推薦モデルトレーナー
既存のEnhancedTwoTowerTrainerとRatingBasedTwoTowerTrainerを統合し、
新しいMLパッケージ構造に対応した統一インターフェースを提供
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pickle

# Backend ML package imports
from backend.ml import STANDARD_MODEL_CONFIG, DEFAULT_EMBEDDING_DIM, PACKAGE_ROOT
from backend.ml.utils.config_loader import ConfigLoader
from backend.ml.utils.logger import get_ml_logger
from backend.ml.utils.gpu_manager import GPUManager

logger = get_ml_logger(__name__)

# Import configuration classes
from backend.ml.config import TrainingConfig, TrainingStats

class UnifiedTwoTowerTrainer:
    """統一Two-Towerトレーナークラス"""
    
    def __init__(self, 
                 config: Optional[Union[TrainingConfig, str, Path]] = None,
                 model_save_dir: Optional[Union[str, Path]] = None,
                 experiment_name: Optional[str] = None):
        """
        初期化
        
        Args:
            config: トレーニング設定（TrainingConfigオブジェクトまたは設定ファイルパス）
            model_save_dir: モデル保存ディレクトリ
            experiment_name: 実験名
        """
        self.experiment_name = experiment_name or f"two_tower_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 設定の読み込み
        if isinstance(config, (str, Path)):
            self.config = ConfigLoader.load_training_config(config)
        elif isinstance(config, TrainingConfig):
            self.config = config
        else:
            self.config = TrainingConfig()
        
        # ディレクトリ設定
        self.model_save_dir = Path(model_save_dir) if model_save_dir else PACKAGE_ROOT / "models" / self.experiment_name
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # ログ設定
        self.log_dir = PACKAGE_ROOT / "training" / "logs" / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント設定
        self.checkpoint_dir = PACKAGE_ROOT / "training" / "checkpoints" / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル関連
        self.user_tower = None
        self.item_tower = None
        self.full_model = None
        self.preprocessors = {}
        
        # 統計情報
        self.stats = TrainingStats(start_time=datetime.now().isoformat())
        self.training_history = {}
        
        # GPU設定
        GPUManager.setup_gpu()
        
        logger.info(f"UnifiedTwoTowerTrainer initialized: {self.experiment_name}")
        logger.info(f"Model save dir: {self.model_save_dir}")
        logger.info(f"Config: {asdict(self.config)}")

    def build_user_tower(self, input_dim: int) -> tf.keras.Model:
        """ユーザータワーの構築"""
        inputs = layers.Input(shape=(input_dim,), name='user_features')
        x = inputs
        
        # 隠れ層
        for i, units in enumerate(self.config.user_hidden_units):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization),
                name=f'user_dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'user_bn_{i+1}')(x)
            x = layers.Dropout(self.config.dropout_rate, name=f'user_dropout_{i+1}')(x)
        
        # 最終埋め込み層
        user_embedding = layers.Dense(
            self.config.user_embedding_dim,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization),
            name='user_embedding'
        )(x)
        
        # L2正規化
        user_embedding = tf.nn.l2_normalize(user_embedding, axis=1, name='user_embedding_normalized')
        
        model = tf.keras.Model(inputs=inputs, outputs=user_embedding, name='user_tower')
        return model

    def build_item_tower(self, input_dim: int) -> tf.keras.Model:
        """アイテムタワーの構築"""
        inputs = layers.Input(shape=(input_dim,), name='item_features')
        x = inputs
        
        # 隠れ層
        for i, units in enumerate(self.config.item_hidden_units):
            x = layers.Dense(
                units,
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization),
                name=f'item_dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'item_bn_{i+1}')(x)
            x = layers.Dropout(self.config.dropout_rate, name=f'item_dropout_{i+1}')(x)
        
        # 最終埋め込み層
        item_embedding = layers.Dense(
            self.config.item_embedding_dim,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization), 
            name='item_embedding'
        )(x)
        
        # L2正規化
        item_embedding = tf.nn.l2_normalize(item_embedding, axis=1, name='item_embedding_normalized')
        
        model = tf.keras.Model(inputs=inputs, outputs=item_embedding, name='item_tower')
        return model

    def build_full_model(self, user_feature_dim: int, item_feature_dim: int) -> tf.keras.Model:
        """完全なTwo-Towerモデルの構築"""
        # タワーの構築
        self.user_tower = self.build_user_tower(user_feature_dim)
        self.item_tower = self.build_item_tower(item_feature_dim)
        
        # 入力
        user_input = layers.Input(shape=(user_feature_dim,), name='user_input')
        item_input = layers.Input(shape=(item_feature_dim,), name='item_input')
        
        # 埋め込み生成
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        
        # 相互作用スコア計算（コサイン類似度）
        similarity = layers.Dot(axes=1, normalize=False, name='similarity')([user_embedding, item_embedding])
        
        # シグモイド活性化で確率に変換
        prediction = layers.Activation('sigmoid', name='prediction')(similarity)
        
        # 完全モデル
        model = tf.keras.Model(
            inputs=[user_input, item_input],
            outputs=prediction,
            name='two_tower_model'
        )
        
        return model

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """モデルのコンパイル"""
        optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=self.config.loss_function,
            metrics=self.config.metrics
        )
        
        return model

    def create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """コールバックの作成"""
        callbacks_list = []
        
        # Early Stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Learning Rate Reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_learning_rate,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # Model Checkpoint
        checkpoint_path = self.checkpoint_dir / "best_model.keras"
        model_checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks_list.append(model_checkpoint)
        
        # TensorBoard
        tensorboard = callbacks.TensorBoard(
            log_dir=str(self.log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks_list.append(tensorboard)
        
        return callbacks_list

    def train(self, 
              user_features: np.ndarray,
              item_features: np.ndarray, 
              interactions: np.ndarray,
              validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        モデルのトレーニング
        
        Args:
            user_features: ユーザー特徴量 (N, user_feature_dim)
            item_features: アイテム特徴量 (N, item_feature_dim)
            interactions: インタラクション (N,) - 1: positive, 0: negative
            validation_data: バリデーションデータ (optional)
            
        Returns:
            トレーニング履歴
        """
        logger.info("Training started")
        start_time = datetime.now()
        
        # データ統計更新
        self.stats.train_samples = len(user_features)
        
        # モデル構築
        user_feature_dim = user_features.shape[1]
        item_feature_dim = item_features.shape[1]
        
        self.full_model = self.build_full_model(user_feature_dim, item_feature_dim)
        self.full_model = self.compile_model(self.full_model)
        
        # モデル統計更新
        self.stats.total_params = self.full_model.count_params()
        self.stats.user_tower_params = self.user_tower.count_params()
        self.stats.item_tower_params = self.item_tower.count_params()
        
        logger.info(f"Model built - Total params: {self.stats.total_params}")
        logger.info(f"User tower params: {self.stats.user_tower_params}")
        logger.info(f"Item tower params: {self.stats.item_tower_params}")
        
        # バリデーションデータの準備
        if validation_data is None:
            x_train, x_val, y_train, y_val = train_test_split(
                [user_features, item_features], 
                interactions,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=interactions
            )
            validation_data = (x_val, y_val)
        else:
            x_train, y_train = [user_features, item_features], interactions
            
        self.stats.validation_samples = len(validation_data[1])
        
        # コールバック準備
        callbacks_list = self.create_callbacks()
        
        # トレーニング実行
        history = self.full_model.fit(
            x_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            shuffle=self.config.shuffle,
            verbose=1
        )
        
        # 統計更新
        end_time = datetime.now()
        self.stats.end_time = end_time.isoformat()
        self.stats.total_duration_seconds = (end_time - start_time).total_seconds()
        
        # 最終メトリクス更新
        final_epoch = len(history.history['loss']) - 1
        self.stats.final_train_loss = history.history['loss'][final_epoch]
        self.stats.final_val_loss = history.history['val_loss'][final_epoch]
        
        if 'accuracy' in history.history:
            self.stats.final_train_accuracy = history.history['accuracy'][final_epoch]
            self.stats.final_val_accuracy = history.history['val_accuracy'][final_epoch]
        
        self.stats.best_epoch = np.argmin(history.history['val_loss'])
        
        self.training_history = history.history
        
        logger.info(f"Training completed in {self.stats.total_duration_seconds:.2f} seconds")
        logger.info(f"Best epoch: {self.stats.best_epoch}")
        logger.info(f"Final validation loss: {self.stats.final_val_loss:.4f}")
        
        return history.history

    def save_model(self, version: str = "latest") -> Dict[str, Path]:
        """モデルとメタデータの保存"""
        saved_files = {}
        
        # 完全モデル保存
        full_model_path = self.model_save_dir / f"full_model_{version}.keras"
        self.full_model.save(full_model_path)
        saved_files['full_model'] = full_model_path
        
        # 個別タワー保存
        user_tower_path = self.model_save_dir / f"user_tower_{version}.keras"
        item_tower_path = self.model_save_dir / f"item_tower_{version}.keras"
        
        self.user_tower.save(user_tower_path)
        self.item_tower.save(item_tower_path)
        
        saved_files['user_tower'] = user_tower_path
        saved_files['item_tower'] = item_tower_path
        
        # 設定・統計保存
        config_path = self.model_save_dir / f"config_{version}.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        saved_files['config'] = config_path
        
        stats_path = self.model_save_dir / f"training_stats_{version}.json"
        with open(stats_path, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)
        saved_files['stats'] = stats_path
        
        # トレーニング履歴保存
        if self.training_history:
            history_path = self.model_save_dir / f"training_history_{version}.json"
            with open(history_path, 'w') as f:
                # NumPyオブジェクトをリストに変換
                history_dict = {k: [float(x) for x in v] for k, v in self.training_history.items()}
                json.dump(history_dict, f, indent=2)
            saved_files['history'] = history_path
        
        logger.info(f"Model saved to {self.model_save_dir}")
        return saved_files

    def load_model(self, version: str = "latest") -> bool:
        """保存されたモデルの読み込み"""
        try:
            full_model_path = self.model_save_dir / f"full_model_{version}.keras"
            user_tower_path = self.model_save_dir / f"user_tower_{version}.keras"
            item_tower_path = self.model_save_dir / f"item_tower_{version}.keras"
            
            self.full_model = tf.keras.models.load_model(full_model_path)
            self.user_tower = tf.keras.models.load_model(user_tower_path)
            self.item_tower = tf.keras.models.load_model(item_tower_path)
            
            # 設定読み込み
            config_path = self.model_save_dir / f"config_{version}.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    self.config = TrainingConfig(**config_dict)
            
            logger.info(f"Model loaded from {self.model_save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_embeddings(self, 
                      user_features: Optional[np.ndarray] = None,
                      item_features: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """埋め込みベクトルの取得"""
        user_embeddings = None
        item_embeddings = None
        
        if user_features is not None and self.user_tower is not None:
            user_embeddings = self.user_tower.predict(user_features)
            
        if item_features is not None and self.item_tower is not None:
            item_embeddings = self.item_tower.predict(item_features)
            
        return user_embeddings, item_embeddings

    def predict(self, user_features: np.ndarray, item_features: np.ndarray) -> np.ndarray:
        """予測実行"""
        if self.full_model is None:
            raise ValueError("Model not trained or loaded")
            
        predictions = self.full_model.predict([user_features, item_features])
        return predictions.flatten()

    def evaluate_model(self, 
                      user_features: np.ndarray,
                      item_features: np.ndarray,
                      true_labels: np.ndarray) -> Dict[str, float]:
        """モデル評価"""
        if self.full_model is None:
            raise ValueError("Model not trained or loaded")
        
        # 予測
        predictions = self.predict(user_features, item_features)
        
        # 評価指標計算
        test_loss = self.full_model.evaluate([user_features, item_features], true_labels, verbose=0)
        
        # AUC計算
        auc_score = roc_auc_score(true_labels, predictions)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        pr_auc = auc(recall, precision)
        
        # Binary predictions for classification metrics
        binary_predictions = (predictions >= 0.5).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_score_val = precision_score(true_labels, binary_predictions)
        recall_score_val = recall_score(true_labels, binary_predictions)
        f1_score_val = f1_score(true_labels, binary_predictions)
        
        evaluation_results = {
            'test_loss': float(test_loss[0] if isinstance(test_loss, list) else test_loss),
            'auc_roc': float(auc_score),
            'auc_pr': float(pr_auc),
            'precision': float(precision_score_val),
            'recall': float(recall_score_val),
            'f1': float(f1_score_val)
        }
        
        # 統計更新
        self.stats.test_auc = evaluation_results['auc_roc']
        self.stats.test_precision = evaluation_results['precision']
        self.stats.test_recall = evaluation_results['recall'] 
        self.stats.test_f1 = evaluation_results['f1']
        
        logger.info("Model evaluation completed")
        for metric, value in evaluation_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return evaluation_results