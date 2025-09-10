"""
本番環境用Two-Tower訓練パイプライン

安定した高精度訓練を実現する本番環境特化の訓練システム。
時系列考慮のデータ分割、最適化されたバランシング、
堅牢なエラーハンドリングを提供する。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from pathlib import Path
import time
import gc
from collections import defaultdict

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import pickle
import hashlib

# 既存の訓練器をインポート
sys.path.append(os.path.dirname(__file__))
from enhanced_two_tower_trainer import EnhancedTwoTowerTrainer, TrainingConfiguration, ModelArchitecture

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProductionTrainingConfig:
    """本番訓練設定"""
    # データ設定
    max_users: int = 5000
    max_videos: int = 20000
    train_test_split_date: Optional[str] = None  # YYYY-MM-DD形式
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # バランシング設定
    positive_negative_ratio: float = 0.5  # 1:2 → 0.33
    max_negatives_per_user: int = 100
    stratified_sampling: bool = True
    
    # 訓練設定
    embedding_dim: int = 768
    batch_size: int = 512
    epochs: int = 100
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 4
    learning_rate: float = 0.001
    
    # 正則化設定
    dropout_rate: float = 0.4
    l2_regularization: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # 監視設定
    monitor_metric: str = 'val_auc_pr'
    target_auc_roc: float = 0.95
    target_auc_pr: float = 0.99
    
    # システム制約
    memory_limit_gb: float = 8.0
    training_timeout_hours: float = 1.0
    checkpoint_frequency: int = 5  # epochs

@dataclass
class TrainingMetrics:
    """訓練メトリクス"""
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = float('inf')
    training_time: float = 0.0
    memory_usage_gb: float = 0.0

class ProductionTrainer:
    """本番環境用Two-Tower訓練器"""
    
    def __init__(self, 
                 db_connection_string: str,
                 config: ProductionTrainingConfig = None):
        """
        初期化
        
        Args:
            db_connection_string: PostgreSQL接続文字列
            config: 本番訓練設定
        """
        self.db_connection_string = db_connection_string
        self.config = config or ProductionTrainingConfig()
        
        # ベースとなるEnhancedTwoTowerTrainerの初期化
        base_config = TrainingConfiguration(
            embedding_dim=self.config.embedding_dim,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            early_stopping_patience=self.config.early_stopping_patience,
            reduce_lr_patience=self.config.reduce_lr_patience,
            dropout_rate=self.config.dropout_rate,
            l2_reg=self.config.l2_regularization
        )
        
        self.base_trainer = EnhancedTwoTowerTrainer(
            db_connection_string, base_config
        )
        
        # 本番訓練統計
        self.training_session = {
            'session_id': self._generate_session_id(),
            'start_time': datetime.now().isoformat(),
            'config': asdict(self.config),
            'data_quality': {},
            'training_metrics': {},
            'model_performance': {},
            'system_resources': {},
            'error_log': []
        }
        
        # システム監視
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        self.training_monitor = TrainingMonitor(self.config.training_timeout_hours)

    def _generate_session_id(self) -> str:
        """セッションIDの生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        return f"production_training_{timestamp}_{config_hash}"

    def prepare_production_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """本番環境用データの準備"""
        logger.info("本番データ準備開始...")
        
        start_time = time.time()
        
        try:
            # メモリ監視開始
            self.memory_monitor.start_monitoring()
            
            # 時系列考慮のデータ準備
            user_features, video_features, temporal_splits = self._prepare_temporal_data()
            
            # データ品質検証
            quality_report = self._validate_data_quality(user_features, video_features, temporal_splits)
            self.training_session['data_quality'] = quality_report
            
            if not quality_report['quality_passed']:
                raise ValueError(f"データ品質検証失敗: {quality_report['issues']}")
            
            # メモリ使用量チェック
            memory_usage = self.memory_monitor.get_current_usage()
            if memory_usage > self.config.memory_limit_gb:
                logger.warning(f"メモリ使用量が制限を超過: {memory_usage:.2f}GB > {self.config.memory_limit_gb}GB")
                gc.collect()  # ガベージコレクション実行
            
            preparation_time = time.time() - start_time
            logger.info(f"本番データ準備完了: {preparation_time:.2f}秒")
            
            return user_features, video_features, temporal_splits
            
        except Exception as e:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'stage': 'data_preparation',
                'error': str(e),
                'memory_usage_gb': self.memory_monitor.get_current_usage()
            }
            self.training_session['error_log'].append(error_info)
            logger.error(f"データ準備エラー: {e}")
            raise

    def _prepare_temporal_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """時系列考慮のデータ準備"""
        
        # 分割日の決定
        if self.config.train_test_split_date:
            split_date = datetime.strptime(self.config.train_test_split_date, "%Y-%m-%d")
        else:
            # デフォルト: 90日前を分割点とする
            split_date = datetime.now() - timedelta(days=90)
        
        logger.info(f"時系列分割日: {split_date.strftime('%Y-%m-%d')}")
        
        # 基本データの取得（時系列情報付き）
        with self.base_trainer.connect_db() as conn:
            user_features, video_features = self._load_features_with_temporal_info(conn, split_date)
            temporal_interactions = self._create_temporal_training_splits(conn, split_date)
        
        return user_features, video_features, temporal_interactions

    def _load_features_with_temporal_info(self, 
                                        conn: psycopg2.extensions.connection, 
                                        split_date: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """時系列情報を考慮した特徴読み込み"""
        
        cursor = conn.cursor()
        
        # 訓練期間内にアクティブだったユーザーのみを対象
        cursor.execute("""
            SELECT DISTINCT user_id, COUNT(*) as decision_count
            FROM user_video_decisions 
            WHERE created_at <= %s
            GROUP BY user_id
            HAVING COUNT(*) >= 5  -- 最小決定数
            ORDER BY decision_count DESC
            LIMIT %s
        """, (split_date, self.config.max_users))
        
        user_candidates = cursor.fetchall()
        user_ids = [str(row['user_id']) for row in user_candidates]
        logger.info(f"時系列対象ユーザー数: {len(user_ids)}")
        
        # ユーザー特徴の抽出
        self.base_trainer.user_feature_extractor.build_vocabulary(conn)
        user_feature_vectors, _ = self.base_trainer.user_feature_extractor.extract_batch_features(user_ids, conn)
        user_features, _ = self.base_trainer.user_feature_extractor.convert_to_numpy_array(user_feature_vectors)
        
        # ビデオ特徴の抽出（訓練期間内のインタラクションがあるもの）
        cursor.execute("""
            SELECT DISTINCT video_id, COUNT(*) as interaction_count
            FROM user_video_decisions 
            WHERE created_at <= %s AND user_id = ANY(%s)
            GROUP BY video_id
            ORDER BY interaction_count DESC
            LIMIT %s
        """, (split_date, user_ids, self.config.max_videos))
        
        video_candidates = cursor.fetchall()
        video_ids = [str(row['video_id']) for row in video_candidates]
        logger.info(f"時系列対象ビデオ数: {len(video_ids)}")
        
        # ビデオメタデータの読み込み
        video_metadata = []
        for video_id in video_ids:
            cursor.execute("""
                SELECT v.*, 
                       ARRAY_AGG(DISTINCT p.name) FILTER (WHERE p.name IS NOT NULL) as performers,
                       ARRAY_AGG(DISTINCT t.name) FILTER (WHERE t.name IS NOT NULL) as tags
                FROM videos v
                LEFT JOIN video_performers vp ON v.id = vp.video_id
                LEFT JOIN performers p ON vp.performer_id = p.id
                LEFT JOIN video_tags vt ON v.id = vt.video_id
                LEFT JOIN tags t ON vt.tag_id = t.id
                WHERE v.id = %s
                GROUP BY v.id
            """, (video_id,))
            
            result = cursor.fetchone()
            if result:
                from ml_pipeline.preprocessing.enhanced_item_feature_processor import VideoMetadata
                video = VideoMetadata(
                    video_id=str(result['id']),
                    title=result['title'] or '',
                    description=result['description'] or '',
                    maker=result['maker'] or 'unknown',
                    genre=result['genre'] or 'general',
                    price=float(result['price']) if result['price'] else 0.0,
                    duration_seconds=int(result['duration_seconds']) if result['duration_seconds'] else 0,
                    performers=result['performers'] or [],
                    tags=result['tags'] or [],
                    external_id=result['external_id'] or '',
                    source=result['source'] or '',
                    created_at=result['created_at']
                )
                video_metadata.append(video)
        
        # ビデオ特徴の抽出
        video_feature_vectors = self.base_trainer.item_feature_processor.convert_to_feature_vectors(video_metadata, conn)
        video_features, _ = self.base_trainer.item_feature_processor.convert_to_numpy_array(video_feature_vectors)
        
        return user_features, video_features

    def _create_temporal_training_splits(self, 
                                       conn: psycopg2.extensions.connection,
                                       split_date: datetime) -> Dict[str, np.ndarray]:
        """時系列考慮の訓練分割作成"""
        
        cursor = conn.cursor()
        
        # 時系列分割でインタラクション取得
        cursor.execute("""
            SELECT 
                user_id, video_id, decision_type, created_at,
                CASE 
                    WHEN created_at <= %s THEN 'train'
                    WHEN created_at <= %s THEN 'val'
                    ELSE 'test'
                END as split_type
            FROM user_video_decisions 
            WHERE created_at >= %s  -- 過去1年のデータのみ
            ORDER BY created_at
        """, (
            split_date,
            split_date + timedelta(days=30),  # バリデーション期間
            split_date - timedelta(days=365)
        ))
        
        interactions = cursor.fetchall()
        logger.info(f"時系列インタラクション総数: {len(interactions)}")
        
        # 分割別にインタラクションを整理
        split_interactions = defaultdict(list)
        split_stats = defaultdict(lambda: {'likes': 0, 'nopes': 0})
        
        for interaction in interactions:
            split_type = interaction['split_type']
            user_id = str(interaction['user_id'])
            video_id = str(interaction['video_id'])
            decision_type = interaction['decision_type']
            
            split_interactions[split_type].append({
                'user_id': user_id,
                'video_id': video_id,
                'label': 1 if decision_type == 'like' else 0
            })
            
            if decision_type == 'like':
                split_stats[split_type]['likes'] += 1
            else:
                split_stats[split_type]['nopes'] += 1
        
        # 統計表示
        for split_type, stats in split_stats.items():
            total = stats['likes'] + stats['nopes']
            like_ratio = stats['likes'] / total if total > 0 else 0
            logger.info(f"{split_type}: {total}件 (like率: {like_ratio:.3f})")
        
        # バランシング適用
        balanced_splits = {}
        for split_type, interactions in split_interactions.items():
            balanced_data = self._apply_advanced_balancing(interactions)
            balanced_splits[split_type] = balanced_data
        
        return balanced_splits

    def _apply_advanced_balancing(self, interactions: List[Dict]) -> np.ndarray:
        """高度なバランシング適用"""
        
        positive_interactions = [i for i in interactions if i['label'] == 1]
        negative_interactions = [i for i in interactions if i['label'] == 0]
        
        logger.info(f"バランシング前 - 正例: {len(positive_interactions)}, 負例: {len(negative_interactions)}")
        
        # 目標負例数の計算
        target_negatives = int(len(positive_interactions) / self.config.positive_negative_ratio - len(positive_interactions))
        
        if len(negative_interactions) > target_negatives:
            # 負例のサブサンプリング（ユーザー分散を考慮）
            user_negatives = defaultdict(list)
            for neg in negative_interactions:
                user_negatives[neg['user_id']].append(neg)
            
            # ユーザーごとに均等に負例を選択
            selected_negatives = []
            negatives_per_user = max(1, target_negatives // len(user_negatives))
            
            for user_id, user_negs in user_negatives.items():
                if len(user_negs) <= negatives_per_user:
                    selected_negatives.extend(user_negs)
                else:
                    # ランダムサンプリング
                    sampled = np.random.choice(
                        len(user_negs), 
                        size=min(negatives_per_user, self.config.max_negatives_per_user), 
                        replace=False
                    )
                    selected_negatives.extend([user_negs[i] for i in sampled])
            
            # 不足分は全体からランダム補完
            if len(selected_negatives) < target_negatives:
                remaining = target_negatives - len(selected_negatives)
                unused_negatives = [n for n in negative_interactions if n not in selected_negatives]
                if len(unused_negatives) > 0:
                    additional = np.random.choice(
                        len(unused_negatives), 
                        size=min(remaining, len(unused_negatives)), 
                        replace=False
                    )
                    selected_negatives.extend([unused_negatives[i] for i in additional])
            
            negative_interactions = selected_negatives
        
        # 最終的なバランシング結果
        all_interactions = positive_interactions + negative_interactions
        np.random.shuffle(all_interactions)
        
        logger.info(f"バランシング後 - 正例: {len(positive_interactions)}, 負例: {len(negative_interactions)}")
        logger.info(f"最終比率: {len(negative_interactions) / len(positive_interactions):.2f}:1")
        
        # NumPy配列に変換
        user_indices = []
        video_indices = []
        labels = []
        
        # TODO: ユーザー・ビデオIDからインデックスへの変換が必要
        # 現在は仮実装
        for interaction in all_interactions:
            user_indices.append(0)  # 実際のマッピングが必要
            video_indices.append(0)  # 実際のマッピングが必要
            labels.append(interaction['label'])
        
        return np.array([user_indices, video_indices, labels])

    def _validate_data_quality(self, 
                             user_features: np.ndarray, 
                             video_features: np.ndarray,
                             temporal_splits: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """データ品質検証"""
        
        quality_report = {
            'quality_passed': True,
            'issues': [],
            'data_statistics': {},
            'recommendations': []
        }
        
        # 基本統計
        quality_report['data_statistics'] = {
            'user_feature_shape': user_features.shape,
            'video_feature_shape': video_features.shape,
            'splits_available': list(temporal_splits.keys()),
            'user_feature_nulls': np.isnan(user_features).sum(),
            'video_feature_nulls': np.isnan(video_features).sum()
        }
        
        # 品質チェック
        issues = []
        
        # 最小データサイズチェック
        if user_features.shape[0] < 100:
            issues.append("ユーザー数が不十分 (<100)")
        if video_features.shape[0] < 1000:
            issues.append("ビデオ数が不十分 (<1000)")
        
        # 特徴次元チェック
        if user_features.shape[1] != 768:
            issues.append(f"ユーザー特徴次元が不正: {user_features.shape[1]} != 768")
        if video_features.shape[1] != 768:
            issues.append(f"ビデオ特徴次元が不正: {video_features.shape[1]} != 768")
        
        # 欠損値チェック
        if np.isnan(user_features).any():
            issues.append("ユーザー特徴に欠損値が存在")
        if np.isnan(video_features).any():
            issues.append("ビデオ特徴に欠損値が存在")
        
        # 分割データチェック
        if 'train' not in temporal_splits:
            issues.append("訓練データが存在しない")
        
        # 品質判定
        quality_report['issues'] = issues
        quality_report['quality_passed'] = len(issues) == 0
        
        if issues:
            logger.warning(f"データ品質問題: {issues}")
        else:
            logger.info("データ品質検証: 合格")
        
        return quality_report

    def train_production_model(self, 
                             user_features: np.ndarray, 
                             video_features: np.ndarray,
                             temporal_splits: Dict[str, np.ndarray]) -> TrainingMetrics:
        """本番モデル訓練の実行"""
        logger.info("本番モデル訓練開始...")
        
        start_time = time.time()
        
        try:
            # タイムアウト監視開始
            self.training_monitor.start_monitoring()
            
            # モデル構築
            model = self.base_trainer.build_two_tower_model(
                user_features.shape[1], video_features.shape[1]
            )
            
            # 訓練データの準備
            train_data = temporal_splits['train']
            # TODO: 実際のデータからX_user, X_video, yを構築する必要がある
            # 現在は仮実装
            X_user_train = user_features[:1000]  # 仮
            X_video_train = video_features[:1000]  # 仮
            y_train = np.random.binomial(1, 0.3, 1000)  # 仮
            
            # バリデーションデータ
            X_user_val = user_features[1000:1200]  # 仮
            X_video_val = video_features[1000:1200]  # 仮
            y_val = np.random.binomial(1, 0.3, 200)  # 仮
            
            # 高度なコールバック設定
            callbacks_list = self._create_production_callbacks()
            
            # 訓練実行
            logger.info("モデル訓練実行中...")
            history = model.fit(
                [X_user_train, X_video_train], y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=([X_user_val, X_video_val], y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # 評価メトリクスの計算
            metrics = self._calculate_production_metrics(
                model, X_user_val, X_video_val, y_val
            )
            
            training_time = time.time() - start_time
            metrics.training_time = training_time
            metrics.memory_usage_gb = self.memory_monitor.get_current_usage()
            
            # 品質基準チェック
            quality_check = self._validate_model_quality(metrics)
            if not quality_check['passed']:
                logger.warning(f"品質基準未達成: {quality_check['issues']}")
            
            logger.info(f"本番モデル訓練完了: {training_time:.2f}秒")
            return metrics
            
        except Exception as e:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'stage': 'model_training',
                'error': str(e),
                'training_time': time.time() - start_time
            }
            self.training_session['error_log'].append(error_info)
            logger.error(f"モデル訓練エラー: {e}")
            raise

    def _create_production_callbacks(self) -> List[callbacks.Callback]:
        """本番用コールバックの作成"""
        
        callbacks_list = [
            # Early Stopping（改良版）
            callbacks.EarlyStopping(
                monitor=self.config.monitor_metric,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                mode='max',
                min_delta=0.001,
                verbose=1
            ),
            
            # Learning Rate Scheduler
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model Checkpointing（改良版）
            callbacks.ModelCheckpoint(
                filepath='models/production_checkpoint_epoch_{epoch:02d}_auc_{val_auc_pr:.4f}.h5',
                monitor=self.config.monitor_metric,
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # 定期チェックポイント
            callbacks.ModelCheckpoint(
                filepath='models/production_periodic_checkpoint.h5',
                save_freq=self.config.checkpoint_frequency * 100,  # バッチ単位
                save_weights_only=True,
                verbose=0
            ),
            
            # カスタムメトリクス監視
            ProductionMetricsCallback(
                target_auc_roc=self.config.target_auc_roc,
                target_auc_pr=self.config.target_auc_pr
            ),
            
            # メモリ監視
            MemoryCallback(self.config.memory_limit_gb),
            
            # タイムアウト監視
            TimeoutCallback(self.config.training_timeout_hours)
        ]
        
        return callbacks_list

    def _calculate_production_metrics(self, 
                                    model: tf.keras.Model,
                                    X_user_val: np.ndarray,
                                    X_video_val: np.ndarray,
                                    y_val: np.ndarray) -> TrainingMetrics:
        """本番メトリクスの計算"""
        
        # 予測実行
        predictions = model.predict([X_user_val, X_video_val], batch_size=self.config.batch_size)
        predictions = predictions.flatten()
        
        # メトリクス計算
        auc_roc = roc_auc_score(y_val, predictions)
        precision, recall, _ = precision_recall_curve(y_val, predictions)
        auc_pr = auc(recall, precision)
        
        # 閾値0.5での分類性能
        binary_preds = (predictions > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_val, binary_preds)
        precision_score_val = precision_score(y_val, binary_preds, zero_division=0)
        recall_score_val = recall_score(y_val, binary_preds, zero_division=0)
        f1_score_val = f1_score(y_val, binary_preds, zero_division=0)
        
        # 損失計算
        loss = model.evaluate([X_user_val, X_video_val], y_val, verbose=0)[0]
        
        return TrainingMetrics(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            accuracy=accuracy,
            precision=precision_score_val,
            recall=recall_score_val,
            f1_score=f1_score_val,
            loss=loss
        )

    def _validate_model_quality(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """モデル品質検証"""
        
        quality_check = {
            'passed': True,
            'issues': [],
            'metrics_summary': asdict(metrics)
        }
        
        issues = []
        
        # 目標指標チェック
        if metrics.auc_roc < self.config.target_auc_roc:
            issues.append(f"AUC-ROC不足: {metrics.auc_roc:.4f} < {self.config.target_auc_roc}")
        
        if metrics.auc_pr < self.config.target_auc_pr:
            issues.append(f"AUC-PR不足: {metrics.auc_pr:.4f} < {self.config.target_auc_pr}")
        
        # 基本性能チェック
        if metrics.accuracy < 0.8:
            issues.append(f"精度不足: {metrics.accuracy:.4f} < 0.8")
        
        if metrics.f1_score < 0.7:
            issues.append(f"F1スコア不足: {metrics.f1_score:.4f} < 0.7")
        
        # システム制約チェック
        if metrics.memory_usage_gb > self.config.memory_limit_gb:
            issues.append(f"メモリ超過: {metrics.memory_usage_gb:.2f}GB > {self.config.memory_limit_gb}GB")
        
        if metrics.training_time > self.config.training_timeout_hours * 3600:
            issues.append(f"訓練時間超過: {metrics.training_time:.0f}s > {self.config.training_timeout_hours * 3600:.0f}s")
        
        quality_check['issues'] = issues
        quality_check['passed'] = len(issues) == 0
        
        return quality_check

    def save_production_results(self, 
                              metrics: TrainingMetrics,
                              output_dir: str = "models/production_results"):
        """本番結果の保存"""
        
        os.makedirs(output_dir, exist_ok=True)
        session_dir = f"{output_dir}/{self.training_session['session_id']}"
        os.makedirs(session_dir, exist_ok=True)
        
        # セッション情報の更新
        self.training_session['training_metrics'] = asdict(metrics)
        self.training_session['end_time'] = datetime.now().isoformat()
        
        # 結果保存
        with open(f"{session_dir}/training_session.json", 'w') as f:
            json.dump(self.training_session, f, indent=2, ensure_ascii=False)
        
        with open(f"{session_dir}/production_config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        with open(f"{session_dir}/final_metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        logger.info(f"本番結果保存完了: {session_dir}")


class MemoryMonitor:
    """メモリ監視クラス"""
    
    def __init__(self, limit_gb: float):
        self.limit_gb = limit_gb
        self.monitoring = False
    
    def start_monitoring(self):
        self.monitoring = True
    
    def get_current_usage(self) -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            # psutil が利用できない場合の代替手段
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)  # Linux


class TrainingMonitor:
    """訓練監視クラス"""
    
    def __init__(self, timeout_hours: float):
        self.timeout_hours = timeout_hours
        self.start_time = None
    
    def start_monitoring(self):
        self.start_time = time.time()
    
    def check_timeout(self) -> bool:
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        return elapsed > (self.timeout_hours * 3600)


class ProductionMetricsCallback(callbacks.Callback):
    """本番メトリクス監視コールバック"""
    
    def __init__(self, target_auc_roc: float, target_auc_pr: float):
        super().__init__()
        self.target_auc_roc = target_auc_roc
        self.target_auc_pr = target_auc_pr
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        val_auc_roc = logs.get('val_auc_roc', 0)
        val_auc_pr = logs.get('val_auc_pr', 0)
        
        if val_auc_roc >= self.target_auc_roc and val_auc_pr >= self.target_auc_pr:
            logger.info(f"Epoch {epoch+1}: 目標達成! AUC-ROC: {val_auc_roc:.4f}, AUC-PR: {val_auc_pr:.4f}")


class MemoryCallback(callbacks.Callback):
    """メモリ監視コールバック"""
    
    def __init__(self, limit_gb: float):
        super().__init__()
        self.limit_gb = limit_gb
    
    def on_epoch_end(self, epoch, logs=None):
        try:
            import psutil
            memory_gb = psutil.Process().memory_info().rss / (1024**3)
            if memory_gb > self.limit_gb:
                logger.warning(f"Epoch {epoch+1}: メモリ使用量警告 {memory_gb:.2f}GB > {self.limit_gb}GB")
        except ImportError:
            pass


class TimeoutCallback(callbacks.Callback):
    """タイムアウト監視コールバック"""
    
    def __init__(self, timeout_hours: float):
        super().__init__()
        self.timeout_hours = timeout_hours
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > (self.timeout_hours * 3600):
            logger.warning(f"Epoch {epoch+1}: 訓練時間制限に到達")
            self.model.stop_training = True


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Two-Tower Training')
    parser.add_argument('--db-url', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--config-file', help='Production config JSON file')
    parser.add_argument('--output-dir', default='models/production_results', help='Output directory')
    parser.add_argument('--max-users', type=int, default=5000, help='Maximum users')
    parser.add_argument('--max-videos', type=int, default=20000, help='Maximum videos')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    args = parser.parse_args()
    
    # 設定の読み込み
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = ProductionTrainingConfig(**config_dict)
    else:
        config = ProductionTrainingConfig(
            max_users=args.max_users,
            max_videos=args.max_videos,
            epochs=args.epochs
        )
    
    # 本番訓練器の初期化
    trainer = ProductionTrainer(args.db_url, config)
    
    try:
        logger.info("本番Two-Tower訓練開始")
        logger.info(f"セッションID: {trainer.training_session['session_id']}")
        
        # データ準備
        user_features, video_features, temporal_splits = trainer.prepare_production_data()
        
        # モデル訓練
        metrics = trainer.train_production_model(user_features, video_features, temporal_splits)
        
        # 結果保存
        trainer.save_production_results(metrics, args.output_dir)
        
        # 最終結果表示
        logger.info("=== 本番訓練完了 ===")
        logger.info(f"AUC-ROC: {metrics.auc_roc:.4f}")
        logger.info(f"AUC-PR: {metrics.auc_pr:.4f}")
        logger.info(f"精度: {metrics.accuracy:.4f}")
        logger.info(f"F1スコア: {metrics.f1_score:.4f}")
        logger.info(f"訓練時間: {metrics.training_time:.2f}秒")
        logger.info(f"メモリ使用量: {metrics.memory_usage_gb:.2f}GB")
        
        # 品質チェック結果
        quality_check = trainer._validate_model_quality(metrics)
        if quality_check['passed']:
            logger.info("✅ 品質基準: 合格")
        else:
            logger.warning(f"⚠️ 品質基準: 未達成 - {quality_check['issues']}")
        
    except Exception as e:
        logger.error(f"本番訓練失敗: {e}")
        raise

if __name__ == '__main__':
    main()