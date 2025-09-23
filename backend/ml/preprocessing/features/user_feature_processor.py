"""
User Feature Processor

ユーザー特徴量専用プロセッサー
既存のRealUserFeatureExtractorを統合し、新しいMLパッケージ構造に対応
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from backend.ml.preprocessing.features.feature_processor import BaseFeatureProcessor, FeatureConfig
from backend.ml.utils.logger import get_ml_logger

logger = get_ml_logger(__name__)

@dataclass
class UserBehaviorMetrics:
    """ユーザー行動指標"""
    interaction_count: int
    positive_rate: float
    avg_session_length: float
    genre_diversity: float
    maker_diversity: float
    temporal_patterns: Dict[str, float]
    preference_strength: float

class UserFeatureProcessor(BaseFeatureProcessor):
    """ユーザー特徴量プロセッサー"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        self.user_stats = {}
        self.global_stats = {}
        
    def extract_behavioral_features(self, user_interactions: pd.DataFrame) -> np.ndarray:
        """ユーザー行動特徴量の抽出"""
        features = []
        
        if len(user_interactions) == 0:
            # コールドスタートユーザー向けの既定値
            return np.zeros(20)  # 20次元の行動特徴量
        
        # 基本統計
        total_interactions = len(user_interactions)
        positive_rate = user_interactions.get('liked', []).mean() if 'liked' in user_interactions else 0.5
        
        features.extend([
            total_interactions,
            positive_rate,
            1 - positive_rate,  # negative rate
            np.log1p(total_interactions)  # log transformed count
        ])
        
        # 時間的パターン
        if 'created_at' in user_interactions:
            dates = pd.to_datetime(user_interactions['created_at'])
            
            # 活動期間
            if len(dates) > 1:
                activity_span_days = (dates.max() - dates.min()).days
                avg_days_between_interactions = activity_span_days / max(1, len(dates) - 1)
            else:
                activity_span_days = 0
                avg_days_between_interactions = 0
            
            features.extend([
                activity_span_days,
                avg_days_between_interactions,
                len(dates.dt.hour.unique()) / 24,  # time diversity
                len(dates.dt.dayofweek.unique()) / 7,  # day diversity
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # ジャンル多様性
        if 'genre' in user_interactions:
            genre_counts = user_interactions['genre'].value_counts()
            genre_entropy = -np.sum(genre_counts / genre_counts.sum() * 
                                  np.log2(genre_counts / genre_counts.sum() + 1e-8))
            genre_coverage = len(genre_counts) / max(1, user_interactions['genre'].nunique())
            features.extend([genre_entropy, genre_coverage])
        else:
            features.extend([0, 0])
        
        # メーカー多様性
        if 'maker' in user_interactions:
            maker_counts = user_interactions['maker'].value_counts()
            maker_entropy = -np.sum(maker_counts / maker_counts.sum() * 
                                  np.log2(maker_counts / maker_counts.sum() + 1e-8))
            maker_coverage = len(maker_counts) / max(1, user_interactions['maker'].nunique())
            features.extend([maker_entropy, maker_coverage])
        else:
            features.extend([0, 0])
        
        # 価格帯の好み
        if 'price' in user_interactions:
            price_mean = user_interactions['price'].mean()
            price_std = user_interactions['price'].std()
            price_min = user_interactions['price'].min()
            price_max = user_interactions['price'].max()
            features.extend([price_mean, price_std, price_min, price_max])
        else:
            features.extend([0, 0, 0, 0])
        
        # 再生時間の好み
        if 'duration_seconds' in user_interactions:
            duration_mean = user_interactions['duration_seconds'].mean() / 60  # 分単位
            duration_std = user_interactions['duration_seconds'].std() / 60
            features.extend([duration_mean, duration_std])
        else:
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_preference_embeddings(self, user_interactions: pd.DataFrame) -> np.ndarray:
        """ユーザー嗜好埋め込みの抽出"""
        # 簡単な嗜好ベクトル（実際のプロジェクトではより高度な手法を使用）
        embedding_dim = 64
        
        if len(user_interactions) == 0:
            return np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
        
        # ジャンル嗜好ベクトル
        genre_embedding = np.zeros(32)
        if 'genre' in user_interactions:
            genre_counts = user_interactions['genre'].value_counts()
            # 上位32ジャンルの重みを使用（簡単な実装）
            for i, (genre, count) in enumerate(genre_counts.head(32).items()):
                if i < 32:
                    genre_embedding[i] = count / len(user_interactions)
        
        # メーカー嗜好ベクトル
        maker_embedding = np.zeros(32)
        if 'maker' in user_interactions:
            maker_counts = user_interactions['maker'].value_counts()
            for i, (maker, count) in enumerate(maker_counts.head(32).items()):
                if i < 32:
                    maker_embedding[i] = count / len(user_interactions)
        
        preference_embedding = np.concatenate([genre_embedding, maker_embedding])
        
        # L2正規化
        norm = np.linalg.norm(preference_embedding)
        if norm > 0:
            preference_embedding = preference_embedding / norm
        
        return preference_embedding.astype(np.float32)
    
    def extract_demographic_features(self, user_data: pd.Series) -> np.ndarray:
        """ユーザー属性特徴量の抽出"""
        features = []
        
        # 年齢関連（仮想的な属性）
        age = user_data.get('age', 30)  # デフォルト年齢
        features.extend([
            age,
            age ** 2,  # age squared
            1 if age < 25 else 0,  # young flag
            1 if 25 <= age < 40 else 0,  # middle flag
            1 if age >= 40 else 0,  # senior flag
        ])
        
        # アカウント作成からの日数
        if 'created_at' in user_data and pd.notna(user_data['created_at']):
            created_date = pd.to_datetime(user_data['created_at'])
            days_since_creation = (datetime.now() - created_date).days
            features.append(days_since_creation)
            features.append(np.log1p(days_since_creation))
        else:
            features.extend([0, 0])
        
        # プレミアム会員フラグ（仮想的）
        is_premium = user_data.get('is_premium', False)
        features.append(1 if is_premium else 0)
        
        return np.array(features, dtype=np.float32)
    
    def fit(self, user_data: pd.DataFrame, interaction_data: pd.DataFrame, target: Optional[np.ndarray] = None):
        """ユーザー特徴量プロセッサーの学習"""
        logger.info(f"Fitting user feature processor on {len(user_data)} users")
        
        # グローバル統計の計算
        if len(interaction_data) > 0:
            self.global_stats = {
                'avg_interactions_per_user': len(interaction_data) / len(user_data),
                'global_positive_rate': interaction_data.get('liked', []).mean() if 'liked' in interaction_data else 0.5,
                'most_common_genre': interaction_data['genre'].mode().iloc[0] if 'genre' in interaction_data else 'unknown',
                'most_common_maker': interaction_data['maker'].mode().iloc[0] if 'maker' in interaction_data else 'unknown',
                'avg_price': interaction_data['price'].mean() if 'price' in interaction_data else 0,
                'avg_duration': interaction_data['duration_seconds'].mean() if 'duration_seconds' in interaction_data else 0
            }
        
        # ユーザー別統計の計算
        for user_id in user_data.index:
            user_interactions = interaction_data[interaction_data['user_id'] == user_id] if 'user_id' in interaction_data else pd.DataFrame()
            
            behavioral_features = self.extract_behavioral_features(user_interactions)
            preference_embeddings = self.extract_preference_embeddings(user_interactions)
            demographic_features = self.extract_demographic_features(user_data.loc[user_id])
            
            self.user_stats[user_id] = {
                'behavioral_features': behavioral_features,
                'preference_embeddings': preference_embeddings,
                'demographic_features': demographic_features
            }
        
        self.is_fitted = True
        logger.info("User feature processor fitted successfully")
    
    def transform(self, user_data: pd.DataFrame, interaction_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """ユーザー特徴量の変換"""
        if not self.is_fitted:
            raise ValueError("User feature processor must be fitted before transform")
        
        user_features = []
        
        for user_id in user_data.index:
            if interaction_data is not None and len(interaction_data) > 0:
                user_interactions = interaction_data[interaction_data['user_id'] == user_id] if 'user_id' in interaction_data else pd.DataFrame()
            else:
                user_interactions = pd.DataFrame()
            
            # 特徴量抽出
            behavioral_features = self.extract_behavioral_features(user_interactions)
            preference_embeddings = self.extract_preference_embeddings(user_interactions)
            demographic_features = self.extract_demographic_features(user_data.loc[user_id])
            
            # 特徴量結合
            combined_features = np.concatenate([
                behavioral_features,
                preference_embeddings,
                demographic_features
            ])
            
            user_features.append(combined_features)
        
        result = np.array(user_features, dtype=np.float32)
        logger.debug(f"Transformed {len(user_data)} users to shape {result.shape}")
        
        return result
    
    def get_user_behavior_metrics(self, user_id: int, interaction_data: pd.DataFrame) -> UserBehaviorMetrics:
        """ユーザー行動指標の取得"""
        user_interactions = interaction_data[interaction_data['user_id'] == user_id] if 'user_id' in interaction_data else pd.DataFrame()
        
        if len(user_interactions) == 0:
            return UserBehaviorMetrics(
                interaction_count=0,
                positive_rate=0.5,
                avg_session_length=0,
                genre_diversity=0,
                maker_diversity=0,
                temporal_patterns={},
                preference_strength=0
            )
        
        # 各指標の計算
        interaction_count = len(user_interactions)
        positive_rate = user_interactions.get('liked', []).mean() if 'liked' in user_interactions else 0.5
        
        # セッション長（仮想的な計算）
        avg_session_length = interaction_count / max(1, user_interactions.groupby(user_interactions['created_at'].dt.date).size().mean() if 'created_at' in user_interactions else 1)
        
        # 多様性指標
        genre_diversity = len(user_interactions['genre'].unique()) / max(1, len(user_interactions)) if 'genre' in user_interactions else 0
        maker_diversity = len(user_interactions['maker'].unique()) / max(1, len(user_interactions)) if 'maker' in user_interactions else 0
        
        # 時間パターン（簡略化）
        temporal_patterns = {}
        if 'created_at' in user_interactions:
            hours = pd.to_datetime(user_interactions['created_at']).dt.hour
            temporal_patterns['peak_hour'] = hours.mode().iloc[0] if len(hours.mode()) > 0 else 12
            temporal_patterns['activity_spread'] = hours.std() if len(hours) > 1 else 0
        
        # 嗜好強度（positive rateからの逸脱）
        preference_strength = abs(positive_rate - 0.5) * 2
        
        return UserBehaviorMetrics(
            interaction_count=interaction_count,
            positive_rate=positive_rate,
            avg_session_length=avg_session_length,
            genre_diversity=genre_diversity,
            maker_diversity=maker_diversity,
            temporal_patterns=temporal_patterns,
            preference_strength=preference_strength
        )