"""
Item Feature Processor

アイテム特徴量専用プロセッサー
既存のEnhancedItemFeatureProcessorを統合し、新しいMLパッケージ構造に対応
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
class ItemMetrics:
    """アイテム指標"""
    popularity_score: float
    quality_score: float
    engagement_rate: float
    genre_popularity: float
    maker_reputation: float
    price_competitiveness: float
    release_recency: float

class ItemFeatureProcessor(BaseFeatureProcessor):
    """アイテム特徴量プロセッサー"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        self.item_stats = {}
        self.global_item_stats = {}
        
    def extract_content_features(self, item_data: pd.DataFrame) -> np.ndarray:
        """コンテンツ特徴量の抽出"""
        features = []
        
        for idx, item in item_data.iterrows():
            item_features = []
            
            # 基本情報
            price = item.get('price', 0)
            duration = item.get('duration_seconds', 0) / 60  # 分単位
            
            item_features.extend([
                price,
                np.log1p(price),  # log価格
                duration,
                np.log1p(duration),  # log再生時間
            ])
            
            # リリース日からの経過日数
            if 'release_date' in item and pd.notna(item['release_date']):
                release_date = pd.to_datetime(item['release_date'])
                days_since_release = (datetime.now() - release_date).days
                item_features.extend([
                    days_since_release,
                    np.log1p(days_since_release),
                    1 if days_since_release <= 30 else 0,  # 新作フラグ
                    1 if days_since_release <= 90 else 0,  # 最近作フラグ
                ])
            else:
                item_features.extend([0, 0, 0, 0])
            
            # ジャンル特徴量（ワンホット）
            genre = item.get('genre', 'unknown')
            common_genres = ['drama', 'romance', 'action', 'comedy', 'horror', 'documentary']
            genre_features = [1 if genre.lower() == g else 0 for g in common_genres]
            item_features.extend(genre_features)
            
            # メーカー特徴量（簡略化）
            maker = item.get('maker', 'unknown')
            is_major_maker = 1 if maker in self.global_item_stats.get('top_makers', []) else 0
            item_features.append(is_major_maker)
            
            # 評価特徴量
            rating = item.get('rating', 0)
            rating_count = item.get('rating_count', 0)
            item_features.extend([
                rating,
                rating_count,
                np.log1p(rating_count),
                rating * np.log1p(rating_count),  # 重み付け評価
            ])
            
            # タイトル長
            title_length = len(str(item.get('title', '')))
            item_features.extend([
                title_length,
                np.log1p(title_length)
            ])
            
            features.append(item_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_popularity_features(self, item_data: pd.DataFrame, interaction_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """人気度特徴量の抽出"""
        features = []
        
        for idx, item in item_data.iterrows():
            item_id = item.get('id') or idx
            popularity_features = []
            
            if interaction_data is not None and 'item_id' in interaction_data:
                # アイテムとのインタラクション統計
                item_interactions = interaction_data[interaction_data['item_id'] == item_id]
                
                total_interactions = len(item_interactions)
                positive_interactions = item_interactions.get('liked', []).sum() if 'liked' in item_interactions else 0
                positive_rate = positive_interactions / max(1, total_interactions)
                
                popularity_features.extend([
                    total_interactions,
                    np.log1p(total_interactions),
                    positive_rate,
                    positive_interactions,
                ])
                
                # 時間的人気度パターン
                if 'created_at' in item_interactions and len(item_interactions) > 0:
                    recent_interactions = item_interactions[
                        pd.to_datetime(item_interactions['created_at']) >= (datetime.now() - timedelta(days=7))
                    ]
                    recent_interaction_count = len(recent_interactions)
                    recent_popularity = recent_interaction_count / max(1, total_interactions)
                    
                    popularity_features.extend([
                        recent_interaction_count,
                        recent_popularity
                    ])
                else:
                    popularity_features.extend([0, 0])
                
            else:
                # インタラクションデータがない場合はゼロ埋め
                popularity_features.extend([0, 0, 0, 0, 0, 0])
            
            # グローバル人気度指標
            genre = item.get('genre', 'unknown')
            maker = item.get('maker', 'unknown')
            
            genre_popularity = self.global_item_stats.get('genre_popularity', {}).get(genre, 0)
            maker_popularity = self.global_item_stats.get('maker_popularity', {}).get(maker, 0)
            
            popularity_features.extend([
                genre_popularity,
                maker_popularity
            ])
            
            features.append(popularity_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_embedding_features(self, item_data: pd.DataFrame) -> np.ndarray:
        """アイテム埋め込み特徴量の抽出"""
        embedding_dim = 64
        features = []
        
        for idx, item in item_data.iterrows():
            # 簡単な特徴量ベースの埋め込み生成
            # 実際のプロジェクトではより高度な手法を使用
            
            # ジャンルベース埋め込み
            genre = item.get('genre', 'unknown')
            genre_embedding = np.zeros(32)
            genre_hash = hash(genre) % 32
            genre_embedding[genre_hash] = 1.0
            
            # メーカーベース埋め込み
            maker = item.get('maker', 'unknown')
            maker_embedding = np.zeros(32)
            maker_hash = hash(maker) % 32
            maker_embedding[maker_hash] = 1.0
            
            # 特徴量の結合と正規化
            item_embedding = np.concatenate([genre_embedding, maker_embedding])
            
            # L2正規化
            norm = np.linalg.norm(item_embedding)
            if norm > 0:
                item_embedding = item_embedding / norm
            
            features.append(item_embedding)
        
        return np.array(features, dtype=np.float32)
    
    def fit(self, item_data: pd.DataFrame, interaction_data: Optional[pd.DataFrame] = None, target: Optional[np.ndarray] = None):
        """アイテム特徴量プロセッサーの学習"""
        logger.info(f"Fitting item feature processor on {len(item_data)} items")
        
        # グローバル統計の計算
        if interaction_data is not None and len(interaction_data) > 0:
            # 人気ジャンル・メーカーの特定
            genre_interactions = interaction_data.groupby('genre').size() if 'genre' in interaction_data else pd.Series()
            maker_interactions = interaction_data.groupby('maker').size() if 'maker' in interaction_data else pd.Series()
            
            total_interactions = len(interaction_data)
            
            self.global_item_stats = {
                'total_items': len(item_data),
                'avg_interactions_per_item': total_interactions / len(item_data),
                'genre_popularity': (genre_interactions / total_interactions).to_dict() if len(genre_interactions) > 0 else {},
                'maker_popularity': (maker_interactions / total_interactions).to_dict() if len(maker_interactions) > 0 else {},
                'top_genres': genre_interactions.head(10).index.tolist() if len(genre_interactions) > 0 else [],
                'top_makers': maker_interactions.head(20).index.tolist() if len(maker_interactions) > 0 else [],
                'avg_price': item_data['price'].mean() if 'price' in item_data else 0,
                'avg_duration': item_data['duration_seconds'].mean() if 'duration_seconds' in item_data else 0,
                'avg_rating': item_data['rating'].mean() if 'rating' in item_data else 0
            }
        else:
            self.global_item_stats = {
                'total_items': len(item_data),
                'avg_interactions_per_item': 0,
                'genre_popularity': {},
                'maker_popularity': {},
                'top_genres': [],
                'top_makers': [],
                'avg_price': item_data['price'].mean() if 'price' in item_data else 0,
                'avg_duration': item_data['duration_seconds'].mean() if 'duration_seconds' in item_data else 0,
                'avg_rating': item_data['rating'].mean() if 'rating' in item_data else 0
            }
        
        # アイテム別統計の計算
        for idx, item in item_data.iterrows():
            item_id = item.get('id') or idx
            
            content_features = self.extract_content_features(item_data.loc[idx:idx])
            popularity_features = self.extract_popularity_features(item_data.loc[idx:idx], interaction_data)
            embedding_features = self.extract_embedding_features(item_data.loc[idx:idx])
            
            self.item_stats[item_id] = {
                'content_features': content_features.flatten(),
                'popularity_features': popularity_features.flatten(),
                'embedding_features': embedding_features.flatten()
            }
        
        self.is_fitted = True
        logger.info("Item feature processor fitted successfully")
    
    def transform(self, item_data: pd.DataFrame, interaction_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """アイテム特徴量の変換"""
        if not self.is_fitted:
            raise ValueError("Item feature processor must be fitted before transform")
        
        item_features = []
        
        # 各特徴量タイプの抽出
        content_features = self.extract_content_features(item_data)
        popularity_features = self.extract_popularity_features(item_data, interaction_data)
        embedding_features = self.extract_embedding_features(item_data)
        
        # 特徴量の結合
        for i in range(len(item_data)):
            combined_features = np.concatenate([
                content_features[i],
                popularity_features[i],
                embedding_features[i]
            ])
            item_features.append(combined_features)
        
        result = np.array(item_features, dtype=np.float32)
        logger.debug(f"Transformed {len(item_data)} items to shape {result.shape}")
        
        return result
    
    def get_item_metrics(self, item_id: int, item_data: pd.DataFrame, interaction_data: Optional[pd.DataFrame] = None) -> ItemMetrics:
        """アイテム指標の取得"""
        item_row = item_data[item_data['id'] == item_id] if 'id' in item_data else item_data.iloc[[item_id]]
        
        if len(item_row) == 0:
            return ItemMetrics(
                popularity_score=0.0,
                quality_score=0.0,
                engagement_rate=0.0,
                genre_popularity=0.0,
                maker_reputation=0.0,
                price_competitiveness=0.0,
                release_recency=0.0
            )
        
        item = item_row.iloc[0]
        
        # 人気度スコア
        if interaction_data is not None and 'item_id' in interaction_data:
            item_interactions = interaction_data[interaction_data['item_id'] == item_id]
            total_interactions = len(item_interactions)
            positive_interactions = item_interactions.get('liked', []).sum() if 'liked' in item_interactions else 0
            popularity_score = np.log1p(total_interactions) / 10  # 正規化
            engagement_rate = positive_interactions / max(1, total_interactions)
        else:
            popularity_score = 0.0
            engagement_rate = 0.0
        
        # 品質スコア
        rating = item.get('rating', 0)
        rating_count = item.get('rating_count', 0)
        quality_score = rating * (np.log1p(rating_count) / 10)  # 重み付け品質
        
        # ジャンル人気度
        genre = item.get('genre', 'unknown')
        genre_popularity = self.global_item_stats.get('genre_popularity', {}).get(genre, 0)
        
        # メーカー評判
        maker = item.get('maker', 'unknown')
        maker_reputation = self.global_item_stats.get('maker_popularity', {}).get(maker, 0)
        
        # 価格競争力
        price = item.get('price', 0)
        avg_price = self.global_item_stats.get('avg_price', 1)
        price_competitiveness = max(0, 2 - (price / max(avg_price, 1)))  # 平均より安いほど高スコア
        
        # リリース新しさ
        if 'release_date' in item and pd.notna(item['release_date']):
            release_date = pd.to_datetime(item['release_date'])
            days_since_release = (datetime.now() - release_date).days
            release_recency = max(0, 1 - (days_since_release / 365))  # 1年以内は高スコア
        else:
            release_recency = 0.0
        
        return ItemMetrics(
            popularity_score=popularity_score,
            quality_score=quality_score,
            engagement_rate=engagement_rate,
            genre_popularity=genre_popularity,
            maker_reputation=maker_reputation,
            price_competitiveness=price_competitiveness,
            release_recency=release_recency
        )