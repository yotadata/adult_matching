"""
Embedding Management System

埋め込みベクトル管理システム
ユーザーとアイテムの埋め込みベクトルの生成・管理・更新を統合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import json

from backend.ml.utils.logger import get_ml_logger
from backend.ml.preprocessing.utils.data_validator import DataValidator

logger = get_ml_logger(__name__)

@dataclass
class EmbeddingConfig:
    """埋め込み設定"""
    user_embedding_dim: int = 768
    item_embedding_dim: int = 768
    update_frequency: str = "daily"  # "daily", "hourly", "batch"
    decay_factor: float = 0.95
    min_interactions: int = 5
    cold_start_strategy: str = "random"  # "random", "global_mean", "zero"
    similarity_threshold: float = 0.7
    max_batch_size: int = 1000

@dataclass
class EmbeddingStats:
    """埋め込み統計情報"""
    total_user_embeddings: int
    total_item_embeddings: int
    avg_user_similarity: float
    avg_item_similarity: float
    cold_start_users: int
    cold_start_items: int
    last_update_time: datetime
    memory_usage_mb: float

class EmbeddingManager:
    """埋め込み管理システム"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.embedding_stats = None
        self.validator = DataValidator()
        self.is_initialized = False
        
        # 更新履歴
        self.update_history = {
            'user_updates': {},
            'item_updates': {},
            'global_updates': []
        }
        
    def initialize_embeddings(self, 
                            user_data: pd.DataFrame, 
                            item_data: pd.DataFrame,
                            interaction_data: Optional[pd.DataFrame] = None):
        """埋め込みの初期化"""
        logger.info("Initializing embedding system")
        
        # データ検証
        self.validator.validate_input_data(user_data)
        self.validator.validate_input_data(item_data)
        
        # ユーザー埋め込みの初期化
        self._initialize_user_embeddings(user_data, interaction_data)
        
        # アイテム埋め込みの初期化
        self._initialize_item_embeddings(item_data, interaction_data)
        
        # 統計情報の計算
        self._update_embedding_stats()
        
        self.is_initialized = True
        logger.info(f"Embedding system initialized - Users: {len(self.user_embeddings)}, Items: {len(self.item_embeddings)}")
        
    def _initialize_user_embeddings(self, 
                                  user_data: pd.DataFrame, 
                                  interaction_data: Optional[pd.DataFrame] = None):
        """ユーザー埋め込みの初期化"""
        for user_id in user_data.index:
            if interaction_data is not None and 'user_id' in interaction_data:
                user_interactions = interaction_data[interaction_data['user_id'] == user_id]
                
                if len(user_interactions) >= self.config.min_interactions:
                    # インタラクションベースの埋め込み生成
                    embedding = self._generate_user_embedding_from_interactions(user_interactions)
                else:
                    # コールドスタート戦略
                    embedding = self._generate_cold_start_embedding('user')
            else:
                embedding = self._generate_cold_start_embedding('user')
            
            self.user_embeddings[user_id] = {
                'embedding': embedding,
                'last_update': datetime.now(),
                'interaction_count': len(user_interactions) if interaction_data is not None else 0,
                'is_cold_start': len(user_interactions) < self.config.min_interactions if interaction_data is not None else True
            }
    
    def _initialize_item_embeddings(self, 
                                  item_data: pd.DataFrame, 
                                  interaction_data: Optional[pd.DataFrame] = None):
        """アイテム埋め込みの初期化"""
        for item_id in item_data.index:
            item_row = item_data.loc[item_id]
            
            if interaction_data is not None and 'item_id' in interaction_data:
                item_interactions = interaction_data[interaction_data['item_id'] == item_id]
                
                if len(item_interactions) >= self.config.min_interactions:
                    # インタラクションベースの埋め込み生成
                    embedding = self._generate_item_embedding_from_interactions(item_row, item_interactions)
                else:
                    # コンテンツベースの埋め込み生成
                    embedding = self._generate_item_embedding_from_content(item_row)
            else:
                embedding = self._generate_item_embedding_from_content(item_row)
            
            self.item_embeddings[item_id] = {
                'embedding': embedding,
                'last_update': datetime.now(),
                'interaction_count': len(item_interactions) if interaction_data is not None else 0,
                'is_cold_start': len(item_interactions) < self.config.min_interactions if interaction_data is not None else True
            }
    
    def _generate_user_embedding_from_interactions(self, interactions: pd.DataFrame) -> np.ndarray:
        """インタラクションからユーザー埋め込みを生成"""
        embedding = np.zeros(self.config.user_embedding_dim, dtype=np.float32)
        
        if len(interactions) == 0:
            return self._generate_cold_start_embedding('user')
        
        # ジャンル嗜好ベクトル
        if 'genre' in interactions:
            genre_counts = interactions['genre'].value_counts()
            # 上位ジャンルの重みを埋め込みに反映
            for i, (genre, count) in enumerate(genre_counts.head(20).items()):
                if i < self.config.user_embedding_dim // 4:
                    embedding[i] = count / len(interactions)
        
        # メーカー嗜好ベクトル
        if 'maker' in interactions:
            maker_counts = interactions['maker'].value_counts()
            offset = self.config.user_embedding_dim // 4
            for i, (maker, count) in enumerate(maker_counts.head(20).items()):
                if i + offset < self.config.user_embedding_dim // 2:
                    embedding[i + offset] = count / len(interactions)
        
        # 行動パターンベクトル
        offset = self.config.user_embedding_dim // 2
        positive_rate = interactions.get('liked', []).mean() if 'liked' in interactions else 0.5
        total_interactions = len(interactions)
        
        # 行動特徴量を埋め込みに組み込み
        behavioral_features = [
            positive_rate,
            1 - positive_rate,
            np.log1p(total_interactions) / 10,
            len(interactions['genre'].unique()) / max(1, len(interactions)) if 'genre' in interactions else 0
        ]
        
        feature_dim = min(len(behavioral_features), self.config.user_embedding_dim - offset)
        embedding[offset:offset + feature_dim] = behavioral_features[:feature_dim]
        
        # L2正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _generate_item_embedding_from_interactions(self, item: pd.Series, interactions: pd.DataFrame) -> np.ndarray:
        """インタラクションからアイテム埋め込みを生成"""
        # コンテンツベースの埋め込みから開始
        embedding = self._generate_item_embedding_from_content(item)
        
        if len(interactions) == 0:
            return embedding
        
        # インタラクション情報で調整
        positive_rate = interactions.get('liked', []).mean() if 'liked' in interactions else 0.5
        total_interactions = len(interactions)
        
        # 人気度を埋め込みに反映
        popularity_boost = np.log1p(total_interactions) / 100
        quality_boost = positive_rate * 0.1
        
        # 埋め込みの調整
        embedding = embedding * (1 + popularity_boost + quality_boost)
        
        # 正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _generate_item_embedding_from_content(self, item: pd.Series) -> np.ndarray:
        """コンテンツからアイテム埋め込みを生成"""
        embedding = np.zeros(self.config.item_embedding_dim, dtype=np.float32)
        
        # ジャンルベース埋め込み
        genre = str(item.get('genre', 'unknown'))
        genre_hash = hash(genre) % (self.config.item_embedding_dim // 4)
        embedding[genre_hash] = 1.0
        
        # メーカーベース埋め込み
        maker = str(item.get('maker', 'unknown'))
        maker_hash = hash(maker) % (self.config.item_embedding_dim // 4)
        offset = self.config.item_embedding_dim // 4
        embedding[offset + maker_hash] = 1.0
        
        # 数値特徴量の埋め込み
        offset = self.config.item_embedding_dim // 2
        price = item.get('price', 0)
        duration = item.get('duration_seconds', 0) / 3600  # 時間単位
        rating = item.get('rating', 0)
        
        numerical_features = [
            np.tanh(price / 10000),  # 価格正規化
            np.tanh(duration),       # 時間正規化
            rating / 5.0,            # 評価正規化
            1.0 if price > 0 else 0.0,  # 有料フラグ
        ]
        
        feature_dim = min(len(numerical_features), self.config.item_embedding_dim - offset)
        embedding[offset:offset + feature_dim] = numerical_features[:feature_dim]
        
        # L2正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _generate_cold_start_embedding(self, entity_type: str) -> np.ndarray:
        """コールドスタート埋め込みの生成"""
        dim = self.config.user_embedding_dim if entity_type == 'user' else self.config.item_embedding_dim
        
        if self.config.cold_start_strategy == "random":
            embedding = np.random.normal(0, 0.1, dim).astype(np.float32)
            # 正規化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
            
        elif self.config.cold_start_strategy == "zero":
            return np.zeros(dim, dtype=np.float32)
            
        elif self.config.cold_start_strategy == "global_mean":
            # 既存の埋め込みの平均を使用
            embeddings = self.user_embeddings if entity_type == 'user' else self.item_embeddings
            if embeddings:
                all_embeddings = [data['embedding'] for data in embeddings.values()]
                return np.mean(all_embeddings, axis=0).astype(np.float32)
            else:
                return np.random.normal(0, 0.1, dim).astype(np.float32)
        
        return np.zeros(dim, dtype=np.float32)
    
    def update_user_embedding(self, 
                            user_id: int, 
                            new_interactions: pd.DataFrame,
                            incremental: bool = True):
        """ユーザー埋め込みの更新"""
        if not self.is_initialized:
            raise ValueError("Embedding manager must be initialized first")
        
        if user_id not in self.user_embeddings:
            # 新規ユーザー
            self.user_embeddings[user_id] = {
                'embedding': self._generate_cold_start_embedding('user'),
                'last_update': datetime.now(),
                'interaction_count': 0,
                'is_cold_start': True
            }
        
        current_data = self.user_embeddings[user_id]
        
        if incremental and not current_data['is_cold_start']:
            # インクリメンタル更新
            old_embedding = current_data['embedding']
            new_embedding = self._generate_user_embedding_from_interactions(new_interactions)
            
            # 減衰係数を使用して更新
            updated_embedding = (
                old_embedding * self.config.decay_factor + 
                new_embedding * (1 - self.config.decay_factor)
            )
            
            # 正規化
            norm = np.linalg.norm(updated_embedding)
            if norm > 0:
                updated_embedding = updated_embedding / norm
                
        else:
            # 全体更新
            updated_embedding = self._generate_user_embedding_from_interactions(new_interactions)
        
        # 更新の記録
        self.user_embeddings[user_id] = {
            'embedding': updated_embedding,
            'last_update': datetime.now(),
            'interaction_count': current_data['interaction_count'] + len(new_interactions),
            'is_cold_start': len(new_interactions) < self.config.min_interactions
        }
        
        self.update_history['user_updates'][user_id] = datetime.now()
        
        logger.debug(f"Updated user embedding for user {user_id}")
    
    def update_item_embedding(self, 
                            item_id: int, 
                            item_data: pd.Series,
                            new_interactions: Optional[pd.DataFrame] = None,
                            incremental: bool = True):
        """アイテム埋め込みの更新"""
        if not self.is_initialized:
            raise ValueError("Embedding manager must be initialized first")
        
        if item_id not in self.item_embeddings:
            # 新規アイテム
            embedding = self._generate_item_embedding_from_content(item_data)
            self.item_embeddings[item_id] = {
                'embedding': embedding,
                'last_update': datetime.now(),
                'interaction_count': 0,
                'is_cold_start': True
            }
            return
        
        current_data = self.item_embeddings[item_id]
        
        if new_interactions is not None and len(new_interactions) > 0:
            updated_embedding = self._generate_item_embedding_from_interactions(item_data, new_interactions)
            
            if incremental and not current_data['is_cold_start']:
                # インクリメンタル更新
                old_embedding = current_data['embedding']
                updated_embedding = (
                    old_embedding * self.config.decay_factor + 
                    updated_embedding * (1 - self.config.decay_factor)
                )
                
                # 正規化
                norm = np.linalg.norm(updated_embedding)
                if norm > 0:
                    updated_embedding = updated_embedding / norm
        else:
            # コンテンツのみの更新
            updated_embedding = self._generate_item_embedding_from_content(item_data)
        
        # 更新の記録
        self.item_embeddings[item_id] = {
            'embedding': updated_embedding,
            'last_update': datetime.now(),
            'interaction_count': current_data['interaction_count'] + (len(new_interactions) if new_interactions is not None else 0),
            'is_cold_start': (current_data['interaction_count'] + (len(new_interactions) if new_interactions is not None else 0)) < self.config.min_interactions
        }
        
        self.update_history['item_updates'][item_id] = datetime.now()
        
        logger.debug(f"Updated item embedding for item {item_id}")
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """ユーザー埋め込みの取得"""
        if user_id not in self.user_embeddings:
            return self._generate_cold_start_embedding('user')
        
        return self.user_embeddings[user_id]['embedding'].copy()
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """アイテム埋め込みの取得"""
        if item_id not in self.item_embeddings:
            return self._generate_cold_start_embedding('item')
        
        return self.item_embeddings[item_id]['embedding'].copy()
    
    def get_similar_users(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """類似ユーザーの取得"""
        if user_id not in self.user_embeddings:
            return []
        
        user_embedding = self.user_embeddings[user_id]['embedding']
        similarities = []
        
        for other_id, other_data in self.user_embeddings.items():
            if other_id != user_id:
                similarity = np.dot(user_embedding, other_data['embedding'])
                similarities.append((other_id, float(similarity)))
        
        # 類似度の降順でソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """類似アイテムの取得"""
        if item_id not in self.item_embeddings:
            return []
        
        item_embedding = self.item_embeddings[item_id]['embedding']
        similarities = []
        
        for other_id, other_data in self.item_embeddings.items():
            if other_id != item_id:
                similarity = np.dot(item_embedding, other_data['embedding'])
                similarities.append((other_id, float(similarity)))
        
        # 類似度の降順でソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def compute_user_item_similarity(self, user_id: int, item_id: int) -> float:
        """ユーザー・アイテム類似度の計算"""
        user_embedding = self.get_user_embedding(user_id)
        item_embedding = self.get_item_embedding(item_id)
        
        return float(np.dot(user_embedding, item_embedding))
    
    def _update_embedding_stats(self):
        """埋め込み統計情報の更新"""
        user_embeddings_array = np.array([data['embedding'] for data in self.user_embeddings.values()])
        item_embeddings_array = np.array([data['embedding'] for data in self.item_embeddings.values()])
        
        # ユーザー間平均類似度
        if len(user_embeddings_array) > 1:
            user_similarities = np.dot(user_embeddings_array, user_embeddings_array.T)
            avg_user_similarity = (user_similarities.sum() - np.trace(user_similarities)) / (len(user_embeddings_array) * (len(user_embeddings_array) - 1))
        else:
            avg_user_similarity = 0.0
        
        # アイテム間平均類似度
        if len(item_embeddings_array) > 1:
            item_similarities = np.dot(item_embeddings_array, item_embeddings_array.T)
            avg_item_similarity = (item_similarities.sum() - np.trace(item_similarities)) / (len(item_embeddings_array) * (len(item_embeddings_array) - 1))
        else:
            avg_item_similarity = 0.0
        
        # コールドスタートユーザー・アイテム数
        cold_start_users = sum(1 for data in self.user_embeddings.values() if data['is_cold_start'])
        cold_start_items = sum(1 for data in self.item_embeddings.values() if data['is_cold_start'])
        
        # メモリ使用量の計算（概算）
        memory_usage = (
            len(user_embeddings_array) * self.config.user_embedding_dim * 4 +  # float32
            len(item_embeddings_array) * self.config.item_embedding_dim * 4
        ) / 1024 / 1024  # MB
        
        self.embedding_stats = EmbeddingStats(
            total_user_embeddings=len(self.user_embeddings),
            total_item_embeddings=len(self.item_embeddings),
            avg_user_similarity=float(avg_user_similarity),
            avg_item_similarity=float(avg_item_similarity),
            cold_start_users=cold_start_users,
            cold_start_items=cold_start_items,
            last_update_time=datetime.now(),
            memory_usage_mb=memory_usage
        )
    
    def save_embeddings(self, save_path: Union[str, Path]):
        """埋め込みの保存"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'config': self.config,
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings,
            'embedding_stats': self.embedding_stats,
            'update_history': self.update_history,
            'is_initialized': self.is_initialized,
            'save_timestamp': datetime.now()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path: Union[str, Path]):
        """埋め込みの読み込み"""
        load_path = Path(load_path)
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.user_embeddings = save_data['user_embeddings']
        self.item_embeddings = save_data['item_embeddings']
        self.embedding_stats = save_data['embedding_stats']
        self.update_history = save_data['update_history']
        self.is_initialized = save_data['is_initialized']
        
        logger.info(f"Embeddings loaded from {load_path}")
    
    def get_embedding_summary(self) -> Dict[str, Any]:
        """埋め込み概要の取得"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        self._update_embedding_stats()
        
        return {
            'status': 'initialized',
            'stats': {
                'total_users': self.embedding_stats.total_user_embeddings,
                'total_items': self.embedding_stats.total_item_embeddings,
                'cold_start_users': self.embedding_stats.cold_start_users,
                'cold_start_items': self.embedding_stats.cold_start_items,
                'avg_user_similarity': self.embedding_stats.avg_user_similarity,
                'avg_item_similarity': self.embedding_stats.avg_item_similarity,
                'memory_usage_mb': self.embedding_stats.memory_usage_mb,
                'last_update': self.embedding_stats.last_update_time.isoformat()
            },
            'config': {
                'user_embedding_dim': self.config.user_embedding_dim,
                'item_embedding_dim': self.config.item_embedding_dim,
                'min_interactions': self.config.min_interactions,
                'cold_start_strategy': self.config.cold_start_strategy,
                'decay_factor': self.config.decay_factor
            }
        }