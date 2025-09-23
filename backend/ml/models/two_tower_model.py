"""
Two Tower Model Implementation

Two-Towerモデルの実装
"""

import tensorflow as tf
from typing import Dict, Any, List, Tuple
import numpy as np


class TwoTowerModel(tf.keras.Model):
    """Two-Towerモデルクラス"""
    
    def __init__(
        self,
        user_embedding_dim: int = 768,
        item_embedding_dim: int = 768,
        user_hidden_units: List[int] = [512, 256, 128],
        item_hidden_units: List[int] = [512, 256, 128],
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.user_hidden_units = user_hidden_units
        self.item_hidden_units = item_hidden_units
        self.dropout_rate = dropout_rate
        
        # User Tower
        self.user_tower = self._build_tower(user_hidden_units, user_embedding_dim, "user")
        
        # Item Tower
        self.item_tower = self._build_tower(item_hidden_units, item_embedding_dim, "item")
        
        # Similarity computation
        self.similarity = tf.keras.layers.Dot(axes=1, normalize=True)
        
        # Final prediction
        self.prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')
    
    def _build_tower(self, hidden_units: List[int], embedding_dim: int, name: str) -> tf.keras.Model:
        """タワー構築"""
        inputs = tf.keras.Input(shape=(None,), name=f"{name}_features")
        x = inputs
        
        for i, units in enumerate(hidden_units):
            x = tf.keras.layers.Dense(
                units, 
                activation='relu', 
                name=f"{name}_dense_{i}"
            )(x)
            x = tf.keras.layers.Dropout(self.dropout_rate, name=f"{name}_dropout_{i}")(x)
        
        # Final embedding layer
        embeddings = tf.keras.layers.Dense(
            embedding_dim, 
            activation=None, 
            name=f"{name}_embedding"
        )(x)
        
        # L2 normalization
        embeddings = tf.keras.utils.normalize(embeddings, axis=1)
        
        return tf.keras.Model(inputs=inputs, outputs=embeddings, name=f"{name}_tower")
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = None) -> tf.Tensor:
        """フォワードパス"""
        user_features = inputs['user_features']
        item_features = inputs['item_features']
        
        # Get embeddings from each tower
        user_embeddings = self.user_tower(user_features, training=training)
        item_embeddings = self.item_tower(item_features, training=training)
        
        # Compute similarity
        similarity_scores = self.similarity([user_embeddings, item_embeddings])
        
        # Final prediction
        predictions = self.prediction(similarity_scores)
        
        return predictions
    
    def get_user_embeddings(self, user_features: tf.Tensor) -> tf.Tensor:
        """ユーザー埋め込み取得"""
        return self.user_tower(user_features, training=False)
    
    def get_item_embeddings(self, item_features: tf.Tensor) -> tf.Tensor:
        """アイテム埋め込み取得"""
        return self.item_tower(item_features, training=False)
    
    def get_config(self) -> Dict[str, Any]:
        """モデル設定取得"""
        config = super().get_config()
        config.update({
            'user_embedding_dim': self.user_embedding_dim,
            'item_embedding_dim': self.item_embedding_dim,
            'user_hidden_units': self.user_hidden_units,
            'item_hidden_units': self.item_hidden_units,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TwoTowerModel':
        """設定からモデル復元"""
        return cls(**config)