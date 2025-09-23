"""
Feature Scaling Utilities

特徴量スケーリングユーティリティ
"""

import numpy as np
from typing import Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
from pathlib import Path

from backend.ml.utils.logger import get_ml_logger

logger = get_ml_logger(__name__)

class FeatureScaler:
    """統一特徴量スケーリングクラス"""
    
    def __init__(self):
        self.scaler = None
        self.is_fitted = False
        self.scaling_method = None
    
    def fit_transform(self, data: np.ndarray, method: str = "standard") -> np.ndarray:
        """スケーリングの学習と変換"""
        self.scaling_method = method
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        elif method == "none":
            self.scaler = None
            self.is_fitted = True
            return data.copy()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        logger.info(f"Feature scaling fitted with method: {method}")
        logger.info(f"Input shape: {data.shape}, Output shape: {scaled_data.shape}")
        
        return scaled_data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """学習済みスケーラーで変換"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        if self.scaler is None:
            return data.copy()
        
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """スケーリングの逆変換"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        if self.scaler is None:
            return data.copy()
        
        return self.scaler.inverse_transform(data)
    
    def save(self, save_path: Union[str, Path]):
        """スケーラーの保存"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'scaling_method': self.scaling_method
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Feature scaler saved to {save_path}")
    
    def load(self, load_path: Union[str, Path]):
        """スケーラーの読み込み"""
        load_path = Path(load_path)
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.scaler = save_data['scaler']
        self.is_fitted = save_data['is_fitted']
        self.scaling_method = save_data['scaling_method']
        
        logger.info(f"Feature scaler loaded from {load_path}")
    
    def get_scaling_stats(self) -> dict:
        """スケーリング統計情報の取得"""
        if not self.is_fitted or self.scaler is None:
            return {}
        
        stats = {
            'method': self.scaling_method
        }
        
        if hasattr(self.scaler, 'mean_'):
            stats['mean'] = self.scaler.mean_.tolist()
        
        if hasattr(self.scaler, 'scale_'):
            stats['scale'] = self.scaler.scale_.tolist()
        
        if hasattr(self.scaler, 'min_'):
            stats['min'] = self.scaler.min_.tolist()
        
        if hasattr(self.scaler, 'data_min_'):
            stats['data_min'] = self.scaler.data_min_.tolist()
        
        if hasattr(self.scaler, 'data_max_'):
            stats['data_max'] = self.scaler.data_max_.tolist()
        
        return stats