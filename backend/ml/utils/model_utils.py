"""
Model Utilities

モデル関連ユーティリティ関数
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class ModelUtils:
    """モデルユーティリティクラス"""
    
    @staticmethod
    def save_model_metadata(
        model_path: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0"
    ):
        """モデルメタデータ保存"""
        metadata = {
            "version": version,
            "model_path": model_path,
            "config": config,
            "metrics": metrics or {},
            "created_at": str(np.datetime64('now')),
            "framework": "tensorflow",
            "model_type": "two_tower"
        }
        
        metadata_path = Path(model_path).parent / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
        """モデルメタデータ読み込み"""
        metadata_path = Path(model_path).parent / "model_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def validate_model_inputs(
        user_features: np.ndarray,
        item_features: np.ndarray
    ) -> bool:
        """モデル入力検証"""
        # 基本形状チェック
        if len(user_features.shape) != 2 or len(item_features.shape) != 2:
            return False
        
        # バッチサイズ一致チェック
        if user_features.shape[0] != item_features.shape[0]:
            return False
        
        # NaN/Inf チェック
        if np.any(np.isnan(user_features)) or np.any(np.isnan(item_features)):
            return False
        
        if np.any(np.isinf(user_features)) or np.any(np.isinf(item_features)):
            return False
        
        return True
    
    @staticmethod
    def normalize_features(features: np.ndarray, method: str = "standard") -> np.ndarray:
        """特徴量正規化"""
        if method == "standard":
            # 標準化 (z-score)
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std = np.where(std == 0, 1, std)  # ゼロ除算防止
            return (features - mean) / std
        
        elif method == "minmax":
            # Min-Max正規化
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)  # ゼロ除算防止
            return (features - min_val) / range_val
        
        elif method == "l2":
            # L2正規化
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return features / norms
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def calculate_similarity(
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        method: str = "cosine"
    ) -> np.ndarray:
        """類似度計算"""
        if method == "cosine":
            # コサイン類似度
            user_norm = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
            item_norm = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
            
            user_normalized = user_embeddings / np.where(user_norm == 0, 1, user_norm)
            item_normalized = item_embeddings / np.where(item_norm == 0, 1, item_norm)
            
            return np.sum(user_normalized * item_normalized, axis=1)
        
        elif method == "dot":
            # 内積
            return np.sum(user_embeddings * item_embeddings, axis=1)
        
        elif method == "euclidean":
            # ユークリッド距離（類似度として使用する場合は逆数）
            distances = np.linalg.norm(user_embeddings - item_embeddings, axis=1)
            return 1.0 / (1.0 + distances)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def batch_predict(
        model,
        user_features: np.ndarray,
        item_features: np.ndarray,
        batch_size: int = 1000
    ) -> np.ndarray:
        """バッチ予測"""
        total_samples = user_features.shape[0]
        predictions = []
        
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            
            batch_user = user_features[start_idx:end_idx]
            batch_item = item_features[start_idx:end_idx]
            
            batch_pred = model.predict({
                'user_features': batch_user,
                'item_features': batch_item
            })
            
            predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0)
    
    @staticmethod
    def model_size_analysis(model_path: str) -> Dict[str, Any]:
        """モデルサイズ解析"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            return {"error": "Model path does not exist"}
        
        total_size = 0
        file_details = []
        
        if model_path.is_file():
            # 単一ファイル
            size = model_path.stat().st_size
            total_size = size
            file_details.append({
                "file": model_path.name,
                "size_bytes": size,
                "size_mb": size / (1024 * 1024)
            })
        else:
            # ディレクトリ
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    file_details.append({
                        "file": str(file_path.relative_to(model_path)),
                        "size_bytes": size,
                        "size_mb": size / (1024 * 1024)
                    })
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "file_count": len(file_details),
            "file_details": file_details
        }
    
    @staticmethod
    def compare_model_architectures(
        model_config_a: Dict[str, Any],
        model_config_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """モデルアーキテクチャ比較"""
        comparison = {
            "differences": {},
            "similarities": {},
            "parameter_changes": {}
        }
        
        # 全キーを取得
        all_keys = set(model_config_a.keys()) | set(model_config_b.keys())
        
        for key in all_keys:
            val_a = model_config_a.get(key)
            val_b = model_config_b.get(key)
            
            if val_a == val_b:
                comparison["similarities"][key] = val_a
            else:
                comparison["differences"][key] = {
                    "model_a": val_a,
                    "model_b": val_b
                }
                
                # 数値変化の場合、変化率計算
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    if val_a != 0:
                        change_rate = (val_b - val_a) / val_a * 100
                        comparison["parameter_changes"][key] = {
                            "absolute_change": val_b - val_a,
                            "relative_change_percent": change_rate
                        }
        
        return comparison
    
    @staticmethod
    def estimate_memory_usage(
        batch_size: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 768,
        dtype_size: int = 4  # float32 = 4 bytes
    ) -> Dict[str, float]:
        """メモリ使用量推定"""
        
        # 入力特徴量メモリ
        user_input_memory = batch_size * user_feature_dim * dtype_size
        item_input_memory = batch_size * item_feature_dim * dtype_size
        
        # 埋め込みメモリ
        user_embedding_memory = batch_size * embedding_dim * dtype_size
        item_embedding_memory = batch_size * embedding_dim * dtype_size
        
        # 出力メモリ
        output_memory = batch_size * dtype_size
        
        # 中間層メモリ（概算）
        intermediate_memory = batch_size * embedding_dim * 2 * dtype_size
        
        total_memory = (
            user_input_memory + item_input_memory +
            user_embedding_memory + item_embedding_memory +
            output_memory + intermediate_memory
        )
        
        return {
            "user_input_mb": user_input_memory / (1024 * 1024),
            "item_input_mb": item_input_memory / (1024 * 1024),
            "user_embedding_mb": user_embedding_memory / (1024 * 1024),
            "item_embedding_mb": item_embedding_memory / (1024 * 1024),
            "output_mb": output_memory / (1024 * 1024),
            "intermediate_mb": intermediate_memory / (1024 * 1024),
            "total_mb": total_memory / (1024 * 1024),
            "total_gb": total_memory / (1024 * 1024 * 1024)
        }