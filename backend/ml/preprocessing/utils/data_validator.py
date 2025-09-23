"""
Data Validation Utilities

データ検証ユーティリティ
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from backend.ml.utils.logger import get_ml_logger

logger = get_ml_logger(__name__)

class DataValidator:
    """データ検証クラス"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """入力データの基本検証"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # 基本チェック
        if data.empty:
            results['is_valid'] = False
            results['errors'].append("Data is empty")
            return results
        
        # 形状チェック
        results['stats']['shape'] = data.shape
        results['stats']['columns'] = list(data.columns)
        
        # 欠損値チェック
        missing_counts = data.isnull().sum()
        missing_ratio = missing_counts / len(data)
        
        results['stats']['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'ratios': missing_ratio.to_dict()
        }
        
        # 高い欠損値率の警告
        high_missing = missing_ratio[missing_ratio > 0.5]
        if not high_missing.empty:
            results['warnings'].append(
                f"High missing value ratio in columns: {high_missing.index.tolist()}"
            )
        
        # データ型チェック
        dtype_counts = data.dtypes.value_counts()
        results['stats']['dtypes'] = dtype_counts.to_dict()
        
        # 重複行チェック
        duplicates = data.duplicated().sum()
        results['stats']['duplicates'] = duplicates
        
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # メモリ使用量
        memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        results['stats']['memory_usage_mb'] = memory_usage
        
        if memory_usage > 1000:  # 1GB
            results['warnings'].append(f"Large memory usage: {memory_usage:.2f} MB")
        
        self.validation_results = results
        
        if results['errors']:
            logger.error(f"Data validation failed: {results['errors']}")
        if results['warnings']:
            logger.warning(f"Data validation warnings: {results['warnings']}")
        
        return results
    
    def validate_feature_consistency(self, 
                                   train_data: pd.DataFrame, 
                                   test_data: pd.DataFrame) -> Dict[str, Any]:
        """訓練・テストデータの一貫性チェック"""
        results = {
            'is_consistent': True,
            'errors': [],
            'warnings': []
        }
        
        # カラム一致チェック
        train_cols = set(train_data.columns)
        test_cols = set(test_data.columns)
        
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        if missing_in_test:
            results['is_consistent'] = False
            results['errors'].append(f"Missing columns in test data: {missing_in_test}")
        
        if extra_in_test:
            results['warnings'].append(f"Extra columns in test data: {extra_in_test}")
        
        # データ型一致チェック
        common_cols = train_cols.intersection(test_cols)
        dtype_mismatches = []
        
        for col in common_cols:
            if train_data[col].dtype != test_data[col].dtype:
                dtype_mismatches.append(col)
        
        if dtype_mismatches:
            results['warnings'].append(f"Data type mismatches: {dtype_mismatches}")
        
        return results
    
    def validate_target_variable(self, target: np.ndarray) -> Dict[str, Any]:
        """ターゲット変数の検証"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        if target is None:
            results['errors'].append("Target variable is None")
            results['is_valid'] = False
            return results
        
        # 基本統計
        results['stats']['shape'] = target.shape
        results['stats']['unique_values'] = len(np.unique(target))
        results['stats']['missing_values'] = np.isnan(target).sum()
        
        # 分類問題の場合の分布チェック
        unique_vals = np.unique(target[~np.isnan(target)])
        results['stats']['value_counts'] = {str(val): np.sum(target == val) for val in unique_vals}
        
        # クラス不均衡の警告
        if len(unique_vals) == 2:  # バイナリ分類
            class_counts = [np.sum(target == val) for val in unique_vals]
            imbalance_ratio = min(class_counts) / max(class_counts)
            
            results['stats']['imbalance_ratio'] = imbalance_ratio
            
            if imbalance_ratio < 0.1:
                results['warnings'].append(
                    f"Severe class imbalance detected: {imbalance_ratio:.3f}"
                )
        
        return results
    
    def validate_model_input(self, features: np.ndarray) -> Dict[str, Any]:
        """モデル入力特徴量の検証"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        if features is None:
            results['is_valid'] = False
            results['errors'].append("Features array is None")
            return results
        
        # 基本統計
        results['stats']['shape'] = features.shape
        results['stats']['dtype'] = str(features.dtype)
        
        # 無限値・NaN チェック
        inf_count = np.isinf(features).sum()
        nan_count = np.isnan(features).sum()
        
        results['stats']['infinite_values'] = inf_count
        results['stats']['nan_values'] = nan_count
        
        if inf_count > 0:
            results['errors'].append(f"Found {inf_count} infinite values")
            results['is_valid'] = False
        
        if nan_count > 0:
            results['errors'].append(f"Found {nan_count} NaN values")
            results['is_valid'] = False
        
        # 特徴量の分布統計
        results['stats']['mean'] = np.mean(features, axis=0).tolist()[:10]  # 最初の10個のみ
        results['stats']['std'] = np.std(features, axis=0).tolist()[:10]
        results['stats']['min'] = np.min(features, axis=0).tolist()[:10]
        results['stats']['max'] = np.max(features, axis=0).tolist()[:10]
        
        # スケールの警告
        feature_scales = np.max(features, axis=0) - np.min(features, axis=0)
        large_scale_features = np.where(feature_scales > 1000)[0]
        
        if len(large_scale_features) > 0:
            results['warnings'].append(
                f"Large scale differences in features: {large_scale_features.tolist()[:5]}"
            )
        
        return results