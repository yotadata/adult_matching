"""
Unified Feature Processor

統一特徴量処理システム
ユーザーとアイテム特徴量の統合処理を提供
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle

from backend.ml.utils.logger import get_ml_logger
from backend.ml.preprocessing.utils.data_validator import DataValidator
from backend.ml.preprocessing.utils.feature_scaler import FeatureScaler

logger = get_ml_logger(__name__)

@dataclass
class FeatureConfig:
    """特徴量処理設定"""
    target_dimension: int = 768
    categorical_encoding: str = "label"  # "label", "onehot", "target", "embedding"
    numerical_scaling: str = "standard"  # "standard", "minmax", "robust", "none"
    handle_missing: str = "mean"  # "mean", "median", "mode", "drop", "zero"
    text_vectorization: str = "tfidf"  # "tfidf", "count", "embedding"
    max_vocab_size: int = 10000
    min_feature_frequency: int = 2

@dataclass
class FeatureStats:
    """特徴量統計情報"""
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_categorical_features: int
    num_numerical_features: int
    num_text_features: int
    missing_value_ratio: float
    processing_time_seconds: float
    
class BaseFeatureProcessor(ABC):
    """基底特徴量プロセッサー"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.is_fitted = False
        self.feature_stats = None
        self.preprocessors = {}
        self.validator = DataValidator()
        self.scaler = FeatureScaler()
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[np.ndarray] = None):
        """前処理パラメータの学習"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """特徴量変換の実行"""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[np.ndarray] = None) -> np.ndarray:
        """学習と変換の一括実行"""
        self.fit(data, target)
        return self.transform(data)
    
    def save_preprocessors(self, save_path: Union[str, Path]):
        """前処理器の保存"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_stats': self.feature_stats,
            'preprocessors': self.preprocessors
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Preprocessors saved to {save_path}")
    
    def load_preprocessors(self, load_path: Union[str, Path]):
        """前処理器の読み込み"""
        load_path = Path(load_path)
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.is_fitted = save_data['is_fitted']
        self.feature_stats = save_data['feature_stats']
        self.preprocessors = save_data['preprocessors']
        
        logger.info(f"Preprocessors loaded from {load_path}")

class FeatureProcessor(BaseFeatureProcessor):
    """統一特徴量プロセッサー"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config)
        self.categorical_columns = []
        self.numerical_columns = []
        self.text_columns = []
        
    def _identify_feature_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """特徴量タイプの自動識別"""
        feature_types = {
            'categorical': [],
            'numerical': [],
            'text': []
        }
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # 文字列データの場合、平均文字数で判定
                avg_length = data[column].astype(str).str.len().mean()
                if avg_length > 20:  # 長いテキストの場合
                    feature_types['text'].append(column)
                else:  # 短いテキストはカテゴリカル
                    feature_types['categorical'].append(column)
            elif data[column].dtype in ['int64', 'float64']:
                # 数値データの場合、一意値の数で判定
                unique_values = data[column].nunique()
                if unique_values < 20:  # 少数の一意値はカテゴリカル
                    feature_types['categorical'].append(column)
                else:  # 多数の一意値は数値
                    feature_types['numerical'].append(column)
            else:
                feature_types['categorical'].append(column)
        
        return feature_types
    
    def _process_categorical_features(self, data: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """カテゴリカル特徴量の処理"""
        if not self.categorical_columns:
            return np.array([]).reshape(len(data), 0)
        
        categorical_features = []
        
        for column in self.categorical_columns:
            if is_training:
                if self.config.categorical_encoding == "label":
                    from sklearn.preprocessing import LabelEncoder
                    encoder = LabelEncoder()
                    # 欠損値を文字列に変換
                    values = data[column].fillna("unknown").astype(str)
                    encoded = encoder.fit_transform(values)
                    self.preprocessors[f"{column}_encoder"] = encoder
                    
                elif self.config.categorical_encoding == "onehot":
                    from sklearn.preprocessing import OneHotEncoder
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    values = data[column].fillna("unknown").astype(str).values.reshape(-1, 1)
                    encoded = encoder.fit_transform(values)
                    self.preprocessors[f"{column}_encoder"] = encoder
                    
            else:
                encoder = self.preprocessors[f"{column}_encoder"]
                if self.config.categorical_encoding == "label":
                    values = data[column].fillna("unknown").astype(str)
                    # 未知のラベルを処理
                    known_classes = set(encoder.classes_)
                    values = values.apply(lambda x: x if x in known_classes else "unknown")
                    encoded = encoder.transform(values)
                    
                elif self.config.categorical_encoding == "onehot":
                    values = data[column].fillna("unknown").astype(str).values.reshape(-1, 1)
                    encoded = encoder.transform(values)
            
            if encoded.ndim == 1:
                encoded = encoded.reshape(-1, 1)
            categorical_features.append(encoded)
        
        if categorical_features:
            return np.hstack(categorical_features)
        else:
            return np.array([]).reshape(len(data), 0)
    
    def _process_numerical_features(self, data: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """数値特徴量の処理"""
        if not self.numerical_columns:
            return np.array([]).reshape(len(data), 0)
        
        numerical_data = data[self.numerical_columns].copy()
        
        # 欠損値処理
        if self.config.handle_missing == "mean":
            if is_training:
                self.preprocessors['numerical_imputer'] = numerical_data.mean()
            numerical_data = numerical_data.fillna(self.preprocessors['numerical_imputer'])
            
        elif self.config.handle_missing == "median":
            if is_training:
                self.preprocessors['numerical_imputer'] = numerical_data.median()
            numerical_data = numerical_data.fillna(self.preprocessors['numerical_imputer'])
            
        elif self.config.handle_missing == "zero":
            numerical_data = numerical_data.fillna(0)
        
        # スケーリング
        if self.config.numerical_scaling != "none":
            if is_training:
                scaled_data = self.scaler.fit_transform(
                    numerical_data.values, 
                    method=self.config.numerical_scaling
                )
                self.preprocessors['numerical_scaler'] = self.scaler
            else:
                scaled_data = self.preprocessors['numerical_scaler'].transform(numerical_data.values)
            
            return scaled_data
        
        return numerical_data.values
    
    def _process_text_features(self, data: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """テキスト特徴量の処理"""
        if not self.text_columns:
            return np.array([]).reshape(len(data), 0)
        
        text_features = []
        
        for column in self.text_columns:
            text_data = data[column].fillna("").astype(str)
            
            if is_training:
                if self.config.text_vectorization == "tfidf":
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vectorizer = TfidfVectorizer(
                        max_features=self.config.max_vocab_size,
                        min_df=self.config.min_feature_frequency,
                        stop_words=None
                    )
                    vectors = vectorizer.fit_transform(text_data).toarray()
                    self.preprocessors[f"{column}_vectorizer"] = vectorizer
                    
                elif self.config.text_vectorization == "count":
                    from sklearn.feature_extraction.text import CountVectorizer
                    vectorizer = CountVectorizer(
                        max_features=self.config.max_vocab_size,
                        min_df=self.config.min_feature_frequency
                    )
                    vectors = vectorizer.fit_transform(text_data).toarray()
                    self.preprocessors[f"{column}_vectorizer"] = vectorizer
                    
            else:
                vectorizer = self.preprocessors[f"{column}_vectorizer"]
                vectors = vectorizer.transform(text_data).toarray()
            
            text_features.append(vectors)
        
        if text_features:
            return np.hstack(text_features)
        else:
            return np.array([]).reshape(len(data), 0)
    
    def fit(self, data: pd.DataFrame, target: Optional[np.ndarray] = None):
        """前処理パラメータの学習"""
        logger.info(f"Fitting feature processor on {len(data)} samples")
        start_time = pd.Timestamp.now()
        
        # データ検証
        self.validator.validate_input_data(data)
        
        # 特徴量タイプの識別
        feature_types = self._identify_feature_types(data)
        self.categorical_columns = feature_types['categorical']
        self.numerical_columns = feature_types['numerical']
        self.text_columns = feature_types['text']
        
        logger.info(f"Identified features - Categorical: {len(self.categorical_columns)}, "
                   f"Numerical: {len(self.numerical_columns)}, Text: {len(self.text_columns)}")
        
        # 各特徴量タイプの処理パラメータ学習
        categorical_features = self._process_categorical_features(data, is_training=True)
        numerical_features = self._process_numerical_features(data, is_training=True)
        text_features = self._process_text_features(data, is_training=True)
        
        # 最終特徴量の結合
        all_features = np.hstack([
            categorical_features,
            numerical_features,
            text_features
        ])
        
        # 統計情報の計算
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        
        self.feature_stats = FeatureStats(
            input_shape=data.shape,
            output_shape=all_features.shape,
            num_categorical_features=len(self.categorical_columns),
            num_numerical_features=len(self.numerical_columns),
            num_text_features=len(self.text_columns),
            missing_value_ratio=missing_ratio,
            processing_time_seconds=processing_time
        )
        
        self.is_fitted = True
        logger.info(f"Feature processor fitted - Output shape: {all_features.shape}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """特徴量変換の実行"""
        if not self.is_fitted:
            raise ValueError("Feature processor must be fitted before transform")
        
        # データ検証
        self.validator.validate_input_data(data)
        
        # 各特徴量タイプの変換
        categorical_features = self._process_categorical_features(data, is_training=False)
        numerical_features = self._process_numerical_features(data, is_training=False)
        text_features = self._process_text_features(data, is_training=False)
        
        # 特徴量の結合
        all_features = np.hstack([
            categorical_features,
            numerical_features, 
            text_features
        ])
        
        logger.debug(f"Transformed {len(data)} samples to {all_features.shape}")
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """特徴量名の取得"""
        feature_names = []
        
        # カテゴリカル特徴量名
        for column in self.categorical_columns:
            if self.config.categorical_encoding == "onehot":
                encoder = self.preprocessors.get(f"{column}_encoder")
                if encoder:
                    for category in encoder.categories_[0]:
                        feature_names.append(f"{column}_{category}")
                else:
                    feature_names.append(column)
            else:
                feature_names.append(column)
        
        # 数値特徴量名
        feature_names.extend(self.numerical_columns)
        
        # テキスト特徴量名
        for column in self.text_columns:
            vectorizer = self.preprocessors.get(f"{column}_vectorizer")
            if vectorizer:
                vocab = vectorizer.get_feature_names_out()
                for word in vocab:
                    feature_names.append(f"{column}_{word}")
            else:
                feature_names.append(column)
        
        return feature_names
    
    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度の取得（統計ベース）"""
        if not self.is_fitted:
            return {}
        
        importance = {}
        
        # 各特徴量タイプの重要度
        total_features = (
            len(self.categorical_columns) + 
            len(self.numerical_columns) + 
            len(self.text_columns)
        )
        
        if total_features == 0:
            return {}
        
        # 简单的重要度計算（実際のプロジェクトではより高度な方法を使用）
        for column in self.categorical_columns:
            importance[column] = 1.0 / total_features
            
        for column in self.numerical_columns:
            importance[column] = 1.0 / total_features
            
        for column in self.text_columns:
            importance[column] = 1.0 / total_features
        
        return importance