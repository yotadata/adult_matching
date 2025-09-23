"""
Configuration Loading Utilities

設定ファイル読み込みユーティリティ
JSON、YAML、Python設定ファイルをサポート
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass

from backend.ml.config import TrainingConfig


class ConfigLoader:
    """設定ファイル読み込みクラス"""

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """JSON設定ファイルの読み込み"""
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """YAML設定ファイルの読み込み"""
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
        """ファイル拡張子に基づいて適切な読み込み方法を選択"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            return ConfigLoader.load_json(file_path)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            return ConfigLoader.load_yaml(file_path)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")

    @staticmethod
    def load_training_config(file_path: Union[str, Path]) -> TrainingConfig:
        """トレーニング設定の読み込み"""
        config_dict = ConfigLoader.load_config(file_path)
        
        # ネストされた設定の処理
        if 'training' in config_dict:
            config_dict = config_dict['training']
        elif 'model' in config_dict and 'training' in config_dict:
            # model設定とtraining設定をマージ
            merged_config = {}
            merged_config.update(config_dict.get('model', {}))
            merged_config.update(config_dict.get('training', {}))
            config_dict = merged_config
        
        # TrainingConfigに変換
        try:
            return TrainingConfig(**config_dict)
        except TypeError as e:
            # 不明なフィールドを除去して再試行
            valid_fields = set(TrainingConfig.__annotations__.keys())
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
            return TrainingConfig(**filtered_config)

    @staticmethod
    def save_config(config: Dict[str, Any], file_path: Union[str, Path]):
        """設定の保存"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """複数の設定をマージ（後の設定が優先）"""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged

    @staticmethod
    def validate_training_config(config: TrainingConfig) -> bool:
        """トレーニング設定の妥当性検証"""
        errors = []
        
        # 基本チェック
        if config.user_embedding_dim <= 0:
            errors.append("user_embedding_dim must be positive")
        
        if config.item_embedding_dim <= 0:
            errors.append("item_embedding_dim must be positive")
        
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.epochs <= 0:
            errors.append("epochs must be positive")
        
        if not (0 < config.learning_rate < 1):
            errors.append("learning_rate must be between 0 and 1")
        
        if not (0 <= config.validation_split < 1):
            errors.append("validation_split must be between 0 and 1")
        
        if config.dropout_rate < 0 or config.dropout_rate >= 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        # 隠れ層の検証
        if not config.user_hidden_units or not all(u > 0 for u in config.user_hidden_units):
            errors.append("user_hidden_units must be a list of positive integers")
        
        if not config.item_hidden_units or not all(u > 0 for u in config.item_hidden_units):
            errors.append("item_hidden_units must be a list of positive integers")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True