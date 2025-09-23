"""
Configuration Manager

データパッケージ設定管理システム
- 環境別設定の一元管理
- 動的設定更新
- 設定検証とデフォルト値管理
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from backend.ml.utils.logger import get_ml_logger

logger = get_ml_logger(__name__)

class Environment(Enum):
    """環境タイプ"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    """データベース設定"""
    host: str
    port: int
    database: str
    user: str
    password: str
    ssl_mode: str = "require"
    connection_timeout: int = 30
    max_connections: int = 10

@dataclass
class APIConfig:
    """API設定"""
    dmm_api_id: str
    dmm_affiliate_id: str
    rate_limit_per_second: float = 1.0
    timeout_seconds: int = 30
    max_retries: int = 3
    base_url: str = "https://api.dmm.com/affiliate/v3/"

@dataclass
class StorageConfig:
    """ストレージ設定"""
    base_directory: str
    max_file_size_mb: int = 100
    retention_days: int = 30
    backup_enabled: bool = True
    compression_enabled: bool = True

@dataclass
class ProcessingConfig:
    """処理設定"""
    batch_size: int = 1000
    max_workers: int = 4
    memory_limit_mb: int = 2048
    timeout_minutes: int = 60
    quality_threshold: float = 0.8

@dataclass
class MLConfig:
    """ML設定"""
    model_path: str
    embedding_dimension: int = 768
    batch_size: int = 32
    gpu_enabled: bool = False
    model_version: str = "latest"

@dataclass
class MonitoringConfig:
    """監視設定"""
    metrics_enabled: bool = True
    log_level: str = "INFO"
    alert_email: Optional[str] = None
    health_check_interval_minutes: int = 5
    performance_tracking: bool = True

@dataclass
class DataConfig:
    """完全データ設定"""
    environment: Environment
    database: DatabaseConfig
    api: APIConfig
    storage: StorageConfig
    processing: ProcessingConfig
    ml: MLConfig
    monitoring: MonitoringConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)

class ConfigManager:
    """設定管理システム"""
    
    def __init__(self, config_dir: Path = None, environment: str = None):
        self.config_dir = config_dir or Path("backend/data/configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 環境の決定
        self.environment = Environment(
            environment or 
            os.getenv("ENVIRONMENT", "development")
        )
        
        # 設定キャッシュ
        self._config_cache: Dict[str, Any] = {}
        self._config_files = {
            "default": self.config_dir / "default.json",
            "development": self.config_dir / "development.json",
            "staging": self.config_dir / "staging.json",
            "production": self.config_dir / "production.json",
            "testing": self.config_dir / "testing.json"
        }
        
        # デフォルト設定を作成
        self._create_default_configs()
        
        logger.info(f"ConfigManager initialized for environment: {self.environment.value}")
    
    def _create_default_configs(self):
        """デフォルト設定ファイルの作成"""
        
        # デフォルト設定
        default_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "adult_matching",
                "user": "postgres",
                "password": "",
                "ssl_mode": "prefer",
                "connection_timeout": 30,
                "max_connections": 10
            },
            "api": {
                "dmm_api_id": "",
                "dmm_affiliate_id": "",
                "rate_limit_per_second": 1.0,
                "timeout_seconds": 30,
                "max_retries": 3,
                "base_url": "https://api.dmm.com/affiliate/v3/"
            },
            "storage": {
                "base_directory": "backend/data/storage",
                "max_file_size_mb": 100,
                "retention_days": 30,
                "backup_enabled": True,
                "compression_enabled": True
            },
            "processing": {
                "batch_size": 1000,
                "max_workers": 4,
                "memory_limit_mb": 2048,
                "timeout_minutes": 60,
                "quality_threshold": 0.8
            },
            "ml": {
                "model_path": "backend/ml/models",
                "embedding_dimension": 768,
                "batch_size": 32,
                "gpu_enabled": False,
                "model_version": "latest"
            },
            "monitoring": {
                "metrics_enabled": True,
                "log_level": "INFO",
                "alert_email": None,
                "health_check_interval_minutes": 5,
                "performance_tracking": True
            }
        }
        
        # 開発環境設定
        development_config = default_config.copy()
        development_config.update({
            "database": {
                **default_config["database"],
                "host": "localhost",
                "database": "adult_matching_dev"
            },
            "monitoring": {
                **default_config["monitoring"],
                "log_level": "DEBUG"
            },
            "processing": {
                **default_config["processing"],
                "batch_size": 100  # 開発時は小さめ
            }
        })
        
        # ステージング環境設定
        staging_config = default_config.copy()
        staging_config.update({
            "database": {
                **default_config["database"],
                "host": "staging-db.example.com",
                "database": "adult_matching_staging",
                "ssl_mode": "require"
            },
            "api": {
                **default_config["api"],
                "rate_limit_per_second": 0.5  # ステージングでは制限強め
            }
        })
        
        # 本番環境設定
        production_config = default_config.copy()
        production_config.update({
            "database": {
                **default_config["database"],
                "host": "prod-db.example.com",
                "database": "adult_matching_prod",
                "ssl_mode": "require",
                "max_connections": 20
            },
            "storage": {
                **default_config["storage"],
                "retention_days": 90,
                "max_file_size_mb": 500
            },
            "processing": {
                **default_config["processing"],
                "batch_size": 5000,
                "max_workers": 8,
                "memory_limit_mb": 8192
            },
            "ml": {
                **default_config["ml"],
                "gpu_enabled": True,
                "batch_size": 128
            },
            "monitoring": {
                **default_config["monitoring"],
                "log_level": "WARNING",
                "alert_email": "admin@example.com"
            }
        })
        
        # テスト環境設定
        testing_config = default_config.copy()
        testing_config.update({
            "database": {
                **default_config["database"],
                "database": "adult_matching_test"
            },
            "processing": {
                **default_config["processing"],
                "batch_size": 10,
                "timeout_minutes": 5
            },
            "monitoring": {
                **default_config["monitoring"],
                "metrics_enabled": False
            }
        })
        
        # 設定ファイルの作成
        configs = {
            "default": default_config,
            "development": development_config,
            "staging": staging_config,
            "production": production_config,
            "testing": testing_config
        }
        
        for env_name, config in configs.items():
            config_file = self._config_files[env_name]
            if not config_file.exists():
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                logger.info(f"Created default config: {config_file}")
    
    def load_config(self) -> DataConfig:
        """現在の環境設定をロード"""
        
        cache_key = f"config_{self.environment.value}"
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # デフォルト設定をロード
        with open(self._config_files["default"], 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 環境固有設定で上書き
        env_config_file = self._config_files[self.environment.value]
        if env_config_file.exists():
            with open(env_config_file, 'r', encoding='utf-8') as f:
                env_config = json.load(f)
                self._deep_update(config_data, env_config)
        
        # 環境変数で上書き
        self._apply_env_overrides(config_data)
        
        # DataConfigオブジェクトに変換
        data_config = DataConfig(
            environment=self.environment,
            database=DatabaseConfig(**config_data["database"]),
            api=APIConfig(**config_data["api"]),
            storage=StorageConfig(**config_data["storage"]),
            processing=ProcessingConfig(**config_data["processing"]),
            ml=MLConfig(**config_data["ml"]),
            monitoring=MonitoringConfig(**config_data["monitoring"])
        )
        
        # キャッシュに保存
        self._config_cache[cache_key] = data_config
        
        logger.info(f"Config loaded for environment: {self.environment.value}")
        return data_config
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """辞書の深い更新"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]):
        """環境変数による設定上書き"""
        
        # データベース設定
        if os.getenv("DATABASE_URL"):
            # PostgreSQL URLをパース
            import urllib.parse as urlparse
            url = urlparse.urlparse(os.getenv("DATABASE_URL"))
            
            config_data["database"].update({
                "host": url.hostname or config_data["database"]["host"],
                "port": url.port or config_data["database"]["port"],
                "database": url.path.lstrip('/') or config_data["database"]["database"],
                "user": url.username or config_data["database"]["user"],
                "password": url.password or config_data["database"]["password"]
            })
        
        # API設定
        if os.getenv("DMM_API_ID"):
            config_data["api"]["dmm_api_id"] = os.getenv("DMM_API_ID")
        
        if os.getenv("DMM_AFFILIATE_ID"):
            config_data["api"]["dmm_affiliate_id"] = os.getenv("DMM_AFFILIATE_ID")
        
        # ストレージ設定
        if os.getenv("DATA_STORAGE_PATH"):
            config_data["storage"]["base_directory"] = os.getenv("DATA_STORAGE_PATH")
        
        # 監視設定
        if os.getenv("LOG_LEVEL"):
            config_data["monitoring"]["log_level"] = os.getenv("LOG_LEVEL")
        
        if os.getenv("ALERT_EMAIL"):
            config_data["monitoring"]["alert_email"] = os.getenv("ALERT_EMAIL")
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """設定の動的更新"""
        try:
            env_config_file = self._config_files[self.environment.value]
            
            # 現在の設定を読み込み
            if env_config_file.exists():
                with open(env_config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            
            # セクションが存在しなければ作成
            if section not in config_data:
                config_data[section] = {}
            
            # 更新を適用
            config_data[section].update(updates)
            
            # ファイルに保存
            with open(env_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # キャッシュをクリア
            cache_key = f"config_{self.environment.value}"
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
            
            logger.info(f"Config updated: {section} in {self.environment.value}")
            return True
            
        except Exception as error:
            logger.error(f"Config update failed: {error}")
            return False
    
    def validate_config(self, config: DataConfig = None) -> Dict[str, List[str]]:
        """設定の検証"""
        
        if config is None:
            config = self.load_config()
        
        errors = {}
        
        # データベース設定検証
        db_errors = []
        if not config.database.host:
            db_errors.append("Database host is required")
        if not config.database.database:
            db_errors.append("Database name is required")
        if not (1 <= config.database.port <= 65535):
            db_errors.append("Database port must be between 1 and 65535")
        
        if db_errors:
            errors["database"] = db_errors
        
        # API設定検証
        api_errors = []
        if not config.api.dmm_api_id:
            api_errors.append("DMM API ID is required")
        if not config.api.dmm_affiliate_id:
            api_errors.append("DMM Affiliate ID is required")
        if config.api.rate_limit_per_second <= 0:
            api_errors.append("Rate limit must be positive")
        
        if api_errors:
            errors["api"] = api_errors
        
        # ストレージ設定検証
        storage_errors = []
        if not config.storage.base_directory:
            storage_errors.append("Base directory is required")
        if config.storage.max_file_size_mb <= 0:
            storage_errors.append("Max file size must be positive")
        if config.storage.retention_days <= 0:
            storage_errors.append("Retention days must be positive")
        
        if storage_errors:
            errors["storage"] = storage_errors
        
        # 処理設定検証
        processing_errors = []
        if config.processing.batch_size <= 0:
            processing_errors.append("Batch size must be positive")
        if config.processing.max_workers <= 0:
            processing_errors.append("Max workers must be positive")
        if not (0 <= config.processing.quality_threshold <= 1):
            processing_errors.append("Quality threshold must be between 0 and 1")
        
        if processing_errors:
            errors["processing"] = processing_errors
        
        # ML設定検証
        ml_errors = []
        if not config.ml.model_path:
            ml_errors.append("Model path is required")
        if config.ml.embedding_dimension <= 0:
            ml_errors.append("Embedding dimension must be positive")
        if config.ml.batch_size <= 0:
            ml_errors.append("ML batch size must be positive")
        
        if ml_errors:
            errors["ml"] = ml_errors
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """設定サマリー取得"""
        config = self.load_config()
        validation_errors = self.validate_config(config)
        
        return {
            "environment": config.environment.value,
            "config_files": {
                name: file.exists() 
                for name, file in self._config_files.items()
            },
            "validation_status": "valid" if not validation_errors else "invalid",
            "validation_errors": validation_errors,
            "last_loaded": self._config_cache.get(f"config_{self.environment.value}_timestamp"),
            "sections": {
                "database": {
                    "host": config.database.host,
                    "database": config.database.database,
                    "ssl_mode": config.database.ssl_mode
                },
                "api": {
                    "has_credentials": bool(config.api.dmm_api_id and config.api.dmm_affiliate_id),
                    "rate_limit": config.api.rate_limit_per_second
                },
                "storage": {
                    "base_directory": config.storage.base_directory,
                    "retention_days": config.storage.retention_days
                },
                "processing": {
                    "batch_size": config.processing.batch_size,
                    "max_workers": config.processing.max_workers,
                    "quality_threshold": config.processing.quality_threshold
                },
                "ml": {
                    "model_path": config.ml.model_path,
                    "embedding_dimension": config.ml.embedding_dimension,
                    "gpu_enabled": config.ml.gpu_enabled
                },
                "monitoring": {
                    "log_level": config.monitoring.log_level,
                    "metrics_enabled": config.monitoring.metrics_enabled
                }
            }
        }
    
    def export_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """設定のエクスポート"""
        config = self.load_config()
        exported = config.to_dict()
        
        if not include_sensitive:
            # 機密情報をマスク
            exported["database"]["password"] = "***"
            exported["api"]["dmm_api_id"] = "***"
            exported["api"]["dmm_affiliate_id"] = "***"
        
        return exported
    
    def reset_to_defaults(self, sections: List[str] = None) -> bool:
        """デフォルト設定にリセット"""
        try:
            # デフォルト設定を読み込み
            with open(self._config_files["default"], 'r', encoding='utf-8') as f:
                default_config = json.load(f)
            
            env_config_file = self._config_files[self.environment.value]
            
            if sections:
                # 特定セクションのみリセット
                if env_config_file.exists():
                    with open(env_config_file, 'r', encoding='utf-8') as f:
                        current_config = json.load(f)
                else:
                    current_config = {}
                
                for section in sections:
                    if section in default_config:
                        current_config[section] = default_config[section]
                
                with open(env_config_file, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, indent=2, ensure_ascii=False)
            else:
                # 全体をリセット
                with open(env_config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            # キャッシュをクリア
            cache_key = f"config_{self.environment.value}"
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
            
            logger.info(f"Config reset to defaults: {sections or 'all sections'}")
            return True
            
        except Exception as error:
            logger.error(f"Config reset failed: {error}")
            return False

# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """グローバル設定管理インスタンス取得"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager

def get_config() -> DataConfig:
    """現在の設定取得（便利関数）"""
    return get_config_manager().load_config()