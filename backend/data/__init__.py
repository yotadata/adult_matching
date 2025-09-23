"""
Unified Data Management Package

統一データ管理パッケージ
- データ取り込み (ingestion)
- データ処理 (processing) 
- データストレージ (storage)
- データ検証 (validation)
- データエクスポート (export)
- スキーマ管理 (schemas)
- パイプライン統合 (pipelines)
"""

import sys
import os
from pathlib import Path

# パッケージルートディレクトリ
PACKAGE_ROOT = Path(__file__).parent.absolute()
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Pythonパスに追加
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# バージョン情報
__version__ = "2.0.0"
__author__ = "Adult Matching Data Team"

# データ管理設定
DATA_CONFIG = {
    "default_batch_size": 1000,
    "max_file_size_mb": 500,
    "supported_formats": ["json", "csv", "parquet", "feather"],
    "storage_backends": ["local", "supabase", "s3"],
    "validation_levels": ["basic", "schema", "comprehensive"],
    "pipeline_modes": ["batch", "streaming", "hybrid"]
}

# ディレクトリ構造
DATA_DIRS = {
    "raw": PACKAGE_ROOT / "storage" / "raw",
    "processed": PACKAGE_ROOT / "storage" / "processed", 
    "validated": PACKAGE_ROOT / "storage" / "validated",
    "exported": PACKAGE_ROOT / "storage" / "exported",
    "temp": PACKAGE_ROOT / "storage" / "temp",
    "schemas": PACKAGE_ROOT / "schemas",
    "logs": PACKAGE_ROOT / "logs"
}

# 自動ディレクトリ作成
for dir_path in DATA_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# 遅延インポート関数
def get_data_ingestion():
    """データ取り込みモジュールの取得"""
    from .ingestion.data_ingestion_manager import DataIngestionManager
    return DataIngestionManager

def get_data_processor():
    """データ処理モジュールの取得"""
    from .processing.unified_data_processor import UnifiedDataProcessor
    return UnifiedDataProcessor

def get_data_validator():
    """データ検証モジュールの取得"""
    from .validation.data_validator import DataValidator
    return DataValidator

def get_pipeline_manager():
    """パイプライン管理モジュールの取得"""
    from .pipelines.pipeline_manager import PipelineManager
    return PipelineManager

def get_schema_manager():
    """スキーマ管理モジュールの取得"""
    from .schemas.schema_manager import SchemaManager
    return SchemaManager

def get_export_manager():
    """エクスポート管理モジュールの取得"""
    from .export.data_exporter import DataExporter
    return DataExporter

# ファクトリー関数
def create_data_pipeline(config=None):
    """統合データパイプラインの作成"""
    pipeline_class = get_pipeline_manager()
    final_config = {**DATA_CONFIG}
    if config:
        final_config.update(config)
    
    return pipeline_class(
        config=final_config,
        base_dir=PACKAGE_ROOT
    )

def create_ingestion_manager(sources=None):
    """データ取り込み管理の作成"""
    ingestion_class = get_data_ingestion()
    return ingestion_class(
        sources=sources or ["api", "scraping", "file"],
        output_dir=DATA_DIRS["raw"]
    )

def create_processing_pipeline(processors=None):
    """データ処理パイプラインの作成"""
    processor_class = get_data_processor()
    return processor_class(
        processors=processors or ["clean", "transform", "enrich"],
        input_dir=DATA_DIRS["raw"],
        output_dir=DATA_DIRS["processed"]
    )

def create_validation_suite(validation_level="comprehensive"):
    """データ検証スイートの作成"""
    validator_class = get_data_validator()
    return validator_class()

def create_quality_assessor():
    """データ品質評価システムの作成"""
    from .validation.quality_assessor import DataQualityAssessor
    return DataQualityAssessor()

def create_schema_validator():
    """スキーマ検証システムの作成"""
    from .validation.schema_validator import SchemaValidator
    return SchemaValidator()

def create_sync_manager(sync_type="dmm"):
    """データ同期管理システムの作成"""
    try:
        if sync_type == "dmm":
            from .sync.dmm.dmm_sync_manager import DMMSyncManager
            return DMMSyncManager()
        else:
            logger.warning(f"Unknown sync type: {sync_type}")
            return None
    except ImportError as e:
        logger.warning(f"Sync manager not available: {e}")
        return None

def create_quality_monitor():
    """データ品質監視システムの作成"""
    try:
        from .quality.quality_monitor import DataQualityMonitor
        return DataQualityMonitor()
    except ImportError as e:
        logger.warning(f"Quality monitor not available: {e}")
        return None

def create_unified_data_manager_legacy(include_optional=True):
    """
    レガシー統合データ管理システムの作成
    
    Returns:
        dict: 全データ管理コンポーネントの辞書
    """
    managers = {
        "ingestion": create_ingestion_manager(),
        "processing": create_processing_pipeline(), 
        "validation": create_validation_suite(),
        "export": get_export_manager()(),
        "pipeline": create_data_pipeline(),
        "quality": create_quality_assessor(),
        "schema": create_schema_validator()
    }
    
    if include_optional:
        # オプショナルコンポーネント（ImportErrorでも動作）
        sync_manager = create_sync_manager()
        if sync_manager:
            managers["sync"] = sync_manager
            
        quality_monitor = create_quality_monitor()
        if quality_monitor:
            managers["quality_monitor"] = quality_monitor
    
    return managers

def create_unified_data_manager(config=None):
    """
    新しい統合データ管理システムの作成
    
    Args:
        config: 設定辞書
        
    Returns:
        UnifiedDataManager: 統合データ管理システムインスタンス
    """
    try:
        from .unified_data_manager import create_unified_data_manager as create_manager
        return create_manager(config)
    except ImportError as e:
        logger.warning(f"Unified data manager not available: {e}")
        # フォールバック
        return create_unified_data_manager_legacy()

def create_integrated_script_manager():
    """
    統合スクリプト管理システムの作成
    
    Returns:
        IntegratedScriptManager: 統合スクリプト管理システムインスタンス
    """
    try:
        from .script_integration import create_integrated_script_manager
        return create_integrated_script_manager(PROJECT_ROOT)
    except ImportError as e:
        logger.warning(f"Integrated script manager not available: {e}")
        return None

def get_config_manager():
    """設定管理システムの取得"""
    try:
        from .config_manager import get_config_manager as get_manager
        return get_manager()
    except ImportError as e:
        logger.warning(f"Config manager not available: {e}")
        return None

def get_config():
    """現在の設定取得"""
    try:
        from .config_manager import get_config as get_cfg
        return get_cfg()
    except ImportError as e:
        logger.warning(f"Config not available: {e}")
        return None

# ログ設定
import logging

# ログディレクトリの作成
log_file = DATA_DIRS["logs"] / "data_package.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Unified Data Management Package v{__version__} initialized")

# パッケージ情報エクスポート
__all__ = [
    "__version__",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_CONFIG",
    "DATA_DIRS",
    "get_data_ingestion",
    "get_data_processor", 
    "get_data_validator",
    "get_pipeline_manager",
    "get_schema_manager",
    "get_export_manager",
    "create_data_pipeline",
    "create_ingestion_manager",
    "create_processing_pipeline", 
    "create_validation_suite",
    "create_quality_assessor",
    "create_schema_validator",
    "create_unified_data_manager",
    "create_integrated_script_manager"
]