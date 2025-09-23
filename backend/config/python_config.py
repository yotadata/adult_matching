"""
Python用バックエンド設定ファイル

データ処理とML Pipeline用の設定を管理
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

@dataclass
class ScrapingConfig:
    """スクレイピング設定"""
    request_delay_ms: int
    max_concurrent_requests: int
    retry_attempts: int
    user_agent: str
    dmm_api_id: str
    dmm_affiliate_id: str
    dmm_base_url: str
    rate_limit_per_second: int
    timeout_ms: int

@dataclass
class CleaningConfig:
    """データクリーニング設定"""
    min_title_length: int
    max_title_length: int
    min_description_length: int
    duplicate_threshold: float
    required_fields: List[str]
    price_range: Dict[str, int]
    image_url_validation: bool

@dataclass
class EmbeddingConfig:
    """エンベディング設定"""
    batch_size: int
    vector_dimension: int
    similarity_index_type: str
    model_name: str
    cache_embeddings: bool
    normalize_vectors: bool

@dataclass
class ValidationConfig:
    """データ検証設定"""
    required_fields: List[str]
    price_range: Dict[str, int]
    image_url_validation: bool
    content_length_limits: Dict[str, Dict[str, int]]
    enable_duplicate_detection: bool

@dataclass
class DatabaseConfig:
    """データベース設定"""
    url: str
    pool_size: int
    max_connections: int
    timeout_ms: int
    enable_query_logging: bool

@dataclass
class MLPipelineConfig:
    """ML Pipeline設定"""
    model_path: str
    embedding_dimension: int
    two_tower_hidden_units: List[int]
    learning_rate: float
    batch_size: int
    epochs: int
    validation_split: float
    early_stopping_patience: int
    similarity_threshold: float
    diversity_weight: float

@dataclass
class MonitoringConfig:
    """監視設定"""
    log_level: str
    enable_performance_logging: bool
    enable_error_tracking: bool
    metrics_collection_interval: int
    log_rotation_max_files: int
    log_rotation_max_size_mb: int

@dataclass
class DataProcessingConfig:
    """データ処理統合設定"""
    scraping: ScrapingConfig
    cleaning: CleaningConfig
    embedding: EmbeddingConfig
    validation: ValidationConfig
    database: DatabaseConfig
    monitoring: MonitoringConfig

def get_scraping_config(environment: str = 'development') -> ScrapingConfig:
    """スクレイピング設定を取得"""
    base_config = ScrapingConfig(
        request_delay_ms=int(os.getenv('SCRAPING_REQUEST_DELAY_MS', '1000')),
        max_concurrent_requests=int(os.getenv('SCRAPING_MAX_CONCURRENT', '5')),
        retry_attempts=int(os.getenv('SCRAPING_RETRY_ATTEMPTS', '3')),
        user_agent=os.getenv('SCRAPING_USER_AGENT', 'Mozilla/5.0 (compatible; AdultMatchingBot/1.0)'),
        dmm_api_id=os.getenv('DMM_API_ID', ''),
        dmm_affiliate_id=os.getenv('DMM_AFFILIATE_ID', ''),
        dmm_base_url=os.getenv('DMM_BASE_URL', 'https://api.dmm.com/affiliate/v3/'),
        rate_limit_per_second=int(os.getenv('DMM_RATE_LIMIT_PER_SECOND', '1')),
        timeout_ms=int(os.getenv('DMM_TIMEOUT_MS', '30000'))
    )
    
    if environment == 'production':
        # 本番環境用の調整
        base_config.max_concurrent_requests = min(base_config.max_concurrent_requests, 3)
        base_config.request_delay_ms = max(base_config.request_delay_ms, 1500)
    
    return base_config

def get_cleaning_config(environment: str = 'development') -> CleaningConfig:
    """クリーニング設定を取得"""
    return CleaningConfig(
        min_title_length=int(os.getenv('MIN_TITLE_LENGTH', '10')),
        max_title_length=int(os.getenv('MAX_TITLE_LENGTH', '200')),
        min_description_length=int(os.getenv('MIN_DESCRIPTION_LENGTH', '20')),
        duplicate_threshold=float(os.getenv('DUPLICATE_THRESHOLD', '0.95')),
        required_fields=['title', 'description', 'maker', 'genre', 'price'],
        price_range={
            'min': int(os.getenv('MIN_PRICE', '0')),
            'max': int(os.getenv('MAX_PRICE', '50000'))
        },
        image_url_validation=os.getenv('IMAGE_URL_VALIDATION', 'true').lower() == 'true'
    )

def get_embedding_config(environment: str = 'development') -> EmbeddingConfig:
    """エンベディング設定を取得"""
    base_config = EmbeddingConfig(
        batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '100')),
        vector_dimension=int(os.getenv('PGVECTOR_DIMENSION', '768')),
        similarity_index_type=os.getenv('PGVECTOR_DISTANCE_FUNCTION', 'cosine'),
        model_name=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        cache_embeddings=True,
        normalize_vectors=True
    )
    
    if environment == 'production':
        # 本番環境用の最適化
        base_config.batch_size = min(base_config.batch_size, 50)
    
    return base_config

def get_validation_config(environment: str = 'development') -> ValidationConfig:
    """検証設定を取得"""
    return ValidationConfig(
        required_fields=['title', 'description', 'maker', 'genre', 'price'],
        price_range={
            'min': int(os.getenv('MIN_PRICE', '0')),
            'max': int(os.getenv('MAX_PRICE', '50000'))
        },
        image_url_validation=os.getenv('IMAGE_URL_VALIDATION', 'true').lower() == 'true',
        content_length_limits={
            'title': {'min': 10, 'max': 200},
            'description': {'min': 20, 'max': 2000},
            'maker': {'min': 1, 'max': 100}
        },
        enable_duplicate_detection=True
    )

def get_database_config(environment: str = 'development') -> DatabaseConfig:
    """データベース設定を取得"""
    base_config = DatabaseConfig(
        url=os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/adult_matching'),
        pool_size=int(os.getenv('DATABASE_POOL_SIZE', '20')),
        max_connections=int(os.getenv('DATABASE_MAX_CONNECTIONS', '100')),
        timeout_ms=int(os.getenv('DATABASE_TIMEOUT', '15000')),
        enable_query_logging=os.getenv('ENABLE_SQL_LOGGING', 'true').lower() == 'true'
    )
    
    if environment == 'production':
        # 本番環境用の最適化
        base_config.pool_size = max(base_config.pool_size, 50)
        base_config.max_connections = max(base_config.max_connections, 200)
        base_config.enable_query_logging = False  # 本番では詳細ログを無効化
    
    return base_config

def get_ml_pipeline_config(environment: str = 'development') -> MLPipelineConfig:
    """ML Pipeline設定を取得"""
    return MLPipelineConfig(
        model_path=os.getenv('ML_MODEL_PATH', './backend/ml-pipeline/models/'),
        embedding_dimension=int(os.getenv('ML_EMBEDDING_DIMENSION', '768')),
        two_tower_hidden_units=[int(x) for x in os.getenv('TWO_TOWER_HIDDEN_UNITS', '512,256,128').split(',')],
        learning_rate=float(os.getenv('ML_LEARNING_RATE', '0.001')),
        batch_size=int(os.getenv('ML_BATCH_SIZE', '128')),
        epochs=int(os.getenv('TWO_TOWER_EPOCHS', '50')),
        validation_split=float(os.getenv('TWO_TOWER_VALIDATION_SPLIT', '0.2')),
        early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE', '5')),
        similarity_threshold=float(os.getenv('ML_SIMILARITY_THRESHOLD', '0.1')),
        diversity_weight=float(os.getenv('ML_DIVERSITY_WEIGHT', '0.3'))
    )

def get_monitoring_config(environment: str = 'development') -> MonitoringConfig:
    """監視設定を取得"""
    base_config = MonitoringConfig(
        log_level=os.getenv('LOG_LEVEL', 'DEBUG' if environment == 'development' else 'INFO'),
        enable_performance_logging=os.getenv('ENABLE_PERFORMANCE_LOGGING', 'true').lower() == 'true',
        enable_error_tracking=os.getenv('ENABLE_ERROR_TRACKING', 'true').lower() == 'true',
        metrics_collection_interval=int(os.getenv('METRICS_COLLECTION_INTERVAL_MS', '60000')),
        log_rotation_max_files=int(os.getenv('LOG_ROTATION_MAX_FILES', '10')),
        log_rotation_max_size_mb=int(os.getenv('LOG_ROTATION_MAX_SIZE_MB', '100'))
    )
    
    return base_config

def get_data_processing_config(environment: str = 'development') -> DataProcessingConfig:
    """データ処理統合設定を取得"""
    return DataProcessingConfig(
        scraping=get_scraping_config(environment),
        cleaning=get_cleaning_config(environment),
        embedding=get_embedding_config(environment),
        validation=get_validation_config(environment),
        database=get_database_config(environment),
        monitoring=get_monitoring_config(environment)
    )

# ============================================================================
# 設定検証関数
# ============================================================================

def validate_config(config: DataProcessingConfig) -> Dict[str, Any]:
    """設定の妥当性検証"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # 必須環境変数のチェック
    required_env_vars = [
        'DMM_API_ID',
        'DMM_AFFILIATE_ID',
        'DATABASE_URL'
    ]
    
    for env_var in required_env_vars:
        if not os.getenv(env_var):
            validation_results['errors'].append(f'必須環境変数が設定されていません: {env_var}')
            validation_results['is_valid'] = False
    
    # 数値範囲のチェック
    if config.scraping.request_delay_ms < 500:
        validation_results['warnings'].append(
            'リクエスト間隔が短すぎます（推奨: 1000ms以上）'
        )
    
    if config.embedding.batch_size > 1000:
        validation_results['warnings'].append(
            'エンベディングバッチサイズが大きすぎます（推奨: 500以下）'
        )
    
    # データベース接続のテスト
    try:
        import psycopg2
        conn = psycopg2.connect(config.database.url)
        conn.close()
    except Exception as e:
        validation_results['errors'].append(f'データベース接続エラー: {str(e)}')
        validation_results['is_valid'] = False
    
    return validation_results

def print_config_summary(config: DataProcessingConfig, environment: str):
    """設定サマリーの表示"""
    print(f"\n=== データ処理設定サマリー (環境: {environment}) ===")
    print(f"スクレイピング:")
    print(f"  - リクエスト間隔: {config.scraping.request_delay_ms}ms")
    print(f"  - 同時接続数: {config.scraping.max_concurrent_requests}")
    print(f"  - DMM API設定: {'✓' if config.scraping.dmm_api_id else '✗'}")
    
    print(f"データクリーニング:")
    print(f"  - 重複判定閾値: {config.cleaning.duplicate_threshold}")
    print(f"  - 必須フィールド数: {len(config.cleaning.required_fields)}")
    
    print(f"エンベディング:")
    print(f"  - バッチサイズ: {config.embedding.batch_size}")
    print(f"  - ベクトル次元: {config.embedding.vector_dimension}")
    
    print(f"データベース:")
    print(f"  - プールサイズ: {config.database.pool_size}")
    print(f"  - クエリログ: {'有効' if config.database.enable_query_logging else '無効'}")
    
    print(f"監視:")
    print(f"  - ログレベル: {config.monitoring.log_level}")
    print(f"  - パフォーマンス監視: {'有効' if config.monitoring.enable_performance_logging else '無効'}")
    print("=" * 50)

if __name__ == "__main__":
    # 設定テスト
    import sys
    
    env = sys.argv[1] if len(sys.argv) > 1 else 'development'
    config = get_data_processing_config(env)
    
    print_config_summary(config, env)
    
    validation = validate_config(config)
    
    print(f"\n設定検証結果:")
    print(f"有効: {'✓' if validation['is_valid'] else '✗'}")
    
    if validation['errors']:
        print("エラー:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("警告:")
        for warning in validation['warnings']:
            print(f"  - {warning}")