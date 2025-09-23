"""
Performance Optimization Package

パフォーマンス最適化パッケージ
- 推薦システム最適化
- MLトレーニング最適化  
- データベースクエリ最適化
- キャッシュ戦略
"""

from .recommendation_optimizer import RecommendationOptimizer, create_recommendation_optimizer
from .ml_training_optimizer import MLTrainingOptimizer, create_ml_training_optimizer
from .database_optimizer import DatabaseOptimizer, create_database_optimizer
from .performance_verifier import PerformanceVerifier, PerformanceReport, BenchmarkResult

__version__ = "1.0.0"
__author__ = "Adult Matching Performance Team"

__all__ = [
    "RecommendationOptimizer",
    "MLTrainingOptimizer", 
    "DatabaseOptimizer",
    "PerformanceVerifier",
    "PerformanceReport",
    "BenchmarkResult",
    "create_recommendation_optimizer",
    "create_ml_training_optimizer",
    "create_database_optimizer",
    "create_unified_optimizer"
]

def create_unified_optimizer(config=None):
    """
    統合パフォーマンス最適化システムの作成
    
    Args:
        config: 最適化設定辞書
        
    Returns:
        tuple: (RecommendationOptimizer, MLTrainingOptimizer, DatabaseOptimizer)
    """
    recommendation_optimizer = create_recommendation_optimizer(config)
    ml_training_optimizer = create_ml_training_optimizer(config)
    database_optimizer = create_database_optimizer(config)
    
    return recommendation_optimizer, ml_training_optimizer, database_optimizer