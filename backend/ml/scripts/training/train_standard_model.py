#!/usr/bin/env python3
"""
Standard Two-Tower Model Training Script

標準768次元Two-Towerモデルのトレーニングスクリプト
統一されたトレーナーを使用した標準的なトレーニング実行
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.ml.training.trainers.unified_two_tower_trainer import UnifiedTwoTowerTrainer, TrainingConfig
from backend.ml.utils.logger import get_ml_logger
from backend.ml import PACKAGE_ROOT

logger = get_ml_logger(__name__)


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='Train standard Two-Tower recommendation model')
    
    parser.add_argument('--config', type=str, default=None,
                      help='Training configuration file path')
    parser.add_argument('--model-dir', type=str, default=None,
                      help='Model save directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Experiment name')
    
    # Data parameters
    parser.add_argument('--user-data', type=str, required=True,
                      help='User features data path')
    parser.add_argument('--item-data', type=str, required=True,
                      help='Item features data path')
    parser.add_argument('--interaction-data', type=str, required=True,
                      help='User-item interactions data path')
    
    # Quick configuration overrides
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=None,
                      help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=768,
                      help='Embedding dimension')
    
    # Output options
    parser.add_argument('--save-version', type=str, default='latest',
                      help='Model save version')
    parser.add_argument('--evaluate', action='store_true',
                      help='Run evaluation after training')
    
    return parser.parse_args()


def load_data(user_data_path: str, item_data_path: str, interaction_data_path: str):
    """データの読み込み"""
    import numpy as np
    import pandas as pd
    
    logger.info("Loading training data...")
    
    # For demonstration - in real implementation, load actual data
    # This is a placeholder that should be replaced with actual data loading logic
    
    # Mock data generation for testing
    logger.warning("Using mock data - replace with actual data loading")
    
    n_users = 1000
    n_items = 5000
    n_interactions = 10000
    
    # Generate mock user features (should be replaced with real data)
    user_features = np.random.randn(n_interactions, 50)  # 50 user features
    
    # Generate mock item features (should be replaced with real data)
    item_features = np.random.randn(n_interactions, 30)  # 30 item features
    
    # Generate mock interactions (should be replaced with real data)
    interactions = np.random.binomial(1, 0.1, n_interactions)  # 10% positive rate
    
    logger.info(f"Data loaded - Users: {n_users}, Items: {n_items}, Interactions: {n_interactions}")
    logger.info(f"User features shape: {user_features.shape}")
    logger.info(f"Item features shape: {item_features.shape}")
    logger.info(f"Interaction shape: {interactions.shape}")
    logger.info(f"Positive interaction rate: {interactions.mean():.3f}")
    
    return user_features, item_features, interactions


def main():
    """メイン実行関数"""
    args = parse_args()
    
    logger.info("Starting standard Two-Tower model training")
    logger.info(f"Arguments: {vars(args)}")
    
    # 設定の準備
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
        trainer = UnifiedTwoTowerTrainer(
            config=config_path,
            model_save_dir=args.model_dir,
            experiment_name=args.experiment_name
        )
    else:
        # デフォルト設定を使用し、引数で上書き
        config = TrainingConfig()
        
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.epochs:
            config.epochs = args.epochs
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        if args.embedding_dim:
            config.user_embedding_dim = args.embedding_dim
            config.item_embedding_dim = args.embedding_dim
        
        trainer = UnifiedTwoTowerTrainer(
            config=config,
            model_save_dir=args.model_dir,
            experiment_name=args.experiment_name
        )
    
    # データの読み込み
    try:
        user_features, item_features, interactions = load_data(
            args.user_data, args.item_data, args.interaction_data
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # トレーニング実行
    try:
        logger.info("Starting training...")
        history = trainer.train(user_features, item_features, interactions)
        logger.info("Training completed successfully")
        
        # モデル保存
        saved_files = trainer.save_model(version=args.save_version)
        logger.info(f"Model saved: {saved_files}")
        
        # 評価実行（オプション）
        if args.evaluate:
            logger.info("Running evaluation...")
            # For evaluation, we'll use a portion of the training data
            # In real implementation, use separate test data
            eval_results = trainer.evaluate_model(user_features, item_features, interactions)
            logger.info(f"Evaluation results: {eval_results}")
        
        logger.info("Training pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)