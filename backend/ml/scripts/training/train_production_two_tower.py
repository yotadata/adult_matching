#!/usr/bin/env python3
"""
Production Two-Tower Model Training Script

実装済みの訓練パイプラインを使用して実際にTwo-Towerモデルを訓練し、
768次元の訓練済みモデルファイルを生成します。

Usage:
    python scripts/train_production_two_tower.py --config config.yaml
    python scripts/train_production_two_tower.py --quick-test  # クイックテスト用
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Import our implemented components from new structure
from backend.ml.training.production_trainer import ProductionTrainer
from backend.ml.training.enhanced_two_tower_trainer import EnhancedTwoTowerTrainer
from backend.ml.preprocessing.features.user_feature_processor import UserFeatureProcessor
from backend.ml.preprocessing.features.item_feature_processor import ItemFeatureProcessor
from backend.ml.preprocessing.embeddings.embedding_manager import EmbeddingManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class ProductionTwoTowerTrainingOrchestrator:
    """Production Two-Tower model training orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = datetime.now()
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / 'models'
        self.data_dir = self.project_root / 'data'
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Database connection
        self.db_url = config.get('db_url') or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("Database URL not provided. Set DATABASE_URL or pass --db-url")
        
        # Training components (will be initialized)
        self.conn = None
        self.production_trainer = None
        self.enhanced_trainer = None
        
        # Training results
        self.training_results = {}
    
    def connect_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                self.db_url,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to database successfully")
            
            # Test connection and get basic stats
            cursor = self.conn.cursor()
            
            # Check user_video_decisions table
            cursor.execute("SELECT COUNT(*) FROM user_video_decisions")
            user_decisions_count = cursor.fetchone()[0]
            
            # Check videos table
            cursor.execute("SELECT COUNT(*) FROM videos")
            videos_count = cursor.fetchone()[0]
            
            # Check unique users
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_video_decisions")
            unique_users = cursor.fetchone()[0]
            
            logger.info(f"Database stats: {user_decisions_count} decisions, {unique_users} users, {videos_count} videos")
            
            if user_decisions_count < 1000:
                logger.warning("Very few user decisions available for training")
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def check_data_availability(self) -> Dict[str, Any]:
        """Check availability of training data"""
        cursor = self.conn.cursor()
        
        # Get data statistics
        stats = {}
        
        # User decisions breakdown
        cursor.execute("""
            SELECT 
                decision,
                COUNT(*) as count
            FROM user_video_decisions 
            GROUP BY decision
        """)
        decision_stats = {row['decision']: row['count'] for row in cursor.fetchall()}
        
        # Recent activity (last 30 days)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM user_video_decisions 
            WHERE created_at >= NOW() - INTERVAL '30 days'
        """)
        recent_decisions = cursor.fetchone()[0]
        
        # User activity distribution
        cursor.execute("""
            SELECT 
                COUNT(*) as decision_count,
                COUNT(DISTINCT user_id) as user_count
            FROM (
                SELECT user_id, COUNT(*) as decisions
                FROM user_video_decisions 
                GROUP BY user_id
                HAVING COUNT(*) >= 5
            ) active_users
        """)
        active_user_stats = cursor.fetchone()
        
        stats = {
            'decision_breakdown': decision_stats,
            'recent_decisions_30d': recent_decisions,
            'active_users_5plus': active_user_stats['user_count'] if active_user_stats else 0,
            'total_decisions_active_users': active_user_stats['decision_count'] if active_user_stats else 0
        }
        
        logger.info(f"Data availability: {json.dumps(stats, indent=2)}")
        return stats
    
    def initialize_training_components(self):
        """Initialize training components with current data"""
        logger.info("Initializing training components...")
        
        # Initialize feature extractors
        user_feature_extractor = RealUserFeatureExtractor(self.conn)
        item_feature_processor = EnhancedItemFeatureProcessor(self.conn)
        
        # Training configuration
        training_config = TrainingConfiguration(
            embedding_dim=768,
            user_tower_hidden_dims=[512, 256, 128],
            item_tower_hidden_dims=[512, 256, 128],
            learning_rate=0.001,
            batch_size=self.config.get('batch_size', 512),
            epochs=self.config.get('epochs', 50),
            validation_split=0.2,
            early_stopping_patience=5,
            dropout_rate=0.3
        )
        
        # Enhanced trainer
        self.enhanced_trainer = EnhancedTwoTowerTrainer(
            config=training_config,
            user_feature_extractor=user_feature_extractor,
            item_feature_processor=item_feature_processor,
            db_connection=self.conn
        )
        
        # Production trainer configuration
        prod_config = ProductionTrainingConfig(
            db_url=self.db_url,
            model_save_path=str(self.models_dir),
            batch_size=self.config.get('batch_size', 512),
            epochs=self.config.get('epochs', 50),
            validation_split=0.2,
            early_stopping_patience=5,
            positive_negative_ratio=0.5,  # 1:2 ratio as specified
            temporal_split_months=2,
            memory_limit_gb=self.config.get('memory_limit_gb', 7.0),
            training_timeout_minutes=self.config.get('timeout_minutes', 55)
        )
        
        # Production trainer
        self.production_trainer = ProductionTrainer(prod_config)
        
        logger.info("Training components initialized")
    
    def run_data_preparation(self) -> Dict[str, Any]:
        """Prepare training data"""
        logger.info("Starting data preparation...")
        
        prep_start = time.time()
        
        try:
            # Use production trainer for data preparation
            training_data = self.production_trainer.prepare_training_data()
            
            prep_time = time.time() - prep_start
            
            prep_results = {
                'preparation_time_minutes': prep_time / 60,
                'training_samples': len(training_data['train_user_ids']),
                'validation_samples': len(training_data['val_user_ids']),
                'total_interactions': training_data.get('total_interactions', 0),
                'positive_ratio': training_data.get('positive_ratio', 0.0)
            }
            
            logger.info(f"Data preparation completed: {json.dumps(prep_results, indent=2)}")
            return prep_results
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def run_model_training(self) -> Dict[str, Any]:
        """Run actual model training"""
        logger.info("Starting model training...")
        
        training_start = time.time()
        
        try:
            # Run production training with monitoring
            training_metrics = self.production_trainer.train_with_monitoring()
            
            training_time = time.time() - training_start
            
            # Extract key metrics
            training_results = {
                'training_time_minutes': training_time / 60,
                'final_loss': training_metrics.final_loss,
                'final_auc_pr': training_metrics.final_auc_pr,
                'final_auc_roc': training_metrics.final_auc_roc,
                'epochs_completed': training_metrics.epochs_completed,
                'early_stopping_triggered': training_metrics.early_stopped,
                'peak_memory_gb': training_metrics.peak_memory_gb,
                'model_saved': training_metrics.model_saved
            }
            
            # Check success criteria
            success_criteria = {
                'auc_pr_target': training_results['final_auc_pr'] > 0.99,
                'training_time_ok': training_results['training_time_minutes'] < 60,
                'memory_usage_ok': training_results['peak_memory_gb'] < 8.0,
                'model_files_exist': self._check_model_files_exist()
            }
            
            training_results['success_criteria'] = success_criteria
            training_results['training_successful'] = all(success_criteria.values())
            
            logger.info(f"Training completed: {json.dumps(training_results, indent=2)}")
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _check_model_files_exist(self) -> Dict[str, bool]:
        """Check if model files were properly generated"""
        model_files = {
            'user_tower': (self.models_dir / 'user_tower.keras').exists(),
            'item_tower': (self.models_dir / 'item_tower.keras').exists(),
            'full_model': (self.models_dir / 'two_tower_model.keras').exists() or 
                         (self.models_dir / 'enhanced_two_tower_model.keras').exists()
        }
        
        logger.info(f"Model files status: {model_files}")
        return model_files
    
    def run_model_validation(self) -> Dict[str, Any]:
        """Validate trained models"""
        logger.info("Starting model validation...")
        
        try:
            # Load and test models
            import tensorflow as tf
            
            validation_results = {}
            
            # Check user tower
            user_tower_path = self.models_dir / 'user_tower.keras'
            if user_tower_path.exists():
                user_tower = tf.keras.models.load_model(user_tower_path)
                
                # Test with dummy input
                dummy_user_features = np.random.random((1, user_tower.input_shape[1]))
                user_embedding = user_tower.predict(dummy_user_features, verbose=0)
                
                validation_results['user_tower'] = {
                    'loadable': True,
                    'input_shape': user_tower.input_shape,
                    'output_shape': user_embedding.shape,
                    'embedding_dim': user_embedding.shape[1],
                    'embedding_normalized': np.allclose(np.linalg.norm(user_embedding, axis=1), 1.0, rtol=1e-3)
                }
            
            # Check item tower
            item_tower_path = self.models_dir / 'item_tower.keras'
            if item_tower_path.exists():
                item_tower = tf.keras.models.load_model(item_tower_path)
                
                # Test with dummy input
                dummy_item_features = np.random.random((1, item_tower.input_shape[1]))
                item_embedding = item_tower.predict(dummy_item_features, verbose=0)
                
                validation_results['item_tower'] = {
                    'loadable': True,
                    'input_shape': item_tower.input_shape,
                    'output_shape': item_embedding.shape,
                    'embedding_dim': item_embedding.shape[1],
                    'embedding_normalized': np.allclose(np.linalg.norm(item_embedding, axis=1), 1.0, rtol=1e-3)
                }
            
            # Overall validation
            validation_results['overall'] = {
                'both_towers_exist': 'user_tower' in validation_results and 'item_tower' in validation_results,
                'consistent_embedding_dim': (
                    validation_results.get('user_tower', {}).get('embedding_dim') == 
                    validation_results.get('item_tower', {}).get('embedding_dim') == 768
                ),
                'embeddings_normalized': (
                    validation_results.get('user_tower', {}).get('embedding_normalized', False) and
                    validation_results.get('item_tower', {}).get('embedding_normalized', False)
                )
            }
            
            logger.info(f"Model validation completed: {json.dumps(validation_results, indent=2)}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise
    
    def save_training_report(self):
        """Save comprehensive training report"""
        report = {
            'training_session': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'config': self.config
            },
            'results': self.training_results
        }
        
        report_path = self.models_dir / f'training_report_{self.start_time.strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training report saved to: {report_path}")
        return report_path
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        logger.info("Starting full Two-Tower model training pipeline...")
        
        try:
            # 1. Connect to database
            self.connect_database()
            
            # 2. Check data availability
            data_stats = self.check_data_availability()
            self.training_results['data_stats'] = data_stats
            
            # 3. Initialize components
            self.initialize_training_components()
            
            # 4. Data preparation
            prep_results = self.run_data_preparation()
            self.training_results['data_preparation'] = prep_results
            
            # 5. Model training
            training_results = self.run_model_training()
            self.training_results['training'] = training_results
            
            # 6. Model validation
            validation_results = self.run_model_validation()
            self.training_results['validation'] = validation_results
            
            # 7. Save report
            report_path = self.save_training_report()
            self.training_results['report_path'] = str(report_path)
            
            # Final success check
            overall_success = (
                training_results.get('training_successful', False) and
                validation_results.get('overall', {}).get('both_towers_exist', False) and
                validation_results.get('overall', {}).get('consistent_embedding_dim', False)
            )
            
            self.training_results['overall_success'] = overall_success
            
            if overall_success:
                logger.info("✅ Two-Tower model training completed successfully!")
            else:
                logger.warning("⚠️ Training completed with issues - check results")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            self.training_results['error'] = str(e)
            raise
        finally:
            if self.conn:
                self.conn.close()

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load training configuration"""
    default_config = {
        'batch_size': 512,
        'epochs': 50,
        'memory_limit_gb': 7.0,
        'timeout_minutes': 55
    }
    
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        default_config.update(file_config)
    
    return default_config

def main():
    parser = argparse.ArgumentParser(description='Production Two-Tower Model Training')
    parser.add_argument('--config', help='Training configuration YAML file')
    parser.add_argument('--db-url', help='PostgreSQL database URL')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with minimal data')
    parser.add_argument('--batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--memory-limit', type=float, default=7.0, help='Memory limit in GB')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line args
    if args.db_url:
        config['db_url'] = args.db_url
    if args.batch_size != 512:
        config['batch_size'] = args.batch_size
    if args.epochs != 50:
        config['epochs'] = args.epochs
    if args.memory_limit != 7.0:
        config['memory_limit_gb'] = args.memory_limit
        
    # Quick test configuration
    if args.quick_test:
        config.update({
            'batch_size': 64,
            'epochs': 5,
            'timeout_minutes': 10
        })
        logger.info("Running in quick test mode")
    
    # Initialize and run training
    orchestrator = ProductionTwoTowerTrainingOrchestrator(config)
    
    try:
        results = orchestrator.run_full_training_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if results.get('overall_success'):
            print("✅ Status: SUCCESS")
        else:
            print("❌ Status: FAILED")
        
        training_info = results.get('training', {})
        print(f"📊 Final AUC-PR: {training_info.get('final_auc_pr', 'N/A'):.4f}")
        print(f"📊 Final AUC-ROC: {training_info.get('final_auc_roc', 'N/A'):.4f}")
        print(f"⏱️  Training Time: {training_info.get('training_time_minutes', 0):.1f} minutes")
        print(f"💾 Peak Memory: {training_info.get('peak_memory_gb', 0):.2f} GB")
        
        model_files = results.get('validation', {}).get('overall', {})
        print(f"📁 Model Files: {'✅' if model_files.get('both_towers_exist') else '❌'}")
        print(f"📐 768-dim Compatible: {'✅' if model_files.get('consistent_embedding_dim') else '❌'}")
        
        print(f"📄 Report: {results.get('report_path', 'N/A')}")
        print("="*60)
        
        return 0 if results.get('overall_success') else 1
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())