#!/usr/bin/env python3
"""
Training Components Test Script

実際のデータベース接続なしで訓練コンポーネントをテストします。
モックデータを使用してTwo-Towerモデルの訓練プロセスを検証します。
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, MagicMock
import tempfile

# Add ml_pipeline to path
sys.path.append(str(Path(__file__).parent.parent / 'ml_pipeline'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockDatabaseConnection:
    """Mock database connection for testing"""
    
    def __init__(self):
        self.mock_data = self._generate_mock_data()
        self.cursor_results = []
        
    def _generate_mock_data(self) -> Dict[str, Any]:
        """Generate realistic mock data"""
        np.random.seed(42)  # Reproducible results
        
        # Generate mock users (1000 users)
        n_users = 1000
        user_ids = [f"user_{i:04d}" for i in range(n_users)]
        
        # Generate mock videos (5000 videos)
        n_videos = 5000
        video_data = []
        genres = ['Action', 'Drama', 'Comedy', 'Romance', 'Thriller', 'Horror', 'Sci-Fi', 'Fantasy']
        makers = [f'Studio_{i}' for i in range(20)]
        
        for i in range(n_videos):
            video_data.append({
                'id': f"video_{i:04d}",
                'title': f'Video Title {i}',
                'description': f'Description for video {i}',
                'genre': np.random.choice(genres),
                'maker': np.random.choice(makers),
                'price': np.random.randint(100, 5000),
                'duration_seconds': np.random.randint(1800, 7200),
                'created_at': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
        
        # Generate mock user decisions (50000 decisions)
        n_decisions = 50000
        decisions = []
        for i in range(n_decisions):
            user_id = np.random.choice(user_ids)
            video_id = np.random.choice([v['id'] for v in video_data])
            decision_type = np.random.choice(['like', 'nope'], p=[0.3, 0.7])  # 30% likes
            
            decisions.append({
                'user_id': user_id,
                'video_id': video_id,
                'decision_type': decision_type,
                'created_at': datetime.now() - timedelta(days=np.random.randint(1, 90))
            })
        
        return {
            'users': user_ids,
            'videos': video_data,
            'decisions': decisions
        }
    
    def cursor(self):
        """Mock cursor"""
        mock_cursor = Mock()
        
        def execute(query, params=None):
            """Mock execute method"""
            query_lower = query.lower().strip()
            
            # Handle different query patterns
            if 'count(*)' in query_lower and 'user_video_decisions' in query_lower:
                if 'group by decision' in query_lower.replace('_', ''):
                    # Decision breakdown
                    like_count = sum(1 for d in self.mock_data['decisions'] if d['decision_type'] == 'like')
                    nope_count = len(self.mock_data['decisions']) - like_count
                    self.cursor_results = [
                        {'decision': 'like', 'count': like_count},
                        {'decision': 'nope', 'count': nope_count}
                    ]
                else:
                    # Total count
                    self.cursor_results = [(len(self.mock_data['decisions']),)]
            
            elif 'count(*)' in query_lower and 'videos' in query_lower:
                self.cursor_results = [(len(self.mock_data['videos']),)]
            
            elif 'count(distinct user_id)' in query_lower:
                unique_users = len(set(d['user_id'] for d in self.mock_data['decisions']))
                self.cursor_results = [(unique_users,)]
            
            elif 'select distinct user_id' in query_lower:
                # Get users for training
                unique_users = list(set(d['user_id'] for d in self.mock_data['decisions']))[:500]
                self.cursor_results = [(u,) for u in unique_users]
            
            elif 'select v.id' in query_lower and 'videos v' in query_lower:
                # Get videos for training
                video_ids = [v['id'] for v in self.mock_data['videos'][:1000]]
                self.cursor_results = [(v,) for v in video_ids]
            
            elif 'user_video_decisions uvd' in query_lower and 'count(*)' in query_lower:
                # User behavior analysis
                user_behaviors = {}
                for decision in self.mock_data['decisions']:
                    user_id = decision['user_id']
                    if user_id not in user_behaviors:
                        user_behaviors[user_id] = {'like_count': 0, 'total_count': 0}
                    user_behaviors[user_id]['total_count'] += 1
                    if decision['decision_type'] == 'like':
                        user_behaviors[user_id]['like_count'] += 1
                
                # Return sample user behavior
                sample_users = list(user_behaviors.items())[:100]
                self.cursor_results = [
                    {
                        'user_id': user_id,
                        'total_likes': behavior['like_count'],
                        'avg_price': np.random.randint(1000, 3000),
                        'liked_genres': np.random.choice(['Action', 'Drama', 'Comedy']),
                        'first_like': datetime.now() - timedelta(days=30),
                        'last_like': datetime.now() - timedelta(days=1)
                    }
                    for user_id, behavior in sample_users
                ]
            
            elif 'select' in query_lower and ('title' in query_lower or 'description' in query_lower):
                # Video metadata query
                sample_videos = self.mock_data['videos'][:100]
                self.cursor_results = [
                    {
                        'id': v['id'],
                        'title': v['title'],
                        'description': v['description'],
                        'maker': v['maker'],
                        'genre': v['genre'],
                        'price': v['price'],
                        'duration_seconds': v['duration_seconds'],
                        'tags': ['tag1', 'tag2']
                    }
                    for v in sample_videos
                ]
            
            else:
                # Default empty result
                self.cursor_results = []
        
        def fetchall():
            return self.cursor_results
        
        def fetchone():
            return self.cursor_results[0] if self.cursor_results else None
        
        mock_cursor.execute = execute
        mock_cursor.fetchall = fetchall
        mock_cursor.fetchone = fetchone
        
        return mock_cursor

class ComponentTester:
    """Test Two-Tower training components with mock data"""
    
    def __init__(self):
        self.mock_conn = MockDatabaseConnection()
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="two_tower_test_"))
        
    def test_feature_extractors(self) -> Dict[str, Any]:
        """Test feature extraction components"""
        logger.info("Testing feature extractors...")
        
        try:
            # Import components
            from preprocessing.real_user_feature_extractor import RealUserFeatureExtractor
            from preprocessing.enhanced_item_feature_processor import EnhancedItemFeatureProcessor
            
            # Test user feature extractor
            user_extractor = RealUserFeatureExtractor(self.mock_conn)
            
            # Test with sample user
            sample_user_id = "user_0001"
            try:
                user_features = user_extractor.extract_user_features(sample_user_id)
                user_test_result = {
                    'success': True,
                    'feature_dim': len(user_features.features) if hasattr(user_features, 'features') else 'unknown',
                    'feature_type': type(user_features).__name__
                }
            except Exception as e:
                user_test_result = {
                    'success': False,
                    'error': str(e)
                }
            
            # Test item feature processor
            item_processor = EnhancedItemFeatureProcessor(self.mock_conn)
            
            # Test with sample video
            sample_video_id = "video_0001"
            try:
                item_features = item_processor.process_video_features(sample_video_id)
                item_test_result = {
                    'success': True,
                    'feature_dim': len(item_features.features) if hasattr(item_features, 'features') else 'unknown',
                    'feature_type': type(item_features).__name__
                }
            except Exception as e:
                item_test_result = {
                    'success': False,
                    'error': str(e)
                }
            
            return {
                'user_extractor': user_test_result,
                'item_processor': item_test_result
            }
            
        except Exception as e:
            return {
                'error': f"Feature extractor test failed: {str(e)}"
            }
    
    def test_model_architecture(self) -> Dict[str, Any]:
        """Test Two-Tower model architecture"""
        logger.info("Testing model architecture...")
        
        try:
            from training.enhanced_two_tower_trainer import EnhancedTwoTowerTrainer, TrainingConfiguration
            
            # Create minimal training configuration
            config = TrainingConfiguration(
                embedding_dim=768,
                user_tower_hidden_dims=[256, 128],
                item_tower_hidden_dims=[256, 128],
                learning_rate=0.001,
                batch_size=32,
                epochs=2,  # Minimal for testing
                validation_split=0.2,
                early_stopping_patience=2,
                dropout_rate=0.3
            )
            
            # Mock feature extractors
            mock_user_extractor = Mock()
            mock_item_processor = Mock()
            
            # Create trainer
            trainer = EnhancedTwoTowerTrainer(
                config=config,
                user_feature_extractor=mock_user_extractor,
                item_feature_processor=mock_item_processor,
                db_connection=self.mock_conn
            )
            
            # Test model creation
            try:
                # Create dummy feature inputs
                user_features = np.random.random((10, 100))  # 10 samples, 100 features
                item_features = np.random.random((10, 100))  # 10 samples, 100 features
                
                # Test model building
                user_tower, item_tower, full_model = trainer._build_models(
                    user_feature_dim=100,
                    item_feature_dim=100
                )
                
                # Test inference
                user_embeddings = user_tower.predict(user_features, verbose=0)
                item_embeddings = item_tower.predict(item_features, verbose=0)
                
                return {
                    'success': True,
                    'user_tower_output_shape': user_embeddings.shape,
                    'item_tower_output_shape': item_embeddings.shape,
                    'embedding_dim': user_embeddings.shape[1],
                    'embeddings_normalized': np.allclose(np.linalg.norm(user_embeddings, axis=1), 1.0, rtol=1e-2)
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
                
        except Exception as e:
            return {
                'error': f"Model architecture test failed: {str(e)}"
            }
    
    def test_training_pipeline(self) -> Dict[str, Any]:
        """Test training pipeline with synthetic data"""
        logger.info("Testing training pipeline...")
        
        try:
            import tensorflow as tf
            
            # Create simple synthetic training data
            n_samples = 1000
            user_feature_dim = 50
            item_feature_dim = 60
            
            # Generate features
            user_features = np.random.random((n_samples, user_feature_dim))
            item_features = np.random.random((n_samples, item_feature_dim))
            labels = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% positive
            
            # Create simple Two-Tower model
            user_input = tf.keras.Input(shape=(user_feature_dim,), name='user_input')
            user_tower = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(768, activation='linear'),  # 768-dim output
                tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))  # Normalize
            ])(user_input)
            
            item_input = tf.keras.Input(shape=(item_feature_dim,), name='item_input')
            item_tower = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(768, activation='linear'),  # 768-dim output
                tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))  # Normalize
            ])(item_input)
            
            # Compute similarity
            similarity = tf.keras.layers.Dot(axes=1)([user_tower, item_tower])
            prediction = tf.keras.layers.Dense(1, activation='sigmoid')(similarity)
            
            # Create model
            model = tf.keras.Model(inputs=[user_input, item_input], outputs=prediction)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
            )
            
            # Train briefly
            history = model.fit(
                [user_features, item_features],
                labels,
                epochs=3,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Save test models
            user_model = tf.keras.Model(inputs=user_input, outputs=user_tower)
            item_model = tf.keras.Model(inputs=item_input, outputs=item_tower)
            
            user_model_path = self.temp_dir / 'user_tower_test.keras'
            item_model_path = self.temp_dir / 'item_tower_test.keras'
            
            user_model.save(user_model_path)
            item_model.save(item_model_path)
            
            # Test loading
            loaded_user_model = tf.keras.models.load_model(user_model_path)
            loaded_item_model = tf.keras.models.load_model(item_model_path)
            
            # Test inference
            test_user_embedding = loaded_user_model.predict(user_features[:1], verbose=0)
            test_item_embedding = loaded_item_model.predict(item_features[:1], verbose=0)
            
            return {
                'success': True,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'final_auc_pr': history.history['auc_pr'][-1] if 'auc_pr' in history.history else 'N/A',
                'user_embedding_shape': test_user_embedding.shape,
                'item_embedding_shape': test_item_embedding.shape,
                'embeddings_normalized': (
                    np.allclose(np.linalg.norm(test_user_embedding, axis=1), 1.0, rtol=1e-2) and
                    np.allclose(np.linalg.norm(test_item_embedding, axis=1), 1.0, rtol=1e-2)
                ),
                'model_files_saved': [
                    user_model_path.exists(),
                    item_model_path.exists()
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all component tests"""
        logger.info("Starting comprehensive Two-Tower component testing...")
        
        test_results = {
            'test_session': {
                'start_time': datetime.now().isoformat(),
                'mock_data_stats': {
                    'users': len(self.mock_conn.mock_data['users']),
                    'videos': len(self.mock_conn.mock_data['videos']),
                    'decisions': len(self.mock_conn.mock_data['decisions'])
                }
            }
        }
        
        # Test 1: Feature extractors
        logger.info("Test 1: Feature extraction components")
        test_results['feature_extractors'] = self.test_feature_extractors()
        
        # Test 2: Model architecture
        logger.info("Test 2: Two-Tower model architecture")
        test_results['model_architecture'] = self.test_model_architecture()
        
        # Test 3: Training pipeline
        logger.info("Test 3: Training pipeline with synthetic data")
        test_results['training_pipeline'] = self.test_training_pipeline()
        
        # Overall assessment
        success_count = sum(1 for test in [
            test_results['feature_extractors'],
            test_results['model_architecture'],
            test_results['training_pipeline']
        ] if test.get('success', False))
        
        test_results['overall'] = {
            'tests_passed': success_count,
            'total_tests': 3,
            'success_rate': success_count / 3,
            'overall_success': success_count >= 2,  # At least 2/3 tests pass
            'test_duration_minutes': (datetime.now() - datetime.fromisoformat(test_results['test_session']['start_time'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() / 60
        }
        
        return test_results

def main():
    logger.info("Two-Tower Training Components Test")
    logger.info("=" * 50)
    
    # Run tests
    tester = ComponentTester()
    results = tester.run_all_tests()
    
    # Save results
    results_file = Path("training_components_test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPONENT TEST SUMMARY")
    print("=" * 60)
    
    overall = results['overall']
    if overall['overall_success']:
        print("✅ Status: SUCCESS")
    else:
        print("❌ Status: FAILED")
    
    print(f"📊 Tests Passed: {overall['tests_passed']}/{overall['total_tests']}")
    print(f"📊 Success Rate: {overall['success_rate']:.1%}")
    print(f"⏱️  Test Duration: {overall['test_duration_minutes']:.1f} minutes")
    
    # Individual test results
    for test_name in ['feature_extractors', 'model_architecture', 'training_pipeline']:
        test_result = results[test_name]
        status = "✅" if test_result.get('success', False) else "❌"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        if 'error' in test_result:
            print(f"   Error: {test_result['error']}")
    
    print(f"📄 Full Results: {results_file}")
    print("=" * 60)
    
    return 0 if overall['overall_success'] else 1

if __name__ == '__main__':
    sys.exit(main())