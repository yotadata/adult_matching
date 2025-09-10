#!/usr/bin/env python3
"""
Simple Two-Tower Model Test

実装したコンポーネントのデバッグなしに、直接Two-Towerモデルを構築・訓練し、
768次元埋め込みの生成をテストします。
"""

import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import tempfile

# Enable unsafe deserialization for testing
tf.keras.config.enable_unsafe_deserialization()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTwoTowerTrainer:
    """シンプルなTwo-Towerモデル訓練器"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.models_dir = Path(tempfile.mkdtemp(prefix="two_tower_models_"))
        self.models_dir.mkdir(exist_ok=True)
        
        # Models
        self.user_tower = None
        self.item_tower = None
        self.full_model = None
        
    def create_user_tower(self, input_dim: int) -> tf.keras.Model:
        """ユーザータワーの作成"""
        inputs = tf.keras.Input(shape=(input_dim,), name='user_input')
        
        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # 768次元埋め込み
        embeddings = tf.keras.layers.Dense(self.embedding_dim, activation='linear', name='user_embeddings')(x)
        
        # L2正規化 (Lambda層の代わりに独立したレイヤーを使用)
        normalized_embeddings = tf.keras.utils.normalize(embeddings, axis=1)
        
        model = tf.keras.Model(inputs=inputs, outputs=normalized_embeddings, name='user_tower')
        return model
    
    def create_item_tower(self, input_dim: int) -> tf.keras.Model:
        """アイテムタワーの作成"""
        inputs = tf.keras.Input(shape=(input_dim,), name='item_input')
        
        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # 768次元埋め込み
        embeddings = tf.keras.layers.Dense(self.embedding_dim, activation='linear', name='item_embeddings')(x)
        
        # L2正規化
        normalized_embeddings = tf.keras.utils.normalize(embeddings, axis=1)
        
        model = tf.keras.Model(inputs=inputs, outputs=normalized_embeddings, name='item_tower')
        return model
    
    def create_two_tower_model(self, user_input_dim: int, item_input_dim: int) -> tf.keras.Model:
        """Two-Towerモデルの作成"""
        # Individual towers
        self.user_tower = self.create_user_tower(user_input_dim)
        self.item_tower = self.create_item_tower(item_input_dim)
        
        # Full model inputs
        user_input = tf.keras.Input(shape=(user_input_dim,), name='user_features')
        item_input = tf.keras.Input(shape=(item_input_dim,), name='item_features')
        
        # Get embeddings
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        
        # Compute similarity (dot product)
        similarity = tf.keras.layers.Dot(axes=1, name='similarity')([user_embedding, item_embedding])
        
        # Final prediction
        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(similarity)
        
        # Full model
        full_model = tf.keras.Model(
            inputs=[user_input, item_input],
            outputs=prediction,
            name='two_tower_model'
        )
        
        return full_model
    
    def generate_synthetic_data(self, n_samples: int = 5000) -> tuple:
        """合成データの生成"""
        np.random.seed(42)
        
        # User features (100 dimensions)
        user_features = np.random.random((n_samples, 100))
        
        # Item features (120 dimensions)
        item_features = np.random.random((n_samples, 120))
        
        # Labels (30% positive interactions)
        labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        logger.info(f"Generated {n_samples} synthetic samples:")
        logger.info(f"- User features: {user_features.shape}")
        logger.info(f"- Item features: {item_features.shape}")
        logger.info(f"- Positive rate: {labels.mean():.1%}")
        
        return user_features, item_features, labels
    
    def train_model(self, user_features, item_features, labels, epochs: int = 10):
        """モデル訓練"""
        # Create model
        self.full_model = self.create_two_tower_model(
            user_features.shape[1], 
            item_features.shape[1]
        )
        
        # Compile model
        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc_roc'), tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
        )
        
        # Display model summary
        logger.info("Two-Tower Model Architecture:")
        self.full_model.summary()
        
        # Train
        logger.info("Starting model training...")
        history = self.full_model.fit(
            [user_features, item_features],
            labels,
            epochs=epochs,
            batch_size=256,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        return history
    
    def save_models(self) -> dict:
        """モデル保存"""
        model_files = {}
        
        try:
            # Save user tower
            user_tower_path = self.models_dir / 'user_tower.keras'
            self.user_tower.save(user_tower_path)
            model_files['user_tower'] = str(user_tower_path)
            logger.info(f"User tower saved: {user_tower_path}")
            
            # Save item tower
            item_tower_path = self.models_dir / 'item_tower.keras'
            self.item_tower.save(item_tower_path)
            model_files['item_tower'] = str(item_tower_path)
            logger.info(f"Item tower saved: {item_tower_path}")
            
            # Save full model
            full_model_path = self.models_dir / 'two_tower_model.keras'
            self.full_model.save(full_model_path)
            model_files['full_model'] = str(full_model_path)
            logger.info(f"Full model saved: {full_model_path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise
        
        return model_files
    
    def validate_models(self, user_features, item_features) -> dict:
        """保存されたモデルの検証"""
        validation_results = {}
        
        try:
            # Load and test user tower
            user_tower_path = self.models_dir / 'user_tower.keras'
            loaded_user_tower = tf.keras.models.load_model(user_tower_path)
            
            test_user_embeddings = loaded_user_tower.predict(user_features[:10], verbose=0)
            
            validation_results['user_tower'] = {
                'loadable': True,
                'output_shape': test_user_embeddings.shape,
                'embedding_dim': test_user_embeddings.shape[1],
                'embeddings_normalized': np.allclose(
                    np.linalg.norm(test_user_embeddings, axis=1), 1.0, rtol=1e-2
                )
            }
            
            # Load and test item tower
            item_tower_path = self.models_dir / 'item_tower.keras'
            loaded_item_tower = tf.keras.models.load_model(item_tower_path)
            
            test_item_embeddings = loaded_item_tower.predict(item_features[:10], verbose=0)
            
            validation_results['item_tower'] = {
                'loadable': True,
                'output_shape': test_item_embeddings.shape,
                'embedding_dim': test_item_embeddings.shape[1],
                'embeddings_normalized': np.allclose(
                    np.linalg.norm(test_item_embeddings, axis=1), 1.0, rtol=1e-2
                )
            }
            
            # Test similarity computation
            similarities = np.sum(test_user_embeddings * test_item_embeddings, axis=1)
            
            validation_results['similarity'] = {
                'range': [float(similarities.min()), float(similarities.max())],
                'mean': float(similarities.mean()),
                'std': float(similarities.std())
            }
            
            # Overall validation
            validation_results['overall'] = {
                'both_towers_loadable': True,
                'consistent_embedding_dim': (
                    test_user_embeddings.shape[1] == test_item_embeddings.shape[1] == self.embedding_dim
                ),
                'embeddings_normalized': (
                    validation_results['user_tower']['embeddings_normalized'] and
                    validation_results['item_tower']['embeddings_normalized']
                )
            }
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results

def main():
    logger.info("Simple Two-Tower Model Training Test")
    logger.info("=" * 50)
    
    start_time = datetime.now()
    
    try:
        # Initialize trainer
        trainer = SimpleTwoTowerTrainer(embedding_dim=768)
        
        # Generate synthetic data
        user_features, item_features, labels = trainer.generate_synthetic_data(n_samples=10000)
        
        # Train model
        history = trainer.train_model(user_features, item_features, labels, epochs=15)
        
        # Get final metrics
        final_metrics = {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'auc_roc': history.history['auc_roc'][-1],
            'auc_pr': history.history['auc_pr'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1],
            'val_auc_roc': history.history['val_auc_roc'][-1],
            'val_auc_pr': history.history['val_auc_pr'][-1]
        }
        
        # Save models
        model_files = trainer.save_models()
        
        # Validate models
        validation_results = trainer.validate_models(user_features, item_features)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds() / 60
        
        # Compile results
        results = {
            'training_session': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': training_time,
                'samples': len(labels),
                'embedding_dim': trainer.embedding_dim
            },
            'final_metrics': final_metrics,
            'model_files': model_files,
            'validation': validation_results,
            'success_criteria': {
                'training_completed': True,
                'models_saved': len(model_files) == 3,
                'validation_passed': validation_results.get('overall', {}).get('both_towers_loadable', False),
                '768_dim_embeddings': validation_results.get('overall', {}).get('consistent_embedding_dim', False),
                'embeddings_normalized': validation_results.get('overall', {}).get('embeddings_normalized', False),
                'reasonable_performance': final_metrics.get('val_auc_pr', 0) > 0.5
            }
        }
        
        # Check overall success
        success_criteria = results['success_criteria']
        overall_success = all(success_criteria.values())
        results['overall_success'] = overall_success
        
        # Save results
        results_file = Path('simple_two_tower_test_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SIMPLE TWO-TOWER TEST SUMMARY")
        print("=" * 60)
        
        if overall_success:
            print("✅ Status: SUCCESS")
        else:
            print("❌ Status: FAILED")
        
        print(f"📊 Final AUC-PR: {final_metrics['val_auc_pr']:.4f}")
        print(f"📊 Final AUC-ROC: {final_metrics['val_auc_roc']:.4f}")
        print(f"📊 Final Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"⏱️  Training Time: {training_time:.1f} minutes")
        print(f"📁 Models Directory: {trainer.models_dir}")
        
        print("\nSuccess Criteria:")
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"{status} {criterion.replace('_', ' ').title()}")
        
        print(f"📄 Full Results: {results_file}")
        print("=" * 60)
        
        return 0 if overall_success else 1
        
    except Exception as e:
        logger.error(f"Training test failed: {e}")
        print(f"\n❌ FAILED: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())