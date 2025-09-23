"""
ML Components Unit Tests

機械学習コンポーネントのユニットテスト
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml.preprocessing.features.user_feature_processor import UserFeatureProcessor
from backend.ml.preprocessing.features.item_feature_processor import ItemFeatureProcessor
from backend.ml.preprocessing.embeddings.embedding_manager import EmbeddingManager
from backend.ml.training.two_tower_trainer import TwoTowerTrainer


@pytest.mark.unit
class TestUserFeatureProcessor:
    """UserFeatureProcessor ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        processor = UserFeatureProcessor()
        assert processor is not None
        assert hasattr(processor, 'process_user_data')
    
    def test_process_user_data_basic(self, sample_user_data):
        """基本的なユーザーデータ処理テスト"""
        processor = UserFeatureProcessor()
        
        user_df = pd.DataFrame([sample_user_data])
        result = processor.process_user_data(user_df)
        
        assert isinstance(result, dict)
        assert 'user_features' in result
        assert 'feature_names' in result
        assert len(result['user_features']) > 0
    
    def test_extract_behavioral_features(self, sample_user_data):
        """行動特徴量抽出テスト"""
        processor = UserFeatureProcessor()
        
        # Mock interaction data
        interaction_data = pd.DataFrame({
            'user_id': [sample_user_data['id']] * 5,
            'video_id': ['video_1', 'video_2', 'video_3', 'video_4', 'video_5'],
            'action': ['like', 'view', 'like', 'skip', 'like'],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
        })
        
        features = processor.extract_behavioral_features(interaction_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0  # Has features
    
    def test_process_with_invalid_data(self):
        """無効なデータでの処理テスト"""
        processor = UserFeatureProcessor()
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            processor.process_user_data(empty_df)
    
    @pytest.mark.asyncio
    async def test_async_processing(self, sample_user_data):
        """非同期処理テスト"""
        processor = UserFeatureProcessor()
        user_df = pd.DataFrame([sample_user_data])
        
        # Mock async processing
        with patch.object(processor, 'process_user_data') as mock_process:
            mock_process.return_value = {'user_features': np.array([1, 2, 3])}
            
            result = await asyncio.create_task(
                asyncio.to_thread(processor.process_user_data, user_df)
            )
            
            assert result is not None
            mock_process.assert_called_once()


@pytest.mark.unit 
class TestItemFeatureProcessor:
    """ItemFeatureProcessor ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        processor = ItemFeatureProcessor()
        assert processor is not None
        assert hasattr(processor, 'process_item_data')
    
    def test_process_item_data_basic(self, sample_video_data):
        """基本的なアイテムデータ処理テスト"""
        processor = ItemFeatureProcessor()
        
        result = processor.process_item_data(sample_video_data)
        
        assert isinstance(result, dict)
        assert 'item_features' in result
        assert 'feature_names' in result
        assert len(result['item_features']) == len(sample_video_data)
    
    def test_extract_content_features(self, sample_video_data):
        """コンテンツ特徴量抽出テスト"""
        processor = ItemFeatureProcessor()
        
        features = processor.extract_content_features(sample_video_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_video_data)
        assert features.shape[1] > 0  # Has feature dimensions
    
    def test_extract_popularity_features(self, sample_video_data):
        """人気度特徴量抽出テスト"""
        processor = ItemFeatureProcessor()
        
        features = processor.extract_popularity_features(sample_video_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_video_data)
    
    def test_process_with_missing_fields(self):
        """必須フィールド欠損データテスト"""
        processor = ItemFeatureProcessor()
        
        # Missing required fields
        incomplete_data = pd.DataFrame({
            'title': ['Video 1'],
            # Missing other required fields
        })
        
        with pytest.raises(KeyError):
            processor.process_item_data(incomplete_data)


@pytest.mark.unit
class TestEmbeddingManager:
    """EmbeddingManager ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        manager = EmbeddingManager()
        assert manager is not None
        assert manager.dimension == 768
    
    def test_initialize_embeddings(self, sample_user_data, sample_video_data):
        """埋め込み初期化テスト"""
        manager = EmbeddingManager()
        
        user_df = pd.DataFrame([sample_user_data])
        
        manager.initialize_embeddings(user_df, sample_video_data)
        
        assert manager.user_embeddings is not None
        assert manager.item_embeddings is not None
        assert manager.user_embeddings.shape[1] == 768
        assert manager.item_embeddings.shape[1] == 768
    
    def test_compute_similarity(self, sample_embeddings):
        """類似度計算テスト"""
        manager = EmbeddingManager()
        
        user_emb = np.array(sample_embeddings['user_embeddings'][0])
        item_emb = np.array(sample_embeddings['video_embeddings'][0])
        
        similarity = manager.compute_similarity(user_emb, item_emb)
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_get_similar_items(self, sample_embeddings):
        """類似アイテム取得テスト"""
        manager = EmbeddingManager()
        
        # Set up embeddings
        manager.user_embeddings = np.array(sample_embeddings['user_embeddings'])
        manager.item_embeddings = np.array(sample_embeddings['video_embeddings'])
        manager.user_ids = [f'user_{i}' for i in range(10)]
        manager.item_ids = [f'video_{i}' for i in range(20)]
        
        user_id = 'user_0'
        similar_items = manager.get_similar_items(user_id, top_k=5)
        
        assert isinstance(similar_items, list)
        assert len(similar_items) == 5
        for item in similar_items:
            assert 'item_id' in item
            assert 'similarity' in item
    
    def test_update_user_embedding(self):
        """ユーザー埋め込み更新テスト"""
        manager = EmbeddingManager()
        
        user_id = 'test_user'
        new_embedding = np.random.randn(768)
        
        manager.update_user_embedding(user_id, new_embedding)
        
        # Verify update
        assert user_id in manager._user_embeddings_cache
        np.testing.assert_array_equal(
            manager._user_embeddings_cache[user_id], 
            new_embedding
        )
    
    def test_batch_similarity_computation(self, sample_embeddings):
        """バッチ類似度計算テスト"""
        manager = EmbeddingManager()
        
        user_embs = np.array(sample_embeddings['user_embeddings'][:5])
        item_embs = np.array(sample_embeddings['video_embeddings'][:10])
        
        similarities = manager.batch_compute_similarity(user_embs, item_embs)
        
        assert similarities.shape == (5, 10)
        assert np.all(similarities >= -1.0) and np.all(similarities <= 1.0)


@pytest.mark.unit
class TestTwoTowerTrainer:
    """TwoTowerTrainer ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        config = {
            'user_tower_dim': 768,
            'item_tower_dim': 768,
            'hidden_dims': [512, 256],
            'learning_rate': 0.001
        }
        
        trainer = TwoTowerTrainer(config)
        assert trainer is not None
        assert trainer.config == config
    
    def test_build_model(self):
        """モデル構築テスト"""
        config = {
            'user_tower_dim': 768,
            'item_tower_dim': 768,
            'hidden_dims': [512, 256],
            'embedding_dim': 768
        }
        
        trainer = TwoTowerTrainer(config)
        model = trainer.build_model()
        
        assert model is not None
        # Check model architecture
        assert hasattr(model, 'user_tower')
        assert hasattr(model, 'item_tower')
    
    @pytest.mark.slow
    def test_training_step(self, sample_user_data, sample_video_data):
        """トレーニングステップテスト"""
        config = {
            'user_tower_dim': 768,
            'item_tower_dim': 768, 
            'hidden_dims': [512, 256],
            'embedding_dim': 768,
            'batch_size': 32
        }
        
        trainer = TwoTowerTrainer(config)
        model = trainer.build_model()
        
        # Create training batch
        user_features = np.random.randn(32, 768)
        item_features = np.random.randn(32, 768)
        labels = np.random.randint(0, 2, 32)
        
        loss = trainer.training_step(model, user_features, item_features, labels)
        
        assert isinstance(loss, float)
        assert loss >= 0.0
    
    def test_evaluate_model(self):
        """モデル評価テスト"""
        config = {
            'user_tower_dim': 768,
            'item_tower_dim': 768,
            'hidden_dims': [512, 256],
            'embedding_dim': 768
        }
        
        trainer = TwoTowerTrainer(config)
        
        # Mock evaluation data
        eval_data = {
            'user_features': np.random.randn(100, 768),
            'item_features': np.random.randn(100, 768),
            'labels': np.random.randint(0, 2, 100)
        }
        
        with patch.object(trainer, '_compute_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'auc': 0.91
            }
            
            metrics = trainer.evaluate_model(Mock(), eval_data)
            
            assert isinstance(metrics, dict)
            assert 'accuracy' in metrics
            assert 'auc' in metrics
            assert 0.0 <= metrics['accuracy'] <= 1.0


@pytest.mark.unit
class TestMLUtilities:
    """ML ユーティリティ関数テスト"""
    
    def test_normalize_embeddings(self):
        """埋め込み正規化テスト"""
        from backend.ml.preprocessing.utils.feature_scaler import normalize_embeddings
        
        embeddings = np.random.randn(10, 768)
        normalized = normalize_embeddings(embeddings)
        
        # Check L2 normalization
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=6)
    
    def test_cosine_similarity_batch(self):
        """バッチコサイン類似度計算テスト"""
        from backend.ml.preprocessing.utils.similarity import cosine_similarity_batch
        
        a = np.random.randn(5, 768)
        b = np.random.randn(10, 768)
        
        similarities = cosine_similarity_batch(a, b)
        
        assert similarities.shape == (5, 10)
        assert np.all(similarities >= -1.0) and np.all(similarities <= 1.0)
    
    def test_feature_scaling(self):
        """特徴量スケーリングテスト"""
        from backend.ml.preprocessing.utils.feature_scaler import StandardScaler
        
        scaler = StandardScaler()
        features = np.random.randn(100, 10)
        
        scaled_features = scaler.fit_transform(features)
        
        # Check standardization
        assert np.allclose(scaled_features.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled_features.std(axis=0), 1, atol=1e-10)
    
    def test_train_test_split_stratified(self):
        """層化サンプリング分割テスト"""
        from backend.ml.preprocessing.utils.data_splitter import train_test_split_stratified
        
        X = np.random.randn(1000, 10)
        y = np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=0.2, random_state=42
        )
        
        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 200
        assert len(y_train) == 800
        assert len(y_test) == 200
        
        # Check stratification
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05  # Within 5%


@pytest.mark.integration
class TestMLIntegration:
    """ML コンポーネント統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_preprocessing_pipeline(self, sample_user_data, sample_video_data):
        """前処理パイプライン統合テスト"""
        user_processor = UserFeatureProcessor()
        item_processor = ItemFeatureProcessor()
        embedding_manager = EmbeddingManager()
        
        # Process features
        user_df = pd.DataFrame([sample_user_data])
        user_features = user_processor.process_user_data(user_df)
        item_features = item_processor.process_item_data(sample_video_data)
        
        # Initialize embeddings
        embedding_manager.initialize_embeddings(user_df, sample_video_data)
        
        assert user_features is not None
        assert item_features is not None
        assert embedding_manager.user_embeddings is not None
        assert embedding_manager.item_embeddings is not None
    
    @pytest.mark.slow
    def test_training_pipeline_integration(self, sample_user_data, sample_video_data):
        """トレーニングパイプライン統合テスト"""
        # Setup components
        user_processor = UserFeatureProcessor()
        item_processor = ItemFeatureProcessor()
        trainer_config = {
            'user_tower_dim': 768,
            'item_tower_dim': 768,
            'hidden_dims': [256, 128],
            'embedding_dim': 768,
            'batch_size': 16,
            'learning_rate': 0.001
        }
        trainer = TwoTowerTrainer(trainer_config)
        
        # Process data
        user_df = pd.DataFrame([sample_user_data])
        user_features = user_processor.process_user_data(user_df)
        item_features = item_processor.process_item_data(sample_video_data)
        
        # Build and train model (minimal training for test)
        model = trainer.build_model()
        
        # Create synthetic training data
        batch_size = 16
        user_batch = np.random.randn(batch_size, 768)
        item_batch = np.random.randn(batch_size, 768)
        labels = np.random.randint(0, 2, batch_size)
        
        # Single training step
        loss = trainer.training_step(model, user_batch, item_batch, labels)
        
        assert isinstance(loss, float)
        assert loss >= 0.0