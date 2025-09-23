"""
ML Pipeline Integration Tests

機械学習パイプライン統合テスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# テスト対象のモジュールをインポート
try:
    from backend.ml.training.unified_trainer import UnifiedTrainer
    from backend.ml.preprocessing.unified_data_processor import UnifiedDataProcessor
    from backend.ml.models.two_tower_model import TwoTowerModel
    from backend.ml.preprocessing.embeddings.embedding_manager import EmbeddingManager
except ImportError:
    pytest.skip("ML modules not found", allow_module_level=True)


class TestMLPipelineIntegration:
    """MLパイプライン統合テスト"""
    
    @pytest.fixture
    def sample_training_data(self):
        """トレーニング用サンプルデータ"""
        return {
            'videos': pd.DataFrame({
                'video_id': [f'video_{i}' for i in range(100)],
                'title': [f'Video Title {i}' for i in range(100)],
                'description': [f'Description {i}' for i in range(100)],
                'genre': np.random.choice(['action', 'comedy', 'drama'], 100),
                'maker': np.random.choice(['maker_a', 'maker_b', 'maker_c'], 100),
                'price': np.random.randint(1000, 5000, 100)
            }),
            'users': pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(50)],
                'preferences': [f'pref_{i}' for i in range(50)]
            }),
            'interactions': pd.DataFrame({
                'user_id': np.random.choice([f'user_{i}' for i in range(50)], 200),
                'video_id': np.random.choice([f'video_{i}' for i in range(100)], 200),
                'rating': np.random.choice([1, 2, 3, 4, 5], 200),
                'liked': np.random.choice([0, 1], 200)
            })
        }
    
    @pytest.fixture
    def ml_config(self):
        """ML設定"""
        return {
            'embedding_dim': 768,
            'batch_size': 16,
            'epochs': 2,
            'learning_rate': 0.001,
            'model_type': 'two_tower',
            'validation_split': 0.2
        }
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_data_preprocessing_pipeline(self, sample_training_data, ml_config):
        """データ前処理パイプライン統合テスト"""
        processor = UnifiedDataProcessor(config=ml_config)
        
        # データ前処理実行
        processed_data = processor.process_training_data(sample_training_data)
        
        # 結果検証
        assert 'user_features' in processed_data
        assert 'item_features' in processed_data
        assert 'interactions' in processed_data
        
        # 埋め込み次元の確認
        user_features = processed_data['user_features']
        item_features = processed_data['item_features']
        
        assert user_features.shape[1] == ml_config['embedding_dim']
        assert item_features.shape[1] == ml_config['embedding_dim']
    
    @pytest.mark.integration
    @pytest.mark.ml
    @pytest.mark.slow
    def test_full_training_pipeline(self, sample_training_data, ml_config):
        """完全トレーニングパイプライン統合テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            model_dir.mkdir()
            
            trainer = UnifiedTrainer(config=ml_config)
            
            # トレーニング実行
            training_result = trainer.train(
                data=sample_training_data,
                model_save_path=model_dir
            )
            
            # 結果検証
            assert training_result['success'] is True
            assert 'model_path' in training_result
            assert 'metrics' in training_result
            
            # モデルファイルが作成されていることを確認
            model_path = Path(training_result['model_path'])
            assert model_path.exists()
            
            # メトリクス検証
            metrics = training_result['metrics']
            assert 'loss' in metrics
            assert 'accuracy' in metrics
            assert metrics['loss'] > 0
            assert 0 <= metrics['accuracy'] <= 1
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_model_inference_pipeline(self, sample_training_data, ml_config):
        """モデル推論パイプライン統合テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            model_dir.mkdir()
            
            # モデル訓練
            trainer = UnifiedTrainer(config=ml_config)
            training_result = trainer.train(
                data=sample_training_data,
                model_save_path=model_dir
            )
            
            # モデル読み込み
            model = TwoTowerModel.load(training_result['model_path'])
            
            # 推論実行
            user_embedding = np.random.random(ml_config['embedding_dim'])
            item_embeddings = np.random.random((10, ml_config['embedding_dim']))
            
            scores = model.predict_user_item_scores(user_embedding, item_embeddings)
            
            # 結果検証
            assert len(scores) == 10
            assert all(isinstance(score, float) for score in scores)
            assert all(0 <= score <= 1 for score in scores)
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_embedding_generation_pipeline(self, sample_training_data):
        """埋め込み生成パイプライン統合テスト"""
        embedding_manager = EmbeddingManager(embedding_dim=768)
        
        # テキストデータから埋め込み生成
        video_texts = sample_training_data['videos']['title'].tolist()
        
        with patch.object(embedding_manager.model_manager, 'get_model') as mock_model:
            # モデルの戻り値をモック
            mock_embeddings = np.random.random((len(video_texts), 768)).astype(np.float32)
            mock_model.return_value.encode.return_value = mock_embeddings
            
            embeddings = embedding_manager.encode_texts(video_texts)
            
            # 結果検証
            assert embeddings.shape == (len(video_texts), 768)
            assert embeddings.dtype == np.float32
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_similarity_calculation_pipeline(self, sample_training_data):
        """類似度計算パイプライン統合テスト"""
        embedding_manager = EmbeddingManager(embedding_dim=768)
        
        # サンプル埋め込みデータ
        user_embeddings = np.random.random((10, 768)).astype(np.float32)
        item_embeddings = np.random.random((20, 768)).astype(np.float32)
        
        # バッチ類似度計算
        similarities = embedding_manager.calculate_similarity_batch(
            user_embeddings, item_embeddings, method='cosine'
        )
        
        # 結果検証
        assert similarities.shape == (10, 20)
        assert np.all(similarities >= -1.0)
        assert np.all(similarities <= 1.0)
        
        # トップK類似アイテム検索
        query_embedding = user_embeddings[0]
        top_items = embedding_manager.find_similar_items(
            query_embedding, item_embeddings, top_k=5
        )
        
        assert len(top_items) == 5
        assert all('index' in item and 'similarity' in item for item in top_items)
        
        # 類似度の降順ソートを確認
        similarities_list = [item['similarity'] for item in top_items]
        assert similarities_list == sorted(similarities_list, reverse=True)
    
    @pytest.mark.integration
    @pytest.mark.ml
    @pytest.mark.performance
    def test_pipeline_performance(self, sample_training_data, ml_config, performance_timer):
        """パイプラインパフォーマンステスト"""
        # データ前処理パフォーマンス
        processor = UnifiedDataProcessor(config=ml_config)
        
        performance_timer.start()
        processed_data = processor.process_training_data(sample_training_data)
        performance_timer.stop()
        
        preprocessing_time = performance_timer.duration
        assert preprocessing_time < 10.0  # 10秒以内
        
        # 埋め込み生成パフォーマンス
        embedding_manager = EmbeddingManager(embedding_dim=768)
        video_texts = sample_training_data['videos']['title'].tolist()
        
        with patch.object(embedding_manager.model_manager, 'get_model') as mock_model:
            mock_embeddings = np.random.random((len(video_texts), 768)).astype(np.float32)
            mock_model.return_value.encode.return_value = mock_embeddings
            
            performance_timer.start()
            embeddings = embedding_manager.encode_texts(video_texts)
            performance_timer.stop()
            
            embedding_time = performance_timer.duration
            assert embedding_time < 5.0  # 5秒以内
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_model_versioning_pipeline(self, sample_training_data, ml_config):
        """モデルバージョニングパイプライン統合テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            model_dir.mkdir()
            
            trainer = UnifiedTrainer(config=ml_config)
            
            # 初回トレーニング
            result_v1 = trainer.train(
                data=sample_training_data,
                model_save_path=model_dir,
                version="v1.0"
            )
            
            # 2回目トレーニング（別バージョン）
            result_v2 = trainer.train(
                data=sample_training_data,
                model_save_path=model_dir,
                version="v2.0"
            )
            
            # 両バージョンのモデルが保存されていることを確認
            assert Path(result_v1['model_path']).exists()
            assert Path(result_v2['model_path']).exists()
            assert result_v1['model_path'] != result_v2['model_path']
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_error_handling_pipeline(self, ml_config):
        """エラーハンドリングパイプライン統合テスト"""
        trainer = UnifiedTrainer(config=ml_config)
        
        # 無効なデータでトレーニング
        invalid_data = {
            'videos': pd.DataFrame(),  # 空のデータフレーム
            'users': pd.DataFrame(),
            'interactions': pd.DataFrame()
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            model_dir.mkdir()
            
            result = trainer.train(
                data=invalid_data,
                model_save_path=model_dir
            )
            
            # エラーが適切に処理されていることを確認
            assert result['success'] is False
            assert 'error' in result
    
    @pytest.mark.integration
    @pytest.mark.ml
    def test_data_validation_pipeline(self, sample_training_data, ml_config):
        """データ検証パイプライン統合テスト"""
        processor = UnifiedDataProcessor(config=ml_config)
        
        # データ検証実行
        validation_result = processor.validate_training_data(sample_training_data)
        
        # 検証結果確認
        assert validation_result['is_valid'] is True
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
        assert len(validation_result['errors']) == 0
        
        # 不正なデータでの検証
        invalid_data = sample_training_data.copy()
        invalid_data['videos'] = invalid_data['videos'].drop(columns=['title'])
        
        invalid_validation_result = processor.validate_training_data(invalid_data)
        
        assert invalid_validation_result['is_valid'] is False
        assert len(invalid_validation_result['errors']) > 0