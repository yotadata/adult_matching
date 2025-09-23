"""
Tests for EmbeddingManager

ML埋め込み管理システムの単体テスト
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# テスト対象のモジュールをインポート
try:
    from backend.ml.preprocessing.embeddings.embedding_manager import EmbeddingManager
    from backend.ml.preprocessing.embeddings.model_manager import ModelManager
except ImportError:
    # テスト実行時にモジュールが見つからない場合はスキップ
    pytest.skip("ML modules not found", allow_module_level=True)


class TestEmbeddingManager:
    """EmbeddingManagerの単体テスト"""
    
    @pytest.fixture
    def mock_model_manager(self):
        """ModelManagerのモック"""
        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_model.return_value = Mock()
        mock_manager.get_tokenizer.return_value = Mock()
        return mock_manager
    
    @pytest.fixture
    def embedding_manager(self, mock_model_manager):
        """テスト用EmbeddingManagerインスタンス"""
        with patch('backend.ml.preprocessing.embeddings.embedding_manager.ModelManager', return_value=mock_model_manager):
            manager = EmbeddingManager(embedding_dim=768)
            return manager
    
    @pytest.fixture
    def sample_texts(self):
        """テスト用テキストデータ"""
        return [
            "これはテスト用の動画タイトルです",
            "別のテスト動画の説明文です",
            "アクション映画のサンプルタイトル"
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """テスト用埋め込みデータ"""
        np.random.seed(42)
        return np.random.random((3, 768)).astype(np.float32)
    
    @pytest.mark.unit
    def test_initialization(self, embedding_manager):
        """初期化テスト"""
        assert embedding_manager.embedding_dim == 768
        assert embedding_manager.model_manager is not None
        assert embedding_manager.cache is not None
    
    @pytest.mark.unit
    def test_encode_texts_single(self, embedding_manager, sample_texts):
        """単一テキストの埋め込み生成テスト"""
        # モデルの戻り値を設定
        mock_embedding = np.random.random(768).astype(np.float32)
        embedding_manager.model_manager.get_model.return_value.encode.return_value = mock_embedding
        
        result = embedding_manager.encode_texts(sample_texts[0])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)
        assert result.dtype == np.float32
    
    @pytest.mark.unit
    def test_encode_texts_batch(self, embedding_manager, sample_texts, sample_embeddings):
        """バッチテキストの埋め込み生成テスト"""
        # モデルの戻り値を設定
        embedding_manager.model_manager.get_model.return_value.encode.return_value = sample_embeddings
        
        result = embedding_manager.encode_texts(sample_texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)
        assert result.dtype == np.float32
    
    @pytest.mark.unit
    def test_encode_texts_empty_input(self, embedding_manager):
        """空入力のハンドリングテスト"""
        result = embedding_manager.encode_texts([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 768)
    
    @pytest.mark.unit
    def test_encode_texts_with_cache(self, embedding_manager, sample_texts):
        """キャッシュ機能テスト"""
        mock_embedding = np.random.random(768).astype(np.float32)
        embedding_manager.model_manager.get_model.return_value.encode.return_value = mock_embedding
        
        # 最初の呼び出し
        result1 = embedding_manager.encode_texts(sample_texts[0])
        
        # 2回目の呼び出し（キャッシュから）
        result2 = embedding_manager.encode_texts(sample_texts[0])
        
        # 結果が同じであることを確認
        np.testing.assert_array_equal(result1, result2)
        
        # モデルが1回だけ呼ばれたことを確認
        embedding_manager.model_manager.get_model.return_value.encode.assert_called_once()
    
    @pytest.mark.unit
    def test_calculate_similarity_cosine(self, embedding_manager, sample_embeddings):
        """コサイン類似度計算テスト"""
        embedding1 = sample_embeddings[0]
        embedding2 = sample_embeddings[1]
        
        similarity = embedding_manager.calculate_similarity(embedding1, embedding2, method='cosine')
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    @pytest.mark.unit
    def test_calculate_similarity_euclidean(self, embedding_manager, sample_embeddings):
        """ユークリッド距離計算テスト"""
        embedding1 = sample_embeddings[0]
        embedding2 = sample_embeddings[1]
        
        distance = embedding_manager.calculate_similarity(embedding1, embedding2, method='euclidean')
        
        assert isinstance(distance, float)
        assert distance >= 0.0
    
    @pytest.mark.unit
    def test_calculate_similarity_batch(self, embedding_manager, sample_embeddings):
        """バッチ類似度計算テスト"""
        similarities = embedding_manager.calculate_similarity_batch(
            sample_embeddings, sample_embeddings, method='cosine'
        )
        
        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (3, 3)
        # 対角成分は1に近い値
        np.testing.assert_array_almost_equal(np.diag(similarities), np.ones(3), decimal=5)
    
    @pytest.mark.unit
    def test_find_similar_items(self, embedding_manager, sample_embeddings):
        """類似アイテム検索テスト"""
        query_embedding = sample_embeddings[0]
        candidate_embeddings = sample_embeddings[1:]
        
        results = embedding_manager.find_similar_items(
            query_embedding, candidate_embeddings, top_k=1
        )
        
        assert len(results) == 1
        assert 'index' in results[0]
        assert 'similarity' in results[0]
        assert isinstance(results[0]['index'], int)
        assert isinstance(results[0]['similarity'], float)
    
    @pytest.mark.unit
    def test_save_load_embeddings(self, embedding_manager, sample_embeddings):
        """埋め込み保存・読み込みテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_embeddings.npy"
            
            # 保存
            embedding_manager.save_embeddings(sample_embeddings, file_path)
            assert file_path.exists()
            
            # 読み込み
            loaded_embeddings = embedding_manager.load_embeddings(file_path)
            
            np.testing.assert_array_equal(sample_embeddings, loaded_embeddings)
    
    @pytest.mark.unit
    def test_dimension_validation(self, mock_model_manager):
        """次元数検証テスト"""
        # 無効な次元数でインスタンス作成
        with pytest.raises(ValueError):
            EmbeddingManager(embedding_dim=-1)
        
        with pytest.raises(ValueError):
            EmbeddingManager(embedding_dim=0)
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_performance_large_batch(self, embedding_manager, performance_timer):
        """大量バッチ処理のパフォーマンステスト"""
        # 大量テキストデータ
        large_texts = [f"テストテキスト{i}" for i in range(1000)]
        mock_embeddings = np.random.random((1000, 768)).astype(np.float32)
        
        embedding_manager.model_manager.get_model.return_value.encode.return_value = mock_embeddings
        
        performance_timer.start()
        result = embedding_manager.encode_texts(large_texts)
        performance_timer.stop()
        
        assert result.shape == (1000, 768)
        assert performance_timer.duration < 5.0  # 5秒以内
    
    @pytest.mark.unit
    def test_error_handling_invalid_input(self, embedding_manager):
        """無効入力のエラーハンドリングテスト"""
        # Noneの処理
        with pytest.raises(ValueError):
            embedding_manager.encode_texts(None)
        
        # 数値入力の処理
        with pytest.raises(TypeError):
            embedding_manager.encode_texts(123)
    
    @pytest.mark.unit
    def test_cache_size_limit(self, embedding_manager):
        """キャッシュサイズ制限テスト"""
        # キャッシュサイズを小さく設定
        embedding_manager.cache.maxsize = 2
        
        mock_embedding = np.random.random(768).astype(np.float32)
        embedding_manager.model_manager.get_model.return_value.encode.return_value = mock_embedding
        
        # 複数のテキストをエンコード
        texts = ["text1", "text2", "text3"]
        for text in texts:
            embedding_manager.encode_texts(text)
        
        # キャッシュサイズが制限内であることを確認
        assert len(embedding_manager.cache) <= 2
    
    @pytest.mark.unit
    def test_clear_cache(self, embedding_manager, sample_texts):
        """キャッシュクリア機能テスト"""
        mock_embedding = np.random.random(768).astype(np.float32)
        embedding_manager.model_manager.get_model.return_value.encode.return_value = mock_embedding
        
        # キャッシュにデータを追加
        embedding_manager.encode_texts(sample_texts[0])
        assert len(embedding_manager.cache) > 0
        
        # キャッシュクリア
        embedding_manager.clear_cache()
        assert len(embedding_manager.cache) == 0
    
    @pytest.mark.unit
    def test_get_cache_stats(self, embedding_manager, sample_texts):
        """キャッシュ統計情報テスト"""
        mock_embedding = np.random.random(768).astype(np.float32)
        embedding_manager.model_manager.get_model.return_value.encode.return_value = mock_embedding
        
        # キャッシュにデータを追加
        embedding_manager.encode_texts(sample_texts[0])
        
        stats = embedding_manager.get_cache_stats()
        
        assert 'size' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert isinstance(stats['size'], int)
        assert isinstance(stats['hits'], int)
        assert isinstance(stats['misses'], int)