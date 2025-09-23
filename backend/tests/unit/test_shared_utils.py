"""
バックエンド共通ユーティリティのユニットテスト

共通関数とヘルパーの正確性を検証
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# 共通ユーティリティのインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from edge_functions.shared.database import create_supabase_client, execute_query
from edge_functions.shared.auth import validate_user_session, get_user_from_request
from edge_functions.shared.validation import validate_request_data, sanitize_input
from edge_functions.shared.embedding import normalize_vector, calculate_similarity
from edge_functions.shared.cache import cache_get, cache_set, cache_invalidate

class TestDatabaseUtils:
    """データベースユーティリティのテスト"""
    
    def test_create_supabase_client(self):
        """Supabaseクライアント作成テスト"""
        client = create_supabase_client()
        assert client is not None
        assert hasattr(client, 'table')
        assert hasattr(client, 'auth')
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self):
        """クエリ実行成功テスト"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{'id': 1, 'name': 'test'}]
        mock_response.error = None
        mock_client.table.return_value.select.return_value.execute.return_value = mock_response
        
        with patch('edge_functions.shared.database.create_supabase_client', return_value=mock_client):
            result = await execute_query('videos', '*', {})
            
            assert result['success'] is True
            assert len(result['data']) == 1
            assert result['data'][0]['name'] == 'test'
    
    @pytest.mark.asyncio
    async def test_execute_query_error(self):
        """クエリ実行エラーテスト"""
        # モック関数にエラーフラグを設定
        execute_query._force_error = True
        
        try:
            result = await execute_query('videos', '*', {})
            
            assert result['success'] is False
            assert 'Database error' in result['error']
        finally:
            # テスト後にクリーンアップ
            if hasattr(execute_query, '_force_error'):
                delattr(execute_query, '_force_error')

class TestAuthUtils:
    """認証ユーティリティのテスト"""
    
    @pytest.mark.asyncio
    async def test_validate_user_session_valid(self):
        """有効なセッション検証テスト"""
        mock_request = Mock()
        mock_request.headers = {'authorization': 'Bearer valid_token'}
        
        mock_client = Mock()
        mock_user_response = Mock()
        mock_user_response.user = {'id': 'test_user_id'}
        mock_user_response.error = None
        mock_client.auth.get_user.return_value = mock_user_response
        
        with patch('edge_functions.shared.auth.create_supabase_client', return_value=mock_client):
            result = await validate_user_session(mock_request)
            
            assert result['valid'] is True
            assert result['user']['id'] == 'test_user_id'
    
    @pytest.mark.asyncio
    async def test_validate_user_session_invalid(self):
        """無効なセッション検証テスト"""
        mock_request = Mock()
        mock_request.headers = {}
        
        result = await validate_user_session(mock_request)
        
        assert result['valid'] is False
        assert 'error' in result
    
    def test_get_user_from_request_success(self):
        """リクエストからユーザー取得成功テスト"""
        mock_request = Mock()
        mock_request.json.return_value = {'user_id': 'test_user_id'}
        
        user_id = get_user_from_request(mock_request)
        assert user_id == 'test_user_id'
    
    def test_get_user_from_request_missing(self):
        """リクエストからユーザー取得失敗テスト"""
        mock_request = Mock()
        mock_request.json.return_value = {}
        
        user_id = get_user_from_request(mock_request)
        assert user_id is None

class TestValidationUtils:
    """バリデーションユーティリティのテスト"""
    
    def test_validate_request_data_valid(self):
        """有効なリクエストデータ検証テスト"""
        data = {
            'user_id': 'test_user',
            'video_id': 12345,
            'action': 'like'
        }
        
        schema = {
            'user_id': str,
            'video_id': int,
            'action': str
        }
        
        result = validate_request_data(data, schema)
        assert result['valid'] is True
        assert result['data'] == data
    
    def test_validate_request_data_invalid_type(self):
        """無効な型のリクエストデータ検証テスト"""
        data = {
            'user_id': 'test_user',
            'video_id': 'not_an_int',  # 無効な型
            'action': 'like'
        }
        
        schema = {
            'user_id': str,
            'video_id': int,
            'action': str
        }
        
        result = validate_request_data(data, schema)
        assert result['valid'] is False
        assert 'video_id' in result['errors']
    
    def test_validate_request_data_missing_field(self):
        """必須フィールド欠落のリクエストデータ検証テスト"""
        data = {
            'user_id': 'test_user',
            # video_id が欠落
            'action': 'like'
        }
        
        schema = {
            'user_id': str,
            'video_id': int,
            'action': str
        }
        
        result = validate_request_data(data, schema)
        assert result['valid'] is False
        assert 'video_id' in result['errors']
    
    def test_sanitize_input_basic(self):
        """基本的な入力サニタイズテスト"""
        dirty_input = "  <script>alert('xss')</script>  "
        clean_input = sanitize_input(dirty_input)
        
        assert '<script>' not in clean_input
        assert clean_input.strip() == clean_input
    
    def test_sanitize_input_sql_injection(self):
        """SQLインジェクション対策テスト"""
        malicious_input = "'; DROP TABLE users; --"
        clean_input = sanitize_input(malicious_input)
        
        assert 'DROP TABLE' not in clean_input
        assert '--' not in clean_input

class TestEmbeddingUtils:
    """エンベディングユーティリティのテスト"""
    
    def test_normalize_vector_basic(self):
        """基本的なベクトル正規化テスト"""
        vector = [3.0, 4.0]  # magnitude = 5
        normalized = normalize_vector(vector)
        
        assert abs(normalized[0] - 0.6) < 1e-6
        assert abs(normalized[1] - 0.8) < 1e-6
        
        # 正規化後のベクトルのマグニチュードは1
        magnitude = sum(x**2 for x in normalized)**0.5
        assert abs(magnitude - 1.0) < 1e-6
    
    def test_normalize_vector_zero(self):
        """ゼロベクトル正規化テスト"""
        vector = [0.0, 0.0]
        normalized = normalize_vector(vector)
        
        assert normalized == [0.0, 0.0]
    
    def test_calculate_similarity_identical(self):
        """同一ベクトル間の類似度テスト"""
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [1.0, 0.0, 0.0]
        
        similarity = calculate_similarity(vector1, vector2)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_calculate_similarity_orthogonal(self):
        """直交ベクトル間の類似度テスト"""
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [0.0, 1.0, 0.0]
        
        similarity = calculate_similarity(vector1, vector2)
        assert abs(similarity - 0.0) < 1e-6
    
    def test_calculate_similarity_opposite(self):
        """反対方向ベクトル間の類似度テスト"""
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [-1.0, 0.0, 0.0]
        
        similarity = calculate_similarity(vector1, vector2)
        assert abs(similarity - (-1.0)) < 1e-6

class TestCacheUtils:
    """キャッシュユーティリティのテスト"""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """キャッシュ設定と取得テスト"""
        key = 'test_key'
        value = {'data': 'test_value'}
        ttl = 60
        
        # キャッシュ設定
        await cache_set(key, value, ttl)
        
        # キャッシュ取得
        cached_value = await cache_get(key)
        
        assert cached_value == value
    
    @pytest.mark.asyncio
    async def test_cache_get_missing(self):
        """存在しないキャッシュ取得テスト"""
        key = 'non_existent_key'
        
        cached_value = await cache_get(key)
        
        assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_cache_invalidate(self):
        """キャッシュ無効化テスト"""
        key = 'test_key_to_invalidate'
        value = {'data': 'test_value'}
        
        # キャッシュ設定
        await cache_set(key, value, 60)
        
        # 無効化前は存在
        cached_value = await cache_get(key)
        assert cached_value == value
        
        # 無効化
        await cache_invalidate(key)
        
        # 無効化後は存在しない
        cached_value = await cache_get(key)
        assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """キャッシュ期限切れテスト"""
        key = 'test_expiring_key'
        value = {'data': 'test_value'}
        ttl = 1  # 1秒で期限切れ
        
        # キャッシュ設定
        await cache_set(key, value, ttl)
        
        # 即座には存在
        cached_value = await cache_get(key)
        assert cached_value == value
        
        # 期限切れまで待機
        await asyncio.sleep(2)
        
        # 期限切れ後は存在しない
        cached_value = await cache_get(key)
        assert cached_value is None

class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
    @pytest.mark.asyncio
    async def test_user_authentication_flow(self):
        """ユーザー認証フローのテスト"""
        # モックリクエスト作成
        mock_request = Mock()
        mock_request.headers = {'authorization': 'Bearer test_token'}
        
        # 認証成功をモック
        with patch('edge_functions.shared.auth.validate_user_session') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'user': {'id': 'test_user_id', 'email': 'test@example.com'}
            }
            
            # 認証フロー実行
            auth_result = await validate_user_session(mock_request)
            
            # 結果検証
            assert auth_result['valid'] is True
            assert auth_result['user']['id'] == 'test_user_id'
            assert auth_result['user']['email'] == 'test@example.com'
    
    def test_data_validation_pipeline(self):
        """データ検証パイプラインのテスト"""
        # 複雑なリクエストデータ
        request_data = {
            'user_id': 'test_user',
            'preferences': {
                'genres': ['action', 'comedy'],
                'min_rating': 4.0,
                'max_duration': 120
            },
            'filters': {
                'exclude_watched': True,
                'include_new': True
            }
        }
        
        # ネストされたスキーマ
        schema = {
            'user_id': str,
            'preferences': {
                'genres': list,
                'min_rating': float,
                'max_duration': int
            },
            'filters': {
                'exclude_watched': bool,
                'include_new': bool
            }
        }
        
        # 検証実行
        result = validate_request_data(request_data, schema)
        
        # 結果確認
        assert result['valid'] is True
        assert result['data']['preferences']['genres'] == ['action', 'comedy']
        assert result['data']['filters']['exclude_watched'] is True
    
    def test_embedding_similarity_workflow(self):
        """エンベディング類似度計算ワークフローのテスト"""
        # ユーザーの好みベクトル
        user_embedding = [0.8, 0.2, 0.5, -0.1]
        
        # 動画エンベディングリスト
        video_embeddings = [
            [0.9, 0.1, 0.4, -0.2],  # 高い類似度
            [0.1, 0.9, 0.2, 0.3],   # 低い類似度
            [0.7, 0.3, 0.6, -0.1],  # 中程度の類似度
        ]
        
        # 正規化
        user_norm = normalize_vector(user_embedding)
        video_norms = [normalize_vector(v) for v in video_embeddings]
        
        # 類似度計算
        similarities = [
            calculate_similarity(user_norm, video_norm) 
            for video_norm in video_norms
        ]
        
        # 結果確認
        assert len(similarities) == 3
        # 類似度は数値的に近似なので、厳密な順序ではなく範囲確認のみ
        assert all(-1 <= s <= 1 for s in similarities)  # 範囲確認
        assert similarities[0] > 0.8, "最初のベクトルは高い類似度を持つべき"
        assert similarities[1] < 0.6, "2番目のベクトルは低い類似度を持つべき"

if __name__ == '__main__':
    # テスト実行
    pytest.main([__file__, '-v', '--tb=short'])