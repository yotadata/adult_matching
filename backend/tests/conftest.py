"""
pytest設定とフィクスチャ

全テストで共通で使用する設定とフィクスチャを定義
"""

import pytest
import os
import sys
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import tempfile
import json

# バックエンドモジュールのパス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# テスト環境変数設定
os.environ['TESTING'] = 'true'
os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test_db'
os.environ['SUPABASE_URL'] = 'https://test.supabase.co'
os.environ['SUPABASE_ANON_KEY'] = 'test_anon_key'
os.environ['DMM_API_ID'] = 'test_api_id'
os.environ['DMM_AFFILIATE_ID'] = 'test_affiliate_id'

# Supabaseクライアントのモック
@pytest.fixture
def mock_supabase_client():
    """Supabaseクライアントのモックフィクスチャ"""
    mock_client = Mock()
    
    # テーブル操作のモック
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    
    # レスポンスのモック
    mock_response = Mock()
    mock_response.data = []
    mock_response.error = None
    mock_table.select.return_value.execute.return_value = mock_response
    mock_table.insert.return_value.execute.return_value = mock_response
    mock_table.update.return_value.execute.return_value = mock_response
    mock_table.delete.return_value.execute.return_value = mock_response
    
    # 認証のモック
    mock_auth = Mock()
    mock_user = Mock()
    mock_user.id = 'test_user_id'
    mock_user.email = 'test@example.com'
    mock_auth.get_user.return_value.user = mock_user
    mock_auth.get_user.return_value.error = None
    mock_client.auth = mock_auth
    
    return mock_client

# データベース接続のモック
@pytest.fixture
def mock_database_connector():
    """データベースコネクターのモックフィクスチャ"""
    mock_connector = Mock()
    
    # 基本的なデータベース操作のモック
    mock_connector.test_connection.return_value = True
    mock_connector.execute_query.return_value = []
    mock_connector.save_scraped_videos.return_value = 0
    mock_connector.get_videos_for_cleaning.return_value = []
    mock_connector.get_videos_for_validation.return_value = []
    mock_connector.get_videos_for_embedding.return_value = []
    mock_connector.get_database_stats.return_value = {
        'video_counts': {'scraped': 0, 'cleaned': 0, 'validated': 0},
        'embedding_count': 0,
        'recent_activity': []
    }
    
    return mock_connector

# テスト用動画データ
@pytest.fixture
def sample_video_data():
    """サンプル動画データフィクスチャ"""
    return [
        {
            'id': 1,
            'external_id': 'test_video_1',
            'source': 'dmm',
            'title': 'テスト動画 1',
            'description': 'これはテスト用の動画説明です。',
            'maker': 'テストメーカー',
            'genre': 'アクション',
            'price': 1000,
            'thumbnail_url': 'https://example.com/thumb1.jpg',
            'preview_video_url': 'https://example.com/preview1.mp4',
            'sample_video_url': 'https://example.com/sample1.mp4',
            'image_urls': ['https://example.com/image1.jpg'],
            'duration_seconds': 3600,
            'data_status': 'scraped'
        },
        {
            'id': 2,
            'external_id': 'test_video_2',
            'source': 'dmm',
            'title': 'テスト動画 2',
            'description': '別のテスト用動画です。',
            'maker': 'テストメーカー2',
            'genre': 'コメディ',
            'price': 1500,
            'thumbnail_url': 'https://example.com/thumb2.jpg',
            'preview_video_url': 'https://example.com/preview2.mp4',
            'sample_video_url': 'https://example.com/sample2.mp4',
            'image_urls': ['https://example.com/image2.jpg'],
            'duration_seconds': 5400,
            'data_status': 'cleaned'
        }
    ]

# テスト用ユーザーデータ
@pytest.fixture
def sample_user_data():
    """サンプルユーザーデータフィクスチャ"""
    return {
        'id': 'test_user_id',
        'email': 'test@example.com',
        'preferences': {
            'genres': ['アクション', 'コメディ'],
            'makers': ['テストメーカー'],
            'price_range': {'min': 500, 'max': 2000}
        },
        'embedding': [0.1, 0.2, 0.3, 0.4, 0.5] * 153 + [0.1, 0.2, 0.3]  # 768次元
    }

# テスト用エンベディングデータ
@pytest.fixture
def sample_embeddings():
    """サンプルエンベディングデータフィクスチャ"""
    import numpy as np
    np.random.seed(42)  # 再現性のため
    
    return {
        'user_embeddings': np.random.rand(10, 768).tolist(),  # 10ユーザー
        'video_embeddings': np.random.rand(20, 768).tolist(),  # 20動画
        'similarity_matrix': np.random.rand(10, 20).tolist()   # 類似度行列
    }

# 一時ファイルディレクトリ
@pytest.fixture
def temp_dir():
    """一時ディレクトリフィクスチャ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

# 設定ファイルのモック
@pytest.fixture
def mock_config():
    """設定オブジェクトのモックフィクスチャ"""
    config = Mock()
    
    # スクレイピング設定
    config.scraping.request_delay_ms = 1000
    config.scraping.max_concurrent_requests = 5
    config.scraping.dmm_api_id = 'test_api_id'
    config.scraping.dmm_affiliate_id = 'test_affiliate_id'
    config.scraping.dmm_base_url = 'https://api.dmm.com/affiliate/v3/'
    
    # クリーニング設定
    config.cleaning.min_title_length = 10
    config.cleaning.max_title_length = 200
    config.cleaning.duplicate_threshold = 0.95
    config.cleaning.required_fields = ['title', 'description', 'maker', 'genre', 'price']
    
    # エンベディング設定
    config.embedding.batch_size = 100
    config.embedding.vector_dimension = 768
    config.embedding.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # データベース設定
    config.database.url = 'postgresql://test:test@localhost:5432/test_db'
    config.database.pool_size = 20
    config.database.max_connections = 100
    config.database.enable_query_logging = False
    
    # 監視設定
    config.monitoring.log_level = 'INFO'
    config.monitoring.enable_performance_logging = True
    
    return config

# 非同期テスト用のイベントループ
@pytest.fixture(scope='session')
def event_loop():
    """セッション全体で共有するイベントループ"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# HTTPリクエストのモック
@pytest.fixture
def mock_http_request():
    """HTTPリクエストのモックフィクスチャ"""
    def create_request(method='GET', url='/', headers=None, json_data=None):
        request = Mock()
        request.method = method
        request.url = url
        request.headers = headers or {}
        
        if json_data:
            request.json.return_value = json_data
        else:
            request.json.return_value = {}
        
        return request
    
    return create_request

# Edge Functionsのモック環境
@pytest.fixture
def mock_edge_function_env():
    """Edge Functions環境のモックフィクスチャ"""
    env_vars = {
        'SUPABASE_URL': 'https://test.supabase.co',
        'SUPABASE_ANON_KEY': 'test_anon_key',
        'SUPABASE_SERVICE_ROLE_KEY': 'test_service_role_key'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

# パフォーマンステスト用のメトリクス
@pytest.fixture
def performance_metrics():
    """パフォーマンステスト用メトリクスフィクスチャ"""
    return {
        'max_response_time': 5.0,      # 最大レスポンス時間（秒）
        'max_memory_usage': 500,       # 最大メモリ使用量（MB）
        'min_throughput': 10,          # 最小スループット（req/sec）
        'max_cpu_usage': 80.0,         # 最大CPU使用率（%）
        'max_error_rate': 0.01         # 最大エラー率（1%）
    }

# テストデータのクリーンアップ
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """テスト後の自動クリーンアップ"""
    yield
    
    # テストで作成された一時ファイルの削除
    test_files = [
        '/tmp/test_cache.json',
        '/tmp/test_embeddings.npy',
        '/tmp/test_model.pkl'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)

# ログレベル設定
@pytest.fixture(autouse=True)
def configure_logging():
    """テスト用ログ設定"""
    import logging
    
    # テスト中はログレベルをWARNING以上に設定
    logging.getLogger().setLevel(logging.WARNING)
    yield
    # テスト後はリセット
    logging.getLogger().setLevel(logging.INFO)

# マーカー設定
def pytest_configure(config):
    """pytest設定"""
    config.addinivalue_line(
        "markers", "integration: 統合テスト用マーカー"
    )
    config.addinivalue_line(
        "markers", "performance: パフォーマンステスト用マーカー"
    )
    config.addinivalue_line(
        "markers", "slow: 実行時間が長いテスト用マーカー"
    )
    config.addinivalue_line(
        "markers", "unit: ユニットテスト用マーカー"
    )

# テスト実行前の共通設定
def pytest_sessionstart(session):
    """テストセッション開始時の処理"""
    print("\n=== バックエンドテストスイート開始 ===")
    print(f"Python バージョン: {sys.version}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    print("=" * 50)

# テスト実行後の共通処理
def pytest_sessionfinish(session, exitstatus):
    """テストセッション終了時の処理"""
    print("\n=== バックエンドテストスイート完了 ===")
    print(f"終了ステータス: {exitstatus}")
    if exitstatus == 0:
        print("✓ すべてのテストが成功しました")
    else:
        print("✗ 一部のテストが失敗しました")
    print("=" * 50)

# スキップ条件
def pytest_runtest_setup(item):
    """個別テスト実行前の処理"""
    # 統合テストのスキップ条件
    if item.get_closest_marker("integration"):
        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip("統合テストはRUN_INTEGRATION_TESTS=1で有効化してください")
    
    # パフォーマンステストのスキップ条件
    if item.get_closest_marker("performance"):
        if not os.getenv("RUN_PERFORMANCE_TESTS"):
            pytest.skip("パフォーマンステストはRUN_PERFORMANCE_TESTS=1で有効化してください")
    
    # 遅いテストのスキップ条件
    if item.get_closest_marker("slow"):
        if os.getenv("SKIP_SLOW_TESTS"):
            pytest.skip("遅いテストはSKIP_SLOW_TESTS=1でスキップされます")