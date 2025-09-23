"""
データ処理統合テスト用 pytest fixtures
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json
from datetime import datetime, timedelta
import psutil
import os

from data.sync.dmm.dmm_sync_manager import DMMSyncConfig, DMMSyncManager


@pytest.fixture(scope="session")
def event_loop():
    """セッションレベルのイベントループ"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def dmm_api_config():
    """DMM API設定"""
    return {
        'api_id': 'test_api_id',
        'affiliate_id': 'test_affiliate_id',
        'base_url': 'https://api.dmm.com/affiliate/v3/ItemList',
        'site': 'FANZA',
        'service': 'digital',
        'floor': 'videoa',
        'rate_limit': 1.0,
        'timeout': 30,
        'batch_size': 50
    }


@pytest.fixture
def dmm_sync_config():
    """DMM同期設定"""
    return DMMSyncConfig(
        api_id="test_api_id",
        affiliate_id="test_affiliate_id",
        rate_limit_delay=0.1,  # テスト用に短縮
        max_retries=2,
        timeout=10,
        batch_size=10,
        max_pages=2
    )


@pytest.fixture
def sample_dmm_api_response():
    """サンプルDMM APIレスポンス"""
    return {
        "result": {
            "status": 200,
            "result_count": 2,
            "total_count": 1000,
            "first_position": 1,
            "items": [
                {
                    "content_id": "test_content_001",
                    "product_id": "test_product_001",
                    "title": "テスト動画1",
                    "URL": "https://test.dmm.com/video1",
                    "date": "2024-01-15 12:00:00",
                    "imageURL": {
                        "list": "https://test.dmm.com/image1_list.jpg",
                        "small": "https://test.dmm.com/image1_small.jpg",
                        "large": "https://test.dmm.com/image1_large.jpg"
                    },
                    "sampleImageURL": {
                        "sample_s": {
                            "image": ["https://test.dmm.com/sample1_1.jpg", "https://test.dmm.com/sample1_2.jpg"]
                        }
                    },
                    "prices": [
                        {
                            "type": "rental",
                            "price": "500円",
                            "list_price": "800円"
                        }
                    ],
                    "iteminfo": {
                        "genre": [
                            {"id": "101", "name": "ジャンル1"},
                            {"id": "102", "name": "ジャンル2"}
                        ],
                        "maker": [
                            {"id": "201", "name": "メーカー1"}
                        ],
                        "actress": [
                            {"id": "301", "name": "女優1"},
                            {"id": "302", "name": "女優2"}
                        ],
                        "series": [
                            {"id": "401", "name": "シリーズ1"}
                        ]
                    }
                },
                {
                    "content_id": "test_content_002",
                    "product_id": "test_product_002",
                    "title": "テスト動画2",
                    "URL": "https://test.dmm.com/video2",
                    "date": "2024-01-16 15:30:00",
                    "imageURL": {
                        "list": "https://test.dmm.com/image2_list.jpg",
                        "small": "https://test.dmm.com/image2_small.jpg",
                        "large": "https://test.dmm.com/image2_large.jpg"
                    },
                    "prices": [
                        {
                            "type": "permanent",
                            "price": "1200円",
                            "list_price": "1500円"
                        }
                    ],
                    "iteminfo": {
                        "genre": [
                            {"id": "103", "name": "ジャンル3"}
                        ],
                        "maker": [
                            {"id": "202", "name": "メーカー2"}
                        ],
                        "actress": [
                            {"id": "303", "name": "女優3"}
                        ]
                    }
                }
            ]
        }
    }


@pytest.fixture
def sample_raw_data():
    """サンプル生データ"""
    return [
        {
            "review_id": "r001",
            "content_id": "test_content_001",
            "user_id": "u001",
            "rating": 4.5,
            "comment": "面白かった",
            "review_date": "2024-01-20 10:00:00",
            "helpful_count": 5
        },
        {
            "review_id": "r002",
            "content_id": "test_content_002",
            "user_id": "u002",
            "rating": 3.8,
            "comment": "普通でした",
            "review_date": "2024-01-21 14:30:00",
            "helpful_count": 2
        },
        {
            "review_id": "r003",
            "content_id": "test_content_001",
            "user_id": "u003",
            "rating": 5.0,
            "comment": "最高！",
            "review_date": "2024-01-22 18:45:00",
            "helpful_count": 8
        }
    ]


@pytest.fixture
def large_dataset():
    """大規模データセット（パフォーマンステスト用）"""
    dataset = []
    for i in range(10000):  # 10,000件のデータ
        dataset.append({
            "review_id": f"r{i:06d}",
            "content_id": f"content_{i % 100:03d}",  # 100種類のコンテンツIDを循環
            "user_id": f"u{i % 1000:04d}",  # 1,000種類のユーザーIDを循環
            "rating": 1.0 + (i % 5),  # 1.0-5.0の評価
            "comment": f"コメント{i}です。" * (i % 10 + 1),  # 長さ可変コメント
            "review_date": (datetime.now() - timedelta(days=i % 365)).strftime("%Y-%m-%d %H:%M:%S"),
            "helpful_count": i % 20
        })
    return dataset


@pytest.fixture
def temp_data_dir():
    """テンポラリデータディレクトリ"""
    temp_dir = tempfile.mkdtemp(prefix="test_data_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_performance_dir():
    """パフォーマンステスト用テンポラリディレクトリ"""
    temp_dir = tempfile.mkdtemp(prefix="test_performance_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def pipeline_config(temp_data_dir):
    """パイプライン設定"""
    return {
        'raw_data_dir': temp_data_dir / 'raw',
        'processed_data_dir': temp_data_dir / 'processed',
        'cleaned_data_dir': temp_data_dir / 'cleaned',
        'embedding_data_dir': temp_data_dir / 'embeddings',
        'log_dir': temp_data_dir / 'logs',
        'batch_size': 100,
        'max_workers': 2,  # テスト環境では小さめ
        'quality_threshold': 0.8,
        'error_tolerance': 0.05
    }


@pytest.fixture
def performance_config():
    """パフォーマンステスト設定"""
    return {
        'max_memory_mb': 500,  # 500MB制限
        'max_processing_time_seconds': 60,  # 60秒制限
        'concurrent_workers': 4,
        'batch_sizes': [100, 500, 1000, 2000],
        'benchmark_target_rps': 10,  # 10 records per second
        'memory_sample_interval': 0.1  # 100ms間隔でメモリ測定
    }


@pytest.fixture
def mock_supabase_client():
    """モックSupabaseクライアント"""
    mock_client = Mock()
    
    # table()メソッドのモック
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    
    # select()メソッドのモック
    mock_select = Mock()
    mock_table.select.return_value = mock_select
    
    # eq()メソッドのモック
    mock_eq = Mock()
    mock_select.eq.return_value = mock_eq
    
    # execute()メソッドのモック
    mock_execute = Mock()
    mock_eq.execute.return_value = mock_execute
    mock_eq.single.return_value = mock_execute
    
    # insert()とupdate()のモック
    mock_insert = Mock()
    mock_update = Mock()
    mock_table.insert.return_value = mock_insert
    mock_table.update.return_value = mock_update
    mock_insert.execute.return_value = Mock(data=[{"id": "test_id"}])
    mock_update.execute.return_value = Mock(data=[{"id": "test_id"}])
    
    # データのモック
    mock_execute.data = []  # 既存データなし（新規作成）
    
    return mock_client


@pytest.fixture
def mock_dmm_sync_manager(dmm_sync_config, mock_supabase_client):
    """モックDMM同期マネージャー"""
    manager = DMMSyncManager(dmm_sync_config)
    manager.supabase = mock_supabase_client
    return manager


@pytest.fixture
def mock_rate_limiter():
    """モックレートリミッター"""
    class MockRateLimiter:
        def __init__(self, calls_per_second=1.0):
            self.calls_per_second = calls_per_second
            self.call_count = 0
            self.start_time = datetime.now()
        
        async def acquire(self):
            """レート制限の取得（テスト用なので即座に許可）"""
            self.call_count += 1
            return True
        
        def get_stats(self):
            elapsed = (datetime.now() - self.start_time).total_seconds()
            current_rate = self.call_count / max(elapsed, 0.001)
            return {
                'total_calls': self.call_count,
                'elapsed_time': elapsed,
                'current_rate': current_rate,
                'rate_limit': self.calls_per_second
            }
    
    return MockRateLimiter()


@pytest.fixture
def error_injection():
    """エラー注入用ヘルパー"""
    class ErrorInjection:
        def __init__(self):
            self.error_scenarios = {
                'network_timeout': lambda: TimeoutError("Network timeout"),
                'api_rate_limit': lambda: Exception("API rate limit exceeded"),
                'invalid_response': lambda: ValueError("Invalid JSON response"),
                'database_error': lambda: Exception("Database connection failed"),
                'memory_error': lambda: MemoryError("Out of memory")
            }
        
        def inject(self, scenario_name):
            """指定されたエラーシナリオを発生させる"""
            if scenario_name in self.error_scenarios:
                raise self.error_scenarios[scenario_name]()
            else:
                raise ValueError(f"Unknown error scenario: {scenario_name}")
    
    return ErrorInjection()


@pytest.fixture
def memory_monitor():
    """メモリ監視ヘルパー"""
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline = self.get_memory_usage()
            self.peak = self.baseline
            self.samples = [self.baseline]
        
        def get_memory_usage(self):
            """現在のメモリ使用量を取得（MB）"""
            return self.process.memory_info().rss / 1024 / 1024
        
        def sample(self):
            """メモリ使用量をサンプリング"""
            current = self.get_memory_usage()
            self.samples.append(current)
            if current > self.peak:
                self.peak = current
            return current
        
        def get_stats(self):
            """メモリ統計を取得"""
            current = self.get_memory_usage()
            return {
                'baseline': self.baseline,
                'current': current,
                'peak': self.peak,
                'increase': current - self.baseline,
                'peak_increase': self.peak - self.baseline,
                'samples': self.samples.copy()
            }
    
    return MemoryMonitor()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """テスト環境セットアップ（各テストで自動実行）"""
    # テスト前の環境設定
    original_env = os.environ.copy()
    
    # テスト用環境変数設定
    os.environ['TEST_MODE'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # テスト後のクリーンアップ
    os.environ.clear()
    os.environ.update(original_env)