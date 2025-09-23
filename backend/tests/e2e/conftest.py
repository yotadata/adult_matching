"""
E2E Test Configuration and Fixtures

エンドツーエンドテスト用設定とフィクスチャ
"""
import pytest
import asyncio
import aiohttp
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
import tempfile
import shutil
import psutil
import json
from datetime import datetime, timezone, timedelta


@pytest.fixture(scope="session")
def event_loop():
    """セッションレベルのイベントループ"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def system_config():
    """システム設定"""
    return {
        'base_url': 'http://localhost:8000',
        'timeout': 30,
        'max_retries': 3,
        'performance_thresholds': {
            'recommendation_latency_ms': 500,
            'api_response_time_ms': 200,
            'database_query_time_ms': 100,
            'concurrent_users': 100,
            'throughput_rps': 50
        },
        'test_data_size': {
            'small': 100,
            'medium': 1000, 
            'large': 10000
        }
    }


@pytest.fixture
async def http_client():
    """HTTPクライアントセッション"""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def performance_monitor():
    """パフォーマンス監視"""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.process = psutil.Process()
            self.start_time = None
            
        def start_monitoring(self, test_name: str):
            """監視開始"""
            self.start_time = time.time()
            self.metrics[test_name] = {
                'start_time': self.start_time,
                'initial_memory': self.process.memory_info().rss / 1024 / 1024,
                'initial_cpu': self.process.cpu_percent(),
                'operations': []
            }
            
        def record_operation(self, test_name: str, operation: str, duration_ms: float, success: bool = True):
            """操作記録"""
            if test_name in self.metrics:
                self.metrics[test_name]['operations'].append({
                    'operation': operation,
                    'duration_ms': duration_ms,
                    'success': success,
                    'timestamp': time.time()
                })
                
        def stop_monitoring(self, test_name: str):
            """監視終了"""
            if test_name in self.metrics:
                end_time = time.time()
                metric = self.metrics[test_name]
                metric.update({
                    'end_time': end_time,
                    'total_duration': end_time - metric['start_time'],
                    'final_memory': self.process.memory_info().rss / 1024 / 1024,
                    'final_cpu': self.process.cpu_percent(),
                    'memory_increase': self.process.memory_info().rss / 1024 / 1024 - metric['initial_memory']
                })
                
        def get_summary(self, test_name: str) -> Dict[str, Any]:
            """メトリクス要約取得"""
            if test_name not in self.metrics:
                return {}
                
            metric = self.metrics[test_name]
            operations = metric.get('operations', [])
            
            if not operations:
                return metric
                
            successful_ops = [op for op in operations if op['success']]
            failed_ops = [op for op in operations if not op['success']]
            
            durations = [op['duration_ms'] for op in successful_ops]
            
            summary = {
                **metric,
                'total_operations': len(operations),
                'successful_operations': len(successful_ops),
                'failed_operations': len(failed_ops),
                'success_rate': len(successful_ops) / len(operations) if operations else 0,
                'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
                'min_duration_ms': min(durations) if durations else 0,
                'max_duration_ms': max(durations) if durations else 0,
                'throughput_ops_per_sec': len(successful_ops) / metric.get('total_duration', 1)
            }
            
            return summary
    
    return PerformanceMonitor()


@pytest.fixture
def test_user_profiles():
    """テストユーザープロファイル"""
    return {
        'new_user': {
            'user_id': 'test_user_new',
            'preferences': [],
            'history': [],
            'demographics': {'age_group': '20-30', 'interests': ['anime', 'games']}
        },
        'active_user': {
            'user_id': 'test_user_active',
            'preferences': ['genre_1', 'genre_2', 'genre_3'],
            'history': ['video_1', 'video_2', 'video_3', 'video_4', 'video_5'],
            'demographics': {'age_group': '25-35', 'interests': ['movies', 'technology']}
        },
        'heavy_user': {
            'user_id': 'test_user_heavy',
            'preferences': ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5'],
            'history': [f'video_{i}' for i in range(1, 51)],  # 50 videos
            'demographics': {'age_group': '30-40', 'interests': ['entertainment', 'sports']}
        }
    }


@pytest.fixture
def test_video_catalog():
    """テストビデオカタログ"""
    videos = []
    for i in range(1, 101):  # 100 videos
        videos.append({
            'video_id': f'video_{i}',
            'title': f'テストビデオ{i}',
            'genre': f'genre_{i % 10 + 1}',
            'maker': f'maker_{i % 5 + 1}',
            'performers': [f'performer_{j}' for j in range(i % 3 + 1)],
            'tags': [f'tag_{k}' for k in range(i % 5 + 1)],
            'rating': 1.0 + (i % 5),
            'duration': 60 + (i % 120),  # 60-180分
            'release_date': datetime.now(timezone.utc) - timedelta(days=i),
            'thumbnail_url': f'https://test.example.com/thumb_{i}.jpg',
            'price': 500 + (i * 10),
            'features': {
                'popularity_score': i / 100.0,
                'quality_score': 0.5 + (i % 5) / 10.0,
                'recency_score': 1.0 - (i / 100.0)
            }
        })
    return videos


@pytest.fixture
def mock_ml_service():
    """MLサービスモック"""
    class MockMLService:
        def __init__(self):
            self.model_loaded = True
            self.inference_count = 0
            
        async def get_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict[str, Any]]:
            """推薦取得"""
            self.inference_count += 1
            
            # シミュレートされた推薦結果
            recommendations = []
            for i in range(num_recommendations):
                recommendations.append({
                    'video_id': f'rec_video_{i + 1}',
                    'score': 0.9 - (i * 0.05),
                    'reason': f'Based on your interest in genre_{i % 5 + 1}',
                    'confidence': 0.85 - (i * 0.02)
                })
                
            # 推薦レスポンス時間をシミュレート
            await asyncio.sleep(0.1)
            
            return recommendations
            
        async def update_user_embedding(self, user_id: str, interactions: List[Dict[str, Any]]) -> bool:
            """ユーザー埋め込み更新"""
            # 更新処理をシミュレート
            await asyncio.sleep(0.05)
            return True
            
        def get_stats(self) -> Dict[str, Any]:
            """統計取得"""
            return {
                'model_loaded': self.model_loaded,
                'inference_count': self.inference_count,
                'model_version': '1.0.0'
            }
    
    return MockMLService()


@pytest.fixture
def mock_database():
    """データベースモック"""
    class MockDatabase:
        def __init__(self):
            self.users = {}
            self.videos = {}
            self.interactions = []
            self.query_count = 0
            
        async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
            """ユーザー取得"""
            self.query_count += 1
            await asyncio.sleep(0.01)  # DB遅延シミュレート
            return self.users.get(user_id)
            
        async def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
            """ビデオ取得"""
            self.query_count += 1
            await asyncio.sleep(0.01)
            return self.videos.get(video_id)
            
        async def record_interaction(self, user_id: str, video_id: str, interaction_type: str) -> bool:
            """インタラクション記録"""
            self.query_count += 1
            interaction = {
                'user_id': user_id,
                'video_id': video_id,
                'interaction_type': interaction_type,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.interactions.append(interaction)
            await asyncio.sleep(0.005)
            return True
            
        async def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
            """ユーザー履歴取得"""
            self.query_count += 1
            user_interactions = [
                i for i in self.interactions 
                if i['user_id'] == user_id
            ][-limit:]
            await asyncio.sleep(0.02)
            return user_interactions
            
        def get_stats(self) -> Dict[str, Any]:
            """統計取得"""
            return {
                'total_users': len(self.users),
                'total_videos': len(self.videos),
                'total_interactions': len(self.interactions),
                'query_count': self.query_count
            }
            
        def seed_data(self, users: List[Dict], videos: List[Dict]):
            """テストデータ投入"""
            for user in users:
                self.users[user['user_id']] = user
            for video in videos:
                self.videos[video['video_id']] = video
    
    return MockDatabase()


@pytest.fixture
def system_under_test(mock_ml_service, mock_database, test_user_profiles, test_video_catalog):
    """テスト対象システム"""
    class SystemUnderTest:
        def __init__(self, ml_service, database):
            self.ml_service = ml_service
            self.database = database
            self.request_count = 0
            
        async def authenticate_user(self, user_id: str, token: str) -> bool:
            """ユーザー認証"""
            self.request_count += 1
            # 認証ロジックをシミュレート
            await asyncio.sleep(0.02)
            return token == f"valid_token_{user_id}"
            
        async def get_recommendations_for_user(self, user_id: str, num_recommendations: int = 10) -> Dict[str, Any]:
            """ユーザー推薦取得"""
            start_time = time.time()
            self.request_count += 1
            
            try:
                # ユーザー情報取得
                user = await self.database.get_user(user_id)
                if not user:
                    return {'error': 'User not found', 'recommendations': []}
                
                # ML推薦取得
                recommendations = await self.ml_service.get_recommendations(user_id, num_recommendations)
                
                # レスポンス時間計算
                response_time = (time.time() - start_time) * 1000
                
                return {
                    'user_id': user_id,
                    'recommendations': recommendations,
                    'response_time_ms': response_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                return {'error': str(e), 'recommendations': []}
                
        async def record_user_interaction(self, user_id: str, video_id: str, interaction_type: str) -> Dict[str, Any]:
            """ユーザーインタラクション記録"""
            start_time = time.time()
            self.request_count += 1
            
            try:
                # インタラクション記録
                success = await self.database.record_interaction(user_id, video_id, interaction_type)
                
                # ユーザー埋め込み更新（非同期）
                if success and interaction_type in ['like', 'dislike']:
                    await self.ml_service.update_user_embedding(user_id, [
                        {'video_id': video_id, 'interaction_type': interaction_type}
                    ])
                
                response_time = (time.time() - start_time) * 1000
                
                return {
                    'success': success,
                    'response_time_ms': response_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
                
        async def get_user_feed(self, user_id: str, page_size: int = 20) -> Dict[str, Any]:
            """ユーザーフィード取得"""
            start_time = time.time()
            self.request_count += 1
            
            try:
                # 推薦ベースのフィード生成
                recommendations_response = await self.get_recommendations_for_user(user_id, page_size)
                
                if 'error' in recommendations_response:
                    return recommendations_response
                
                # ビデオ詳細情報取得
                video_details = []
                for rec in recommendations_response['recommendations']:
                    video = await self.database.get_video(rec['video_id'])
                    if video:
                        video_details.append({
                            **video,
                            'recommendation_score': rec['score'],
                            'recommendation_reason': rec['reason']
                        })
                
                response_time = (time.time() - start_time) * 1000
                
                return {
                    'user_id': user_id,
                    'feed': video_details,
                    'response_time_ms': response_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                return {'error': str(e), 'feed': []}
                
        def get_system_stats(self) -> Dict[str, Any]:
            """システム統計"""
            return {
                'request_count': self.request_count,
                'ml_stats': self.ml_service.get_stats(),
                'db_stats': self.database.get_stats(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    # テストデータを投入
    users = list(test_user_profiles.values())
    mock_database.seed_data(users, test_video_catalog)
    
    return SystemUnderTest(mock_ml_service, mock_database)


@pytest.fixture
def load_test_config():
    """負荷テスト設定"""
    return {
        'concurrent_users': [1, 5, 10, 25, 50],
        'test_duration_seconds': 30,
        'ramp_up_time_seconds': 5,
        'operations_per_user': 10,
        'think_time_ms': 100,
        'scenarios': [
            'user_registration_and_onboarding',
            'recommendation_browsing',
            'video_interaction_workflow',
            'mixed_user_activity'
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """テスト環境セットアップ（各テストで自動実行）"""
    # テスト前の環境設定
    original_env = {}
    
    # テスト用環境変数設定
    test_env = {
        'TEST_MODE': 'e2e',
        'LOG_LEVEL': 'INFO',
        'PERFORMANCE_MONITORING': 'enabled'
    }
    
    import os
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # テスト後のクリーンアップ
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value