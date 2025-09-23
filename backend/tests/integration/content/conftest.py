"""
コンテンツAPI統合テスト用設定

テストフィクスチャとセットアップ機能を提供
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import httpx
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def event_loop():
    """セッションスコープのイベントループ"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def content_api_test_environment():
    """コンテンツAPI統合テスト環境（セッションレベル）"""
    # テスト用一時ディレクトリ作成
    base_temp_dir = tempfile.mkdtemp(prefix="content_api_test_session_")

    # ディレクトリ構造作成
    directories = [
        "data", "logs", "mocks", "fixtures", "reports"
    ]

    env = {"base_dir": Path(base_temp_dir)}

    for dir_name in directories:
        dir_path = Path(base_temp_dir) / dir_name
        dir_path.mkdir(exist_ok=True)
        env[f"{dir_name}_dir"] = dir_path

    # 環境変数設定
    os.environ['CONTENT_API_TEST_MODE'] = 'true'
    os.environ['CONTENT_API_TEST_BASE_DIR'] = str(base_temp_dir)

    yield env

    # クリーンアップ
    shutil.rmtree(base_temp_dir, ignore_errors=True)
    os.environ.pop('CONTENT_API_TEST_MODE', None)
    os.environ.pop('CONTENT_API_TEST_BASE_DIR', None)


@pytest.fixture
def mock_supabase_environment():
    """モックSupabase環境"""
    class MockSupabaseEnvironment:
        def __init__(self):
            self.data = {
                'videos': [],
                'users': [],
                'video_embeddings': [],
                'user_embeddings': [],
                'likes': [],
                'user_video_decisions': []
            }

        def add_videos(self, videos: List[Dict[str, Any]]):
            self.data['videos'].extend(videos)

        def add_users(self, users: List[Dict[str, Any]]):
            self.data['users'].extend(users)

        def add_video_embeddings(self, embeddings: List[Dict[str, Any]]):
            self.data['video_embeddings'].extend(embeddings)

        def add_user_embeddings(self, embeddings: List[Dict[str, Any]]):
            self.data['user_embeddings'].extend(embeddings)

        def add_likes(self, likes: List[Dict[str, Any]]):
            self.data['likes'].extend(likes)

        def clear_all(self):
            for key in self.data:
                self.data[key].clear()

        def get_table_data(self, table_name: str) -> List[Dict[str, Any]]:
            return self.data.get(table_name, [])

    return MockSupabaseEnvironment()


@pytest.fixture
def content_api_test_data():
    """コンテンツAPIテスト用データ"""
    # 基本的なテストデータセット
    videos = [
        {
            "id": f"video_{i}",
            "title": f"Test Video {i}",
            "description": f"Description for test video {i}",
            "thumbnail_url": f"https://example.com/thumb{i}.jpg",
            "preview_video_url": f"https://example.com/preview{i}.mp4",
            "maker": f"Maker{i % 5}",
            "genre": ["Action", "Romance", "Comedy", "Drama", "Thriller"][i % 5],
            "price": 1000 + (i * 100),
            "duration_seconds": 3600 + (i * 60),
            "published_at": f"2024-0{(i % 9) + 1}-01T00:00:00Z",
            "performers": [f"Performer{j}" for j in range(i % 3 + 1)],
            "tags": [f"tag{j}" for j in range(i % 4 + 1)]
        }
        for i in range(100)
    ]

    users = [
        {
            "id": f"user_{i}",
            "email": f"user{i}@example.com",
            "created_at": f"2024-01-{(i % 28) + 1:02d}T00:00:00Z"
        }
        for i in range(20)
    ]

    # ランダムな埋め込みベクター
    np.random.seed(42)
    video_embeddings = [
        {
            "video_id": f"video_{i}",
            "embedding": np.random.random(768).tolist(),
            "updated_at": "2024-01-01T00:00:00Z"
        }
        for i in range(100)
    ]

    user_embeddings = [
        {
            "user_id": f"user_{i}",
            "embedding": np.random.random(768).tolist(),
            "updated_at": "2024-01-01T00:00:00Z"
        }
        for i in range(20)
    ]

    # いいねデータ
    likes = []
    for user_i in range(20):
        for video_i in range(0, 100, 5):  # ユーザーごとに20個の動画にいいね
            if (user_i + video_i) % 3 == 0:  # 3分の1の確率でいいね
                likes.append({
                    "user_id": f"user_{user_i}",
                    "video_id": f"video_{video_i}",
                    "created_at": f"2024-01-{((user_i + video_i) % 28) + 1:02d}T00:00:00Z"
                })

    return {
        'videos': videos,
        'users': users,
        'video_embeddings': video_embeddings,
        'user_embeddings': user_embeddings,
        'likes': likes
    }


@pytest.fixture
def performance_test_config():
    """パフォーマンステスト設定"""
    return {
        'max_response_time_ms': 500,
        'max_concurrent_requests': 10,
        'sustained_load_requests': 50,
        'acceptable_error_rate': 0.05,  # 5%
        'cache_ttl_seconds': 300
    }


@pytest.fixture
async def http_client_session():
    """HTTPクライアントセッション"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest.fixture
def mock_auth_user():
    """認証済みユーザーのモック"""
    return {
        'user_id': 'test-user-123',
        'email': 'test@example.com',
        'authenticated': True
    }


@pytest.fixture
def mock_anonymous_user():
    """匿名ユーザーのモック"""
    return {
        'user_id': '',
        'authenticated': False
    }


@pytest.fixture
def content_api_urls():
    """コンテンツAPIのURL設定"""
    base_url = os.getenv('SUPABASE_URL', 'http://localhost:54321')
    return {
        'base_url': base_url,
        'functions_url': f"{base_url}/functions/v1",
        'content_feed': f"{base_url}/functions/v1/content/feed",
        'content_search': f"{base_url}/functions/v1/content/search",
        'recommendations': f"{base_url}/functions/v1/content/recommendations"
    }


@pytest.fixture
def test_request_headers():
    """テストリクエスト用ヘッダー"""
    return {
        'Content-Type': 'application/json',
        'User-Agent': 'ContentAPI-Test-Suite/1.0',
        'X-Test-Environment': 'true'
    }


@pytest.fixture
def auth_request_headers(mock_auth_user):
    """認証付きリクエスト用ヘッダー"""
    return {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer mock-jwt-token',
        'User-Agent': 'ContentAPI-Test-Suite/1.0',
        'X-Test-Environment': 'true'
    }


class ContentAPITestHelper:
    """コンテンツAPIテストヘルパークラス"""

    @staticmethod
    def validate_video_data(video: Dict[str, Any]) -> bool:
        """動画データの妥当性検証"""
        required_fields = ['id', 'title', 'thumbnail_url', 'maker', 'genre']
        return all(field in video and video[field] is not None for field in required_fields)

    @staticmethod
    def validate_feed_response(response_data: Dict[str, Any]) -> bool:
        """フィードレスポンスの妥当性検証"""
        if not response_data.get('success'):
            return False

        data = response_data.get('data', {})
        if 'videos' not in data:
            return False

        videos = data['videos']
        if not isinstance(videos, list):
            return False

        return all(ContentAPITestHelper.validate_video_data(video) for video in videos)

    @staticmethod
    def calculate_diversity_score(videos: List[Dict[str, Any]]) -> float:
        """動画リストの多様性スコア計算"""
        if not videos:
            return 0.0

        genres = set(video.get('genre') for video in videos if video.get('genre'))
        makers = set(video.get('maker') for video in videos if video.get('maker'))

        genre_diversity = len(genres) / len(videos)
        maker_diversity = len(makers) / len(videos)

        return (genre_diversity + maker_diversity) / 2

    @staticmethod
    async def measure_request_time(client: httpx.AsyncClient, url: str, json_data: Dict[str, Any], headers: Dict[str, str]) -> tuple:
        """リクエスト時間測定"""
        import time

        start_time = time.time()
        response = await client.post(url, json=json_data, headers=headers)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000
        return response, response_time_ms


@pytest.fixture
def content_api_helper():
    """コンテンツAPIテストヘルパー"""
    return ContentAPITestHelper()


# カスタムマーカー設定
def pytest_configure(config):
    """カスタムマーカー設定"""
    config.addinivalue_line("markers", "content_api: Content API related tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "auth: Authentication related tests")


# テスト実行前のセットアップ
def pytest_sessionstart(session):
    """テストセッション開始時処理"""
    print("\n🚀 Starting Content API Integration Test Session...")
    print("Setting up comprehensive test environment...")


def pytest_sessionfinish(session, exitstatus):
    """テストセッション終了時処理"""
    print("\n✅ Content API Integration Test Session Complete")
    print(f"Exit status: {exitstatus}")


# テストファイル毎のセットアップ
def pytest_runtest_setup(item):
    """各テスト実行前処理"""
    if item.get_closest_marker("slow"):
        # 環境変数でスロー テストスキップ
        if os.environ.get("SKIP_SLOW_TESTS", "false").lower() == "true":
            pytest.skip("Slow test skipped")

    if item.get_closest_marker("auth"):
        # 認証テスト用の特別な設定があればここで行う
        pass


# テスト結果レポート
def pytest_runtest_logreport(report):
    """テスト結果レポート"""
    if report.when == "call":
        if report.passed:
            print(f"✅ {report.nodeid}")
        elif report.failed:
            print(f"❌ {report.nodeid}")
        elif report.skipped:
            print(f"⏭️  {report.nodeid}")