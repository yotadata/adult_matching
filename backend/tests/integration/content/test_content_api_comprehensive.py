"""
統合コンテンツAPI包括的テスト

統合されたコンテンツAPIの全フィードタイプとエラーハンドリングの
包括的テストスイート
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import httpx
import pandas as pd
import numpy as np

from backend.tests.fixtures.test_data import (
    create_test_videos,
    create_test_users,
    create_test_embeddings,
    create_test_likes
)


class ContentAPITestSuite:
    """統合コンテンツAPIテストスイート"""

    def __init__(self, base_url: str = "http://localhost:54321"):
        self.base_url = base_url
        self.functions_url = f"{base_url}/functions/v1"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


@pytest.fixture(scope="session")
async def content_api_suite():
    """コンテンツAPIテストスイート"""
    async with ContentAPITestSuite() as suite:
        yield suite


@pytest.fixture(scope="session")
def comprehensive_test_data():
    """包括的テストデータ"""
    # 多様な動画データセット
    videos = create_test_videos(count=200, diverse=True)

    # ユーザーデータ
    users = create_test_users(count=50)

    # 埋め込みデータ
    video_embeddings = create_test_embeddings(
        entity_type="video",
        count=200,
        dimension=768
    )
    user_embeddings = create_test_embeddings(
        entity_type="user",
        count=50,
        dimension=768
    )

    # いいねデータ
    likes = create_test_likes(
        user_count=50,
        video_count=200,
        interaction_ratio=0.3
    )

    return {
        'videos': videos,
        'users': users,
        'video_embeddings': video_embeddings,
        'user_embeddings': user_embeddings,
        'likes': likes
    }


class TestContentFeedEndpoints:
    """コンテンツフィードエンドポイントのテスト"""

    @pytest.mark.asyncio
    async def test_explore_feed_basic(self, content_api_suite, comprehensive_test_data):
        """探索フィード基本テスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "explore",
                "limit": 20,
                "offset": 0
            },
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "videos" in data["data"]
        assert len(data["data"]["videos"]) <= 20
        assert data["data"]["feed_type"] == "explore"
        assert "diversity_score" in data["data"]
        assert "pagination" in data["data"]

    @pytest.mark.asyncio
    async def test_personalized_feed_authenticated(self, content_api_suite, comprehensive_test_data):
        """パーソナライズドフィード（認証済み）テスト"""
        # Mock JWT token
        auth_token = "Bearer mock-jwt-token"

        with patch('supabase.functions._shared.auth.authenticateUser') as mock_auth:
            mock_auth.return_value = {
                'authenticated': True,
                'user_id': 'test-user-123'
            }

            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json={
                    "feed_type": "personalized",
                    "limit": 15,
                    "user_id": "test-user-123"
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": auth_token
                }
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["data"]["feed_type"] == "personalized"
            assert len(data["data"]["videos"]) <= 15

    @pytest.mark.asyncio
    async def test_latest_feed(self, content_api_suite, comprehensive_test_data):
        """最新フィードテスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "latest",
                "limit": 10
            },
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["feed_type"] == "latest"
        assert len(data["data"]["videos"]) <= 10

        # 時系列順序の確認
        videos = data["data"]["videos"]
        if len(videos) > 1:
            for i in range(len(videos) - 1):
                if videos[i].get("published_at") and videos[i+1].get("published_at"):
                    assert videos[i]["published_at"] >= videos[i+1]["published_at"]

    @pytest.mark.asyncio
    async def test_popular_feed(self, content_api_suite, comprehensive_test_data):
        """人気フィードテスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "popular",
                "limit": 25
            },
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["feed_type"] == "popular"
        assert len(data["data"]["videos"]) <= 25

    @pytest.mark.asyncio
    async def test_random_feed(self, content_api_suite, comprehensive_test_data):
        """ランダムフィードテスト"""
        # 複数回リクエストして結果が異なることを確認
        responses = []

        for _ in range(3):
            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json={
                    "feed_type": "random",
                    "limit": 10
                },
                headers={"Content-Type": "application/json"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            responses.append([video["id"] for video in data["data"]["videos"]])

            # 短時間待機
            await asyncio.sleep(0.1)

        # 結果が異なることを確認（ランダム性）
        assert not all(resp == responses[0] for resp in responses[1:])


class TestContentFeedFiltering:
    """コンテンツフィードフィルタリングのテスト"""

    @pytest.mark.asyncio
    async def test_feed_with_exclude_ids(self, content_api_suite, comprehensive_test_data):
        """除外IDフィルタテスト"""
        exclude_ids = ["video_1", "video_2", "video_3"]

        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "explore",
                "limit": 20,
                "exclude_ids": exclude_ids
            },
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()

        returned_ids = [video["id"] for video in data["data"]["videos"]]
        for exclude_id in exclude_ids:
            assert exclude_id not in returned_ids

    @pytest.mark.asyncio
    async def test_feed_pagination(self, content_api_suite, comprehensive_test_data):
        """ページネーションテスト"""
        limit = 10

        # 第1ページ
        response1 = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "latest",
                "limit": limit,
                "offset": 0
            },
            headers={"Content-Type": "application/json"}
        )

        # 第2ページ
        response2 = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "latest",
                "limit": limit,
                "offset": limit
            },
            headers={"Content-Type": "application/json"}
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        ids1 = [video["id"] for video in data1["data"]["videos"]]
        ids2 = [video["id"] for video in data2["data"]["videos"]]

        # 重複なし
        assert len(set(ids1) & set(ids2)) == 0


class TestContentAPIPerformance:
    """コンテンツAPIパフォーマンステスト"""

    @pytest.mark.asyncio
    async def test_feed_response_time(self, content_api_suite, comprehensive_test_data):
        """フィードレスポンス時間テスト"""
        feed_types = ["explore", "latest", "popular", "random"]

        for feed_type in feed_types:
            start_time = time.time()

            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json={
                    "feed_type": feed_type,
                    "limit": 20
                },
                headers={"Content-Type": "application/json"}
            )

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms

            assert response.status_code == 200
            assert response_time < 500  # 500ms以下

            # レスポンスヘッダーの処理時間も確認
            if "X-Processing-Time" in response.headers:
                processing_time = float(response.headers["X-Processing-Time"].replace("ms", ""))
                assert processing_time < 500

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, content_api_suite, comprehensive_test_data):
        """同時リクエストテスト"""
        async def make_request(feed_type: str, request_id: int):
            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json={
                    "feed_type": feed_type,
                    "limit": 15
                },
                headers={"Content-Type": "application/json"}
            )
            return response.status_code, request_id

        # 10並行リクエスト
        tasks = [
            make_request("explore", i) for i in range(10)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # 全て成功
        for status_code, _ in results:
            assert status_code == 200

        # 合計時間が妥当
        total_time = (end_time - start_time) * 1000
        assert total_time < 3000  # 3秒以下


class TestContentAPIErrorHandling:
    """コンテンツAPIエラーハンドリングテスト"""

    @pytest.mark.asyncio
    async def test_invalid_feed_type(self, content_api_suite):
        """無効なフィードタイプテスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "invalid_feed_type",
                "limit": 20
            },
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_invalid_limit_values(self, content_api_suite):
        """無効なリミット値テスト"""
        invalid_limits = [-1, 0, 101, "invalid"]

        for limit in invalid_limits:
            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json={
                    "feed_type": "explore",
                    "limit": limit
                },
                headers={"Content-Type": "application/json"}
            )

            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False

    @pytest.mark.asyncio
    async def test_malformed_json(self, content_api_suite):
        """不正なJSONテスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            content="invalid json{",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_content_type(self, content_api_suite):
        """Content-Typeヘッダー欠如テスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "explore",
                "limit": 20
            }
            # Content-Typeヘッダーなし
        )

        # サーバーによっては受け入れる場合もあるため、適切に処理されることを確認
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_database_connection_failure(self, content_api_suite):
        """データベース接続障害テスト"""
        with patch('supabase.functions._shared.database.getSupabaseClientFromRequest') as mock_client:
            # データベース接続エラーをシミュレート
            mock_client.side_effect = Exception("Database connection failed")

            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json={
                    "feed_type": "explore",
                    "limit": 20
                },
                headers={"Content-Type": "application/json"}
            )

            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "error" in data


class TestContentAPIAuth:
    """コンテンツAPI認証テスト"""

    @pytest.mark.asyncio
    async def test_personalized_feed_without_auth(self, content_api_suite):
        """認証なしパーソナライズドフィードテスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "personalized",
                "limit": 20
            },
            headers={"Content-Type": "application/json"}
        )

        # 認証なしの場合、人気フィードにフォールバック
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_invalid_auth_token(self, content_api_suite):
        """無効な認証トークンテスト"""
        invalid_token = "Bearer invalid-token"

        with patch('supabase.functions._shared.auth.authenticateUser') as mock_auth:
            mock_auth.return_value = {
                'authenticated': False,
                'error': 'Invalid token'
            }

            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json={
                    "feed_type": "personalized",
                    "limit": 20
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": invalid_token
                }
            )

            # 無効トークンでもフォールバックして成功するべき
            assert response.status_code == 200


class TestContentAPIDataQuality:
    """コンテンツAPIデータ品質テスト"""

    @pytest.mark.asyncio
    async def test_video_data_completeness(self, content_api_suite, comprehensive_test_data):
        """動画データ完全性テスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "explore",
                "limit": 20
            },
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()

        required_fields = ["id", "title", "thumbnail_url", "maker", "genre"]
        optional_fields = ["description", "preview_video_url", "performers", "tags"]

        for video in data["data"]["videos"]:
            # 必須フィールドの確認
            for field in required_fields:
                assert field in video
                assert video[field] is not None

            # 配列フィールドの確認
            if "performers" in video:
                assert isinstance(video["performers"], list)
            if "tags" in video:
                assert isinstance(video["tags"], list)

    @pytest.mark.asyncio
    async def test_diversity_metrics(self, content_api_suite, comprehensive_test_data):
        """多様性メトリクステスト"""
        response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "explore",
                "limit": 30
            },
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()

        if "diversity_score" in data["data"]:
            diversity_score = data["data"]["diversity_score"]
            assert 0 <= diversity_score <= 1

        # ジャンルの多様性確認
        videos = data["data"]["videos"]
        genres = set(video["genre"] for video in videos if video.get("genre"))

        # 少なくとも2つ以上のジャンルが含まれることを期待
        if len(videos) >= 10:
            assert len(genres) >= 2


class TestContentAPIIntegration:
    """コンテンツAPI統合テスト"""

    @pytest.mark.asyncio
    async def test_end_to_end_user_journey(self, content_api_suite, comprehensive_test_data):
        """エンドツーエンドユーザージャーニーテスト"""
        # 1. 探索フィードで動画発見
        explore_response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "explore",
                "limit": 10
            },
            headers={"Content-Type": "application/json"}
        )

        assert explore_response.status_code == 200
        explore_data = explore_response.json()
        explore_videos = explore_data["data"]["videos"]

        # 2. いくつかの動画IDを除外リストに追加してパーソナライズドフィード取得
        exclude_ids = [video["id"] for video in explore_videos[:3]]

        personalized_response = await content_api_suite.client.post(
            f"{content_api_suite.functions_url}/content/feed",
            json={
                "feed_type": "personalized",
                "limit": 15,
                "exclude_ids": exclude_ids
            },
            headers={"Content-Type": "application/json"}
        )

        assert personalized_response.status_code == 200
        personalized_data = personalized_response.json()

        # 除外された動画が含まれていないことを確認
        personalized_ids = [video["id"] for video in personalized_data["data"]["videos"]]
        for exclude_id in exclude_ids:
            assert exclude_id not in personalized_ids

    @pytest.mark.asyncio
    async def test_cache_consistency(self, content_api_suite, comprehensive_test_data):
        """キャッシュ一貫性テスト"""
        # 同じパラメータで複数回リクエスト
        params = {
            "feed_type": "latest",
            "limit": 15,
            "offset": 0
        }

        responses = []
        for _ in range(3):
            response = await content_api_suite.client.post(
                f"{content_api_suite.functions_url}/content/feed",
                json=params,
                headers={"Content-Type": "application/json"}
            )

            assert response.status_code == 200
            responses.append(response.json())

        # キャッシュされた結果で一貫性があるかチェック（短時間内であれば）
        video_ids_first = [video["id"] for video in responses[0]["data"]["videos"]]

        for response in responses[1:]:
            video_ids = [video["id"] for video in response["data"]["videos"]]
            # 結果の順序が一致する（latestフィードなので）
            assert video_ids == video_ids_first


@pytest.mark.integration
@pytest.mark.slow
class TestContentAPILoad:
    """コンテンツAPIロードテスト"""

    @pytest.mark.asyncio
    async def test_sustained_load(self, content_api_suite, comprehensive_test_data):
        """持続負荷テスト"""
        async def make_sustained_requests():
            results = []
            for i in range(50):  # 50リクエスト
                start_time = time.time()

                response = await content_api_suite.client.post(
                    f"{content_api_suite.functions_url}/content/feed",
                    json={
                        "feed_type": "explore",
                        "limit": 20
                    },
                    headers={"Content-Type": "application/json"}
                )

                end_time = time.time()
                response_time = (end_time - start_time) * 1000

                results.append({
                    "request_id": i,
                    "status_code": response.status_code,
                    "response_time": response_time
                })

                if i % 10 == 0:
                    await asyncio.sleep(0.1)  # 短時間休憩

            return results

        results = await make_sustained_requests()

        # 全リクエスト成功
        success_count = sum(1 for r in results if r["status_code"] == 200)
        assert success_count >= 45  # 90%以上成功

        # 平均レスポンス時間
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        assert avg_response_time < 1000  # 1秒以下


# テスト実行用ヘルパー関数
def run_comprehensive_content_api_tests():
    """包括的コンテンツAPIテストの実行"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow",  # デフォルトでは高速テストのみ実行
        "--durations=10"
    ])


def run_full_content_api_tests():
    """完全コンテンツAPIテストの実行（ロードテスト含む）"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])


if __name__ == "__main__":
    run_comprehensive_content_api_tests()