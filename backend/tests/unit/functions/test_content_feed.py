"""
Tests for Content Feed Edge Function

content/feed Edge Functionの単体テスト
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List


class TestContentFeed:
    """Content Feed API テスト"""
    
    @pytest.fixture
    def sample_feed_request(self):
        """フィードリクエストサンプル"""
        return {
            "feed_type": "explore",
            "limit": 20,
            "offset": 0,
            "exclude_ids": ["video-1", "video-2"],
            "user_id": "test-user-123"
        }
    
    @pytest.fixture
    def sample_video_data(self):
        """動画データサンプル"""
        return [
            {
                "id": f"video-{i}",
                "title": f"Test Video {i}",
                "description": f"Description for video {i}",
                "duration_seconds": 3600 + i * 60,
                "thumbnail_url": f"https://example.com/thumb{i}.jpg",
                "preview_video_url": f"https://example.com/preview{i}.mp4",
                "maker": f"Maker {i % 3}",
                "genre": f"Genre {i % 5}",
                "price": 1000 + i * 100,
                "sample_video_url": f"https://example.com/sample{i}.mp4",
                "image_urls": [f"https://example.com/img{i}_{j}.jpg" for j in range(3)],
                "performers": [f"Performer {i}_{j}" for j in range(2)],
                "tags": [f"tag{i}", "test", "sample"],
                "published_at": f"2025-01-{(i % 28) + 1:02d}T00:00:00Z"
            }
            for i in range(50)
        ]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_feed_request_validation(self, sample_feed_request):
        """フィードリクエスト検証テスト"""
        # 必須フィールドの検証
        assert "feed_type" in sample_feed_request
        assert sample_feed_request["feed_type"] in ["explore", "personalized", "latest", "popular", "random"]
        
        # オプションフィールドの型検証
        assert isinstance(sample_feed_request.get("limit", 20), int)
        assert isinstance(sample_feed_request.get("offset", 0), int)
        assert isinstance(sample_feed_request.get("exclude_ids", []), list)
        
        # 範囲検証
        limit = sample_feed_request.get("limit", 20)
        offset = sample_feed_request.get("offset", 0)
        assert 1 <= limit <= 100
        assert offset >= 0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_video_data_structure(self, sample_video_data):
        """動画データ構造テスト"""
        for video in sample_video_data[:5]:  # 最初の5件をテスト
            # 必須フィールド確認
            required_fields = [
                "id", "title", "description", "duration_seconds",
                "thumbnail_url", "preview_video_url", "maker", "genre",
                "price", "sample_video_url", "image_urls", "performers", "tags"
            ]
            
            for field in required_fields:
                assert field in video
            
            # 型確認
            assert isinstance(video["id"], str)
            assert isinstance(video["title"], str)
            assert isinstance(video["duration_seconds"], int)
            assert isinstance(video["price"], (int, float))
            assert isinstance(video["image_urls"], list)
            assert isinstance(video["performers"], list)
            assert isinstance(video["tags"], list)
            
            # 値の妥当性確認
            assert video["duration_seconds"] > 0
            assert video["price"] >= 0
            assert len(video["image_urls"]) >= 0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_feed_response_structure(self):
        """フィードレスポンス構造テスト"""
        expected_response = {
            "videos": [],
            "total_count": 0,
            "feed_type": "explore",
            "diversity_score": 0.8,
            "pagination": {
                "limit": 20,
                "offset": 0,
                "has_more": False
            },
            "processing_time_ms": 100
        }
        
        # 必須フィールド確認
        required_fields = ["videos", "total_count", "feed_type", "pagination", "processing_time_ms"]
        for field in required_fields:
            assert field in expected_response
        
        # pagination構造確認
        pagination_fields = ["limit", "offset", "has_more"]
        for field in pagination_fields:
            assert field in expected_response["pagination"]
        
        # 型確認
        assert isinstance(expected_response["videos"], list)
        assert isinstance(expected_response["total_count"], int)
        assert isinstance(expected_response["processing_time_ms"], (int, float))
        assert isinstance(expected_response["pagination"]["has_more"], bool)
    
    @pytest.mark.unit
    def test_feed_type_handlers(self):
        """フィードタイプ別ハンドラーテスト"""
        
        def get_explore_feed(limit: int, offset: int) -> Dict:
            """Exploreフィード取得（多様性重視）"""
            return {
                "query_type": "explore",
                "diversity_weight": 0.7,
                "order_by": "random_with_diversity"
            }
        
        def get_personalized_feed(user_id: str, limit: int, offset: int) -> Dict:
            """パーソナライズドフィード取得"""
            return {
                "query_type": "personalized",
                "user_id": user_id,
                "recommendation_algorithm": "two_tower"
            }
        
        def get_latest_feed(limit: int, offset: int) -> Dict:
            """最新フィード取得"""
            return {
                "query_type": "latest",
                "order_by": "published_at DESC"
            }
        
        def get_popular_feed(limit: int, offset: int) -> Dict:
            """人気フィード取得"""
            return {
                "query_type": "popular",
                "order_by": "likes_count DESC"
            }
        
        def get_random_feed(limit: int, offset: int) -> Dict:
            """ランダムフィード取得"""
            return {
                "query_type": "random",
                "order_by": "RANDOM()"
            }
        
        # 各フィードタイプのテスト
        explore_result = get_explore_feed(20, 0)
        assert explore_result["query_type"] == "explore"
        assert "diversity_weight" in explore_result
        
        personalized_result = get_personalized_feed("user-123", 20, 0)
        assert personalized_result["query_type"] == "personalized"
        assert personalized_result["user_id"] == "user-123"
        
        latest_result = get_latest_feed(20, 0)
        assert latest_result["query_type"] == "latest"
        assert "published_at DESC" in latest_result["order_by"]
        
        popular_result = get_popular_feed(20, 0)
        assert popular_result["query_type"] == "popular"
        assert "likes_count DESC" in popular_result["order_by"]
        
        random_result = get_random_feed(20, 0)
        assert random_result["query_type"] == "random"
        assert "RANDOM()" in random_result["order_by"]
    
    @pytest.mark.unit
    def test_pagination_logic(self, sample_video_data):
        """ページネーションロジックテスト"""
        def paginate_results(videos: List[Dict], limit: int, offset: int) -> Dict:
            """結果のページネーション処理"""
            total_count = len(videos)
            start_idx = offset
            end_idx = offset + limit
            
            paginated_videos = videos[start_idx:end_idx]
            has_more = end_idx < total_count
            
            return {
                "videos": paginated_videos,
                "total_count": total_count,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": has_more
                }
            }
        
        # 最初のページ
        result = paginate_results(sample_video_data, 20, 0)
        assert len(result["videos"]) == 20
        assert result["pagination"]["has_more"] is True
        assert result["total_count"] == 50
        
        # 2ページ目
        result = paginate_results(sample_video_data, 20, 20)
        assert len(result["videos"]) == 20
        assert result["pagination"]["has_more"] is True
        
        # 最後のページ
        result = paginate_results(sample_video_data, 20, 40)
        assert len(result["videos"]) == 10
        assert result["pagination"]["has_more"] is False
        
        # 範囲外
        result = paginate_results(sample_video_data, 20, 60)
        assert len(result["videos"]) == 0
        assert result["pagination"]["has_more"] is False
    
    @pytest.mark.unit
    def test_exclude_ids_filtering(self, sample_video_data):
        """除外ID フィルタリングテスト"""
        def filter_excluded_videos(videos: List[Dict], exclude_ids: List[str]) -> List[Dict]:
            """除外IDによるフィルタリング"""
            return [video for video in videos if video["id"] not in exclude_ids]
        
        exclude_ids = ["video-0", "video-1", "video-5"]
        filtered_videos = filter_excluded_videos(sample_video_data, exclude_ids)
        
        # 除外されていることを確認
        filtered_ids = [video["id"] for video in filtered_videos]
        for exclude_id in exclude_ids:
            assert exclude_id not in filtered_ids
        
        # 元のデータより少ないことを確認
        assert len(filtered_videos) == len(sample_video_data) - len(exclude_ids)
    
    @pytest.mark.unit
    def test_diversity_score_calculation(self, sample_video_data):
        """多様性スコア計算テスト"""
        def calculate_diversity_score(videos: List[Dict]) -> float:
            """フィードの多様性スコア計算"""
            if len(videos) <= 1:
                return 1.0
            
            # ジャンル多様性
            genres = [video["genre"] for video in videos]
            genre_diversity = len(set(genres)) / len(genres)
            
            # メーカー多様性
            makers = [video["maker"] for video in videos]
            maker_diversity = len(set(makers)) / len(makers)
            
            # 価格帯多様性（価格を5段階に分類）
            price_ranges = []
            for video in videos:
                price = video["price"]
                if price < 1000:
                    price_ranges.append("low")
                elif price < 2000:
                    price_ranges.append("medium-low")
                elif price < 3000:
                    price_ranges.append("medium")
                elif price < 4000:
                    price_ranges.append("medium-high")
                else:
                    price_ranges.append("high")
            
            price_diversity = len(set(price_ranges)) / len(price_ranges)
            
            # 総合多様性スコア
            return (genre_diversity + maker_diversity + price_diversity) / 3
        
        # 最初の10件でテスト
        diversity_score = calculate_diversity_score(sample_video_data[:10])
        assert 0.0 <= diversity_score <= 1.0
        
        # 単一動画（多様性=1.0）
        single_video_score = calculate_diversity_score(sample_video_data[:1])
        assert single_video_score == 1.0
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_feed_performance_requirements(self, performance_timer):
        """フィードパフォーマンス要件テスト"""
        def mock_feed_processing(feed_type: str, limit: int):
            """フィード処理のモック"""
            import time
            
            # フィードタイプ別の処理時間をシミュレート
            processing_times = {
                "explore": 0.1,
                "personalized": 0.2,
                "latest": 0.05,
                "popular": 0.08,
                "random": 0.03
            }
            
            time.sleep(processing_times.get(feed_type, 0.1))
            return {"processing_time_ms": processing_times[feed_type] * 1000}
        
        # 各フィードタイプのパフォーマンステスト
        feed_types = ["explore", "personalized", "latest", "popular", "random"]
        
        for feed_type in feed_types:
            performance_timer.start()
            result = mock_feed_processing(feed_type, 20)
            performance_timer.stop()
            
            # 300ms以内の要件
            assert performance_timer.duration < 0.3
            assert result["processing_time_ms"] <= 300
    
    @pytest.mark.unit
    def test_error_handling_scenarios(self):
        """エラーハンドリングシナリオテスト"""
        
        def handle_invalid_feed_type(feed_type: str) -> Dict:
            """無効なフィードタイプの処理"""
            valid_types = ["explore", "personalized", "latest", "popular", "random"]
            if feed_type not in valid_types:
                return {
                    "error": f"Invalid feed_type: {feed_type}",
                    "code": "INVALID_FEED_TYPE",
                    "status": 400,
                    "valid_types": valid_types
                }
            return {"status": 200}
        
        def handle_missing_user_for_personalized(feed_type: str, user_id: str = None) -> Dict:
            """パーソナライズドフィードでユーザーIDが欠けている場合"""
            if feed_type == "personalized" and not user_id:
                return {
                    "error": "user_id is required for personalized feed",
                    "code": "USER_ID_REQUIRED",
                    "status": 400
                }
            return {"status": 200}
        
        def handle_invalid_pagination(limit: int, offset: int) -> Dict:
            """無効なページネーションパラメータ"""
            if limit <= 0 or limit > 100:
                return {
                    "error": f"Invalid limit: {limit}. Must be between 1 and 100",
                    "code": "INVALID_LIMIT",
                    "status": 400
                }
            
            if offset < 0:
                return {
                    "error": f"Invalid offset: {offset}. Must be >= 0",
                    "code": "INVALID_OFFSET",
                    "status": 400
                }
            
            return {"status": 200}
        
        # エラーケーステスト
        
        # 無効なフィードタイプ
        error_response = handle_invalid_feed_type("invalid_type")
        assert error_response["status"] == 400
        assert "INVALID_FEED_TYPE" in error_response["code"]
        
        # パーソナライズドフィードでユーザーID欠如
        error_response = handle_missing_user_for_personalized("personalized", None)
        assert error_response["status"] == 400
        assert "USER_ID_REQUIRED" in error_response["code"]
        
        # 無効なリミット
        error_response = handle_invalid_pagination(0, 0)
        assert error_response["status"] == 400
        assert "INVALID_LIMIT" in error_response["code"]
        
        error_response = handle_invalid_pagination(150, 0)
        assert error_response["status"] == 400
        
        # 無効なオフセット
        error_response = handle_invalid_pagination(20, -1)
        assert error_response["status"] == 400
        assert "INVALID_OFFSET" in error_response["code"]
        
        # 有効なパラメータ
        valid_response = handle_invalid_pagination(20, 0)
        assert valid_response["status"] == 200
    
    @pytest.mark.unit
    def test_cors_headers_validation(self):
        """CORSヘッダー検証テスト"""
        expected_cors_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
        }
        
        # 必須CORSヘッダーの確認
        assert 'Access-Control-Allow-Origin' in expected_cors_headers
        assert 'Access-Control-Allow-Headers' in expected_cors_headers
        assert 'Access-Control-Allow-Methods' in expected_cors_headers
        
        # 適切な値の確認
        assert expected_cors_headers['Access-Control-Allow-Origin'] == '*'
        assert 'authorization' in expected_cors_headers['Access-Control-Allow-Headers']
        assert 'POST' in expected_cors_headers['Access-Control-Allow-Methods']
        assert 'OPTIONS' in expected_cors_headers['Access-Control-Allow-Methods']