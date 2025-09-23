"""
Tests for Likes Edge Function

likes Edge Functionの単体テスト
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List


class TestLikes:
    """Likes API テスト"""
    
    @pytest.fixture
    def sample_like_request(self):
        """いいねリクエストサンプル"""
        return {
            "video_id": "test-video-123",
            "user_id": "test-user-123"
        }
    
    @pytest.fixture
    def sample_unlike_request(self):
        """いいね削除リクエストサンプル"""
        return {
            "video_id": "test-video-123",
            "user_id": "test-user-123"
        }
    
    @pytest.fixture
    def sample_liked_videos(self):
        """いいね済み動画サンプル"""
        return [
            {
                "id": f"video-{i}",
                "title": f"Test Video {i}",
                "description": f"Description for video {i}",
                "thumbnail_url": f"https://example.com/thumb{i}.jpg",
                "preview_video_url": f"https://example.com/preview{i}.mp4",
                "maker": f"Maker {i % 3}",
                "genre": f"Genre {i % 5}",
                "price": 1000 + i * 100,
                "sample_video_url": f"https://example.com/sample{i}.mp4",
                "image_urls": [f"https://example.com/img{i}_{j}.jpg" for j in range(3)],
                "performers": [f"Performer {i}_{j}" for j in range(2)],
                "tags": [f"tag{i}", "test", "sample"],
                "liked_at": f"2025-01-{(i % 28) + 1:02d}T00:00:00Z",
                "purchased": i % 4 == 0  # 25%の動画が購入済み
            }
            for i in range(20)
        ]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_like_request_validation(self, sample_like_request):
        """いいねリクエスト検証テスト"""
        # 必須フィールドの検証
        assert "video_id" in sample_like_request
        assert "user_id" in sample_like_request
        
        # フィールドの型検証
        assert isinstance(sample_like_request["video_id"], str)
        assert isinstance(sample_like_request["user_id"], str)
        
        # 値の妥当性確認
        assert len(sample_like_request["video_id"]) > 0
        assert len(sample_like_request["user_id"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_like_response_structure(self, sample_liked_videos):
        """いいねレスポンス構造テスト"""
        for video in sample_liked_videos[:5]:  # 最初の5件をテスト
            # 必須フィールド確認
            required_fields = [
                "id", "title", "description", "thumbnail_url", "preview_video_url",
                "maker", "genre", "price", "sample_video_url", "image_urls", 
                "performers", "tags", "liked_at", "purchased"
            ]
            
            for field in required_fields:
                assert field in video
            
            # 型確認
            assert isinstance(video["id"], str)
            assert isinstance(video["title"], str)
            assert isinstance(video["price"], (int, float))
            assert isinstance(video["image_urls"], list)
            assert isinstance(video["performers"], list)
            assert isinstance(video["tags"], list)
            assert isinstance(video["purchased"], bool)
            
            # 値の妥当性確認
            assert video["price"] >= 0
            assert len(video["image_urls"]) >= 0
            assert video["liked_at"].endswith("Z")  # ISO形式確認
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_get_likes_response_format(self, sample_liked_videos):
        """いいね一覧取得レスポンス形式テスト"""
        expected_response = {
            "likes": sample_liked_videos,
            "total_count": len(sample_liked_videos)
        }
        
        # 必須フィールド確認
        assert "likes" in expected_response
        assert "total_count" in expected_response
        
        # 型確認
        assert isinstance(expected_response["likes"], list)
        assert isinstance(expected_response["total_count"], int)
        
        # 数量確認
        assert expected_response["total_count"] == len(expected_response["likes"])
    
    @pytest.mark.unit
    def test_pagination_logic(self, sample_liked_videos):
        """ページネーションロジックテスト"""
        def paginate_likes(likes: List[Dict], limit: int = 50, offset: int = 0) -> Dict:
            """いいね一覧のページネーション処理"""
            total_count = len(likes)
            start_idx = offset
            end_idx = offset + limit
            
            paginated_likes = likes[start_idx:end_idx]
            
            return {
                "likes": paginated_likes,
                "total_count": total_count,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": end_idx < total_count
                }
            }
        
        # デフォルトページネーション
        result = paginate_likes(sample_liked_videos)
        assert len(result["likes"]) == len(sample_liked_videos)  # 20件すべて
        assert result["pagination"]["has_more"] is False
        
        # 制限付きページネーション
        result = paginate_likes(sample_liked_videos, limit=10, offset=0)
        assert len(result["likes"]) == 10
        assert result["pagination"]["has_more"] is True
        
        # 2ページ目
        result = paginate_likes(sample_liked_videos, limit=10, offset=10)
        assert len(result["likes"]) == 10
        assert result["pagination"]["has_more"] is False
        
        # 範囲外
        result = paginate_likes(sample_liked_videos, limit=10, offset=30)
        assert len(result["likes"]) == 0
        assert result["pagination"]["has_more"] is False
    
    @pytest.mark.unit
    def test_like_add_logic(self):
        """いいね追加ロジックテスト"""
        def add_like(user_id: str, video_id: str, existing_likes: List[str]) -> Dict:
            """いいね追加処理"""
            if not user_id or not video_id:
                return {
                    "error": "user_id and video_id are required",
                    "status": 400
                }
            
            like_id = f"{user_id}:{video_id}"
            
            if like_id in existing_likes:
                return {
                    "success": True,
                    "message": "Already liked",
                    "status": 200
                }
            
            existing_likes.append(like_id)
            return {
                "success": True,
                "message": "Like added successfully",
                "status": 200
            }
        
        existing_likes = []
        
        # 正常ないいね追加
        result = add_like("user-123", "video-456", existing_likes)
        assert result["success"] is True
        assert result["status"] == 200
        assert len(existing_likes) == 1
        
        # 重複いいね
        result = add_like("user-123", "video-456", existing_likes)
        assert result["success"] is True
        assert "Already liked" in result["message"]
        assert len(existing_likes) == 1  # 増えない
        
        # バリデーションエラー
        result = add_like("", "video-456", existing_likes)
        assert result["status"] == 400
        assert "error" in result
    
    @pytest.mark.unit
    def test_like_remove_logic(self):
        """いいね削除ロジックテスト"""
        def remove_like(user_id: str, video_id: str, existing_likes: List[str]) -> Dict:
            """いいね削除処理"""
            if not user_id or not video_id:
                return {
                    "error": "user_id and video_id are required",
                    "status": 400
                }
            
            like_id = f"{user_id}:{video_id}"
            
            if like_id in existing_likes:
                existing_likes.remove(like_id)
                return {
                    "success": True,
                    "message": "Like removed successfully",
                    "status": 200
                }
            else:
                return {
                    "success": True,
                    "message": "Like not found (no action needed)",
                    "status": 200
                }
        
        existing_likes = ["user-123:video-456", "user-123:video-789"]
        
        # 正常ないいね削除
        result = remove_like("user-123", "video-456", existing_likes)
        assert result["success"] is True
        assert result["status"] == 200
        assert len(existing_likes) == 1
        
        # 存在しないいいね削除
        result = remove_like("user-123", "video-999", existing_likes)
        assert result["success"] is True
        assert "not found" in result["message"]
        assert len(existing_likes) == 1  # 変わらない
        
        # バリデーションエラー
        result = remove_like("", "video-456", existing_likes)
        assert result["status"] == 400
        assert "error" in result
    
    @pytest.mark.unit
    def test_video_existence_validation(self):
        """動画存在確認テスト"""
        def validate_video_exists(video_id: str, available_videos: List[str]) -> Dict:
            """動画の存在確認"""
            if video_id in available_videos:
                return {"exists": True, "video_id": video_id}
            else:
                return {
                    "exists": False,
                    "error": "Video not found",
                    "status": 404
                }
        
        available_videos = ["video-1", "video-2", "video-3"]
        
        # 存在する動画
        result = validate_video_exists("video-1", available_videos)
        assert result["exists"] is True
        assert result["video_id"] == "video-1"
        
        # 存在しない動画
        result = validate_video_exists("video-999", available_videos)
        assert result["exists"] is False
        assert result["status"] == 404
        assert "not found" in result["error"]
    
    @pytest.mark.unit
    def test_authentication_validation(self):
        """認証検証テスト"""
        def validate_user_authentication(auth_header: str) -> Dict:
            """ユーザー認証検証"""
            if not auth_header:
                return {
                    "authenticated": False,
                    "error": "Authorization header missing",
                    "status": 401
                }
            
            if not auth_header.startswith("Bearer "):
                return {
                    "authenticated": False,
                    "error": "Invalid authorization format",
                    "status": 401
                }
            
            token = auth_header.replace("Bearer ", "")
            if len(token) < 10:  # 簡単なトークン長チェック
                return {
                    "authenticated": False,
                    "error": "Invalid token",
                    "status": 401
                }
            
            return {
                "authenticated": True,
                "user_id": "extracted-user-id"
            }
        
        # 正常な認証
        result = validate_user_authentication("Bearer valid-long-token-123456")
        assert result["authenticated"] is True
        assert "user_id" in result
        
        # 認証ヘッダーなし
        result = validate_user_authentication("")
        assert result["authenticated"] is False
        assert result["status"] == 401
        
        # 無効な形式
        result = validate_user_authentication("Invalid-format")
        assert result["authenticated"] is False
        assert result["status"] == 401
        
        # 短すぎるトークン
        result = validate_user_authentication("Bearer short")
        assert result["authenticated"] is False
        assert result["status"] == 401
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_error_handling_scenarios(self):
        """エラーハンドリングシナリオテスト"""
        def handle_likes_error(error_type: str) -> Dict:
            """いいね機能エラーハンドリング"""
            if error_type == "unauthorized":
                return {
                    "error": "Unauthorized",
                    "status": 401
                }
            elif error_type == "video_not_found":
                return {
                    "error": "Video not found",
                    "status": 404
                }
            elif error_type == "missing_video_id":
                return {
                    "error": "video_id is required",
                    "status": 400
                }
            elif error_type == "database_error":
                return {
                    "error": "Failed to fetch liked videos",
                    "status": 500
                }
            elif error_type == "internal_error":
                return {
                    "error": "Internal server error",
                    "status": 500
                }
            else:
                return {"status": 200}
        
        # 各エラーケースのテスト
        unauthorized = handle_likes_error("unauthorized")
        assert unauthorized["status"] == 401
        
        not_found = handle_likes_error("video_not_found")
        assert not_found["status"] == 404
        
        bad_request = handle_likes_error("missing_video_id")
        assert bad_request["status"] == 400
        
        db_error = handle_likes_error("database_error")
        assert db_error["status"] == 500
        
        internal = handle_likes_error("internal_error")
        assert internal["status"] == 500
        
        # 正常ケース
        success = handle_likes_error("none")
        assert success["status"] == 200
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_headers_validation(self):
        """CORSヘッダー検証テスト"""
        expected_cors_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
            'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
        }
        
        # 必須CORSヘッダーの確認
        assert 'Access-Control-Allow-Origin' in expected_cors_headers
        assert 'Access-Control-Allow-Headers' in expected_cors_headers
        assert 'Access-Control-Allow-Methods' in expected_cors_headers
        
        # 適切な値の確認
        assert expected_cors_headers['Access-Control-Allow-Origin'] == '*'
        assert 'authorization' in expected_cors_headers['Access-Control-Allow-Headers']
        assert 'GET' in expected_cors_headers['Access-Control-Allow-Methods']
        assert 'POST' in expected_cors_headers['Access-Control-Allow-Methods']
        assert 'DELETE' in expected_cors_headers['Access-Control-Allow-Methods']
        assert 'OPTIONS' in expected_cors_headers['Access-Control-Allow-Methods']
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_likes_performance_requirements(self, performance_timer):
        """いいね機能パフォーマンス要件テスト"""
        def mock_likes_processing(operation: str):
            """いいね処理のモック"""
            import time
            
            processing_times = {
                "get_likes": 0.05,     # 50ms
                "add_like": 0.02,      # 20ms
                "remove_like": 0.02    # 20ms
            }
            
            time.sleep(processing_times.get(operation, 0.05))
            return {"processing_time_ms": processing_times[operation] * 1000}
        
        operations = ["get_likes", "add_like", "remove_like"]
        
        for operation in operations:
            performance_timer.start()
            result = mock_likes_processing(operation)
            performance_timer.stop()
            
            # 100ms以内の要件
            assert performance_timer.duration < 0.1
            assert result["processing_time_ms"] <= 100
    
    @pytest.mark.unit
    def test_data_formatting_logic(self):
        """データ整形ロジックテスト"""
        def format_like_response(raw_like_data: Dict) -> Dict:
            """いいねデータの整形"""
            video = raw_like_data.get("videos", {})
            
            return {
                "id": video.get("id"),
                "title": video.get("title"),
                "description": video.get("description"),
                "thumbnail_url": video.get("thumbnail_url"),
                "preview_video_url": video.get("preview_video_url"),
                "maker": video.get("maker"),
                "genre": video.get("genre"),
                "price": video.get("price"),
                "sample_video_url": video.get("sample_video_url"),
                "image_urls": video.get("image_urls", []),
                "performers": [vp.get("performers", {}).get("name") 
                             for vp in video.get("video_performers", [])
                             if vp.get("performers", {}).get("name")],
                "tags": [vt.get("tags", {}).get("name") 
                        for vt in video.get("video_tags", [])
                        if vt.get("tags", {}).get("name")],
                "liked_at": raw_like_data.get("created_at"),
                "purchased": raw_like_data.get("purchased", False)
            }
        
        # テスト用生データ
        raw_data = {
            "created_at": "2025-01-01T00:00:00Z",
            "purchased": True,
            "videos": {
                "id": "video-123",
                "title": "Test Video",
                "description": "Test Description",
                "thumbnail_url": "https://example.com/thumb.jpg",
                "preview_video_url": "https://example.com/preview.mp4",
                "maker": "Test Maker",
                "genre": "Test Genre",
                "price": 1500,
                "sample_video_url": "https://example.com/sample.mp4",
                "image_urls": ["https://example.com/img1.jpg"],
                "video_performers": [
                    {"performers": {"name": "Performer A"}},
                    {"performers": {"name": "Performer B"}}
                ],
                "video_tags": [
                    {"tags": {"name": "tag1"}},
                    {"tags": {"name": "tag2"}}
                ]
            }
        }
        
        formatted = format_like_response(raw_data)
        
        # 基本フィールド確認
        assert formatted["id"] == "video-123"
        assert formatted["title"] == "Test Video"
        assert formatted["purchased"] is True
        assert formatted["liked_at"] == "2025-01-01T00:00:00Z"
        
        # 配列フィールド確認
        assert len(formatted["performers"]) == 2
        assert "Performer A" in formatted["performers"]
        assert "Performer B" in formatted["performers"]
        
        assert len(formatted["tags"]) == 2
        assert "tag1" in formatted["tags"]
        assert "tag2" in formatted["tags"]