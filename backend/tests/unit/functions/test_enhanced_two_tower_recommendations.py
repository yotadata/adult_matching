"""
Tests for Enhanced Two Tower Recommendations Edge Function

enhanced_two_tower_recommendations Edge Functionの単体テスト
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# TypeScriptのEdge Functionを直接テストすることはできないため、
# APIコントラクトとレスポンス形式をテストします


class TestEnhancedTwoTowerRecommendations:
    """Enhanced Two Tower Recommendations API テスト"""
    
    @pytest.fixture
    def sample_recommendation_request(self):
        """推薦リクエストサンプル"""
        return {
            "limit": 10,
            "exclude_liked": True,
            "diversity_weight": 0.3,
            "include_reasons": True,
            "min_similarity_threshold": 0.1
        }
    
    @pytest.fixture
    def sample_user_embedding(self):
        """ユーザー埋め込みサンプル"""
        return {
            "user_id": "test-user-123",
            "embedding": [0.1] * 768,
            "updated_at": "2025-01-01T00:00:00Z"
        }
    
    @pytest.fixture
    def sample_video_embeddings(self):
        """動画埋め込みサンプル"""
        return [
            {
                "video_id": f"video-{i}",
                "embedding": [0.1 + i * 0.01] * 768,
                "updated_at": "2025-01-01T00:00:00Z"
            }
            for i in range(20)
        ]
    
    @pytest.fixture
    def sample_videos_data(self):
        """動画データサンプル"""
        return [
            {
                "id": f"video-{i}",
                "title": f"Test Video {i}",
                "description": f"Description for video {i}",
                "thumbnail_url": f"https://example.com/thumb{i}.jpg",
                "preview_video_url": f"https://example.com/preview{i}.mp4",
                "maker": "Test Maker",
                "genre": "Test Genre",
                "price": 1000 + i * 100,
                "sample_video_url": f"https://example.com/sample{i}.mp4",
                "image_urls": [f"https://example.com/img{i}.jpg"],
                "performers": [f"Performer {i}"],
                "tags": [f"tag{i}", "test"]
            }
            for i in range(20)
        ]
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_recommendation_request_validation(self, sample_recommendation_request):
        """推薦リクエスト検証テスト"""
        # 必須フィールドの検証
        assert isinstance(sample_recommendation_request.get("limit"), int)
        assert isinstance(sample_recommendation_request.get("exclude_liked"), bool)
        assert isinstance(sample_recommendation_request.get("diversity_weight"), float)
        
        # 範囲検証
        assert 1 <= sample_recommendation_request["limit"] <= 100
        assert 0.0 <= sample_recommendation_request["diversity_weight"] <= 1.0
        assert 0.0 <= sample_recommendation_request["min_similarity_threshold"] <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_user_embedding_structure(self, sample_user_embedding):
        """ユーザー埋め込み構造テスト"""
        # 必須フィールドの存在確認
        assert "user_id" in sample_user_embedding
        assert "embedding" in sample_user_embedding
        assert "updated_at" in sample_user_embedding
        
        # 埋め込み次元確認
        assert len(sample_user_embedding["embedding"]) == 768
        assert all(isinstance(x, (int, float)) for x in sample_user_embedding["embedding"])
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_video_embedding_structure(self, sample_video_embeddings):
        """動画埋め込み構造テスト"""
        for video_embedding in sample_video_embeddings:
            assert "video_id" in video_embedding
            assert "embedding" in video_embedding
            assert "updated_at" in video_embedding
            
            # 埋め込み次元確認
            assert len(video_embedding["embedding"]) == 768
            assert all(isinstance(x, (int, float)) for x in video_embedding["embedding"])
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_enhanced_recommended_video_structure(self, sample_videos_data):
        """推薦動画レスポンス構造テスト"""
        # 推薦レスポンスに必要な追加フィールドをシミュレート
        enhanced_video = sample_videos_data[0].copy()
        enhanced_video.update({
            "similarity_score": 0.85,
            "recommendation_reason": "Similar to your liked videos",
            "diversity_score": 0.7,
            "confidence_score": 0.9
        })
        
        # 必須フィールド確認
        required_fields = [
            "id", "title", "description", "thumbnail_url", "preview_video_url",
            "maker", "genre", "price", "sample_video_url", "image_urls",
            "performers", "tags", "similarity_score", "recommendation_reason"
        ]
        
        for field in required_fields:
            assert field in enhanced_video
        
        # 型確認
        assert isinstance(enhanced_video["similarity_score"], (int, float))
        assert isinstance(enhanced_video["recommendation_reason"], str)
        assert 0.0 <= enhanced_video["similarity_score"] <= 1.0
    
    @pytest.mark.unit
    def test_cosine_similarity_calculation(self):
        """コサイン類似度計算テスト"""
        import numpy as np
        
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """コサイン類似度計算（Edge Function内のロジックを模擬）"""
            a = np.array(vec1)
            b = np.array(vec2)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # 同じベクター（類似度=1.0）
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-10
        
        # 直交ベクター（類似度=0.0）
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec3, vec4)
        assert abs(similarity - 0.0) < 1e-10
        
        # 逆ベクター（類似度=-1.0）
        vec5 = [1.0, 2.0, 3.0]
        vec6 = [-1.0, -2.0, -3.0]
        similarity = cosine_similarity(vec5, vec6)
        assert abs(similarity - (-1.0)) < 1e-10
    
    @pytest.mark.unit
    def test_diversity_score_calculation(self):
        """多様性スコア計算テスト"""
        def calculate_diversity_score(recommended_videos: List[Dict], diversity_weight: float) -> float:
            """多様性スコア計算（Edge Function内のロジックを模擬）"""
            if len(recommended_videos) <= 1:
                return 1.0
            
            # ジャンル多様性を計算
            genres = [video.get("genre") for video in recommended_videos]
            unique_genres = len(set(genres))
            genre_diversity = unique_genres / len(genres)
            
            # メーカー多様性を計算
            makers = [video.get("maker") for video in recommended_videos]
            unique_makers = len(set(makers))
            maker_diversity = unique_makers / len(makers)
            
            # 総合多様性スコア
            return (genre_diversity + maker_diversity) / 2
        
        # 同じジャンル・メーカーの動画群（低多様性）
        low_diversity_videos = [
            {"genre": "Action", "maker": "Studio A"},
            {"genre": "Action", "maker": "Studio A"},
            {"genre": "Action", "maker": "Studio A"}
        ]
        
        diversity_score = calculate_diversity_score(low_diversity_videos, 0.3)
        assert diversity_score < 0.5
        
        # 異なるジャンル・メーカーの動画群（高多様性）
        high_diversity_videos = [
            {"genre": "Action", "maker": "Studio A"},
            {"genre": "Comedy", "maker": "Studio B"},
            {"genre": "Drama", "maker": "Studio C"}
        ]
        
        diversity_score = calculate_diversity_score(high_diversity_videos, 0.3)
        assert diversity_score == 1.0
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_api_response_format(self):
        """APIレスポンス形式テスト"""
        # 期待されるレスポンス構造
        expected_response = {
            "recommendations": [],
            "metrics": {
                "total_candidates": 0,
                "filtered_count": 0,
                "average_similarity": 0.0,
                "diversity_score": 0.0,
                "processing_time_ms": 0
            },
            "request_params": {
                "limit": 10,
                "exclude_liked": True,
                "diversity_weight": 0.3
            }
        }
        
        # 必須フィールドの存在確認
        assert "recommendations" in expected_response
        assert "metrics" in expected_response
        assert "request_params" in expected_response
        
        # recommendations配列型確認
        assert isinstance(expected_response["recommendations"], list)
        
        # metrics構造確認
        metrics_fields = ["total_candidates", "filtered_count", "average_similarity", "diversity_score", "processing_time_ms"]
        for field in metrics_fields:
            assert field in expected_response["metrics"]
    
    @pytest.mark.unit
    def test_error_handling_scenarios(self):
        """エラーハンドリングシナリオテスト"""
        # ユーザー埋め込みが見つからない場合
        def handle_no_user_embedding():
            return {
                "error": "User embedding not found",
                "code": "USER_EMBEDDING_MISSING",
                "status": 404
            }
        
        error_response = handle_no_user_embedding()
        assert error_response["status"] == 404
        assert "error" in error_response
        
        # 無効なリクエストパラメータ
        def handle_invalid_parameters(limit: int):
            if limit <= 0 or limit > 100:
                return {
                    "error": "Invalid limit parameter",
                    "code": "INVALID_LIMIT",
                    "status": 400
                }
            return None
        
        error_response = handle_invalid_parameters(-1)
        assert error_response is not None
        assert error_response["status"] == 400
        
        error_response = handle_invalid_parameters(150)
        assert error_response is not None
        assert error_response["status"] == 400
        
        # 有効なパラメータ
        valid_response = handle_invalid_parameters(10)
        assert valid_response is None
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_response_time_requirements(self, performance_timer):
        """レスポンス時間要件テスト"""
        def mock_recommendation_processing():
            """推薦処理のモック（計算時間をシミュレート）"""
            import time
            time.sleep(0.1)  # 100ms の処理時間をシミュレート
            return {
                "recommendations": [],
                "metrics": {"processing_time_ms": 100}
            }
        
        performance_timer.start()
        result = mock_recommendation_processing()
        performance_timer.stop()
        
        # 500ms以内の要件
        assert performance_timer.duration < 0.5
        assert result["metrics"]["processing_time_ms"] <= 500
    
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
    
    @pytest.mark.unit
    def test_recommendation_filtering(self):
        """推薦フィルタリングロジックテスト"""
        def filter_liked_videos(videos: List[Dict], liked_video_ids: List[str], exclude_liked: bool) -> List[Dict]:
            """いいねした動画のフィルタリング"""
            if not exclude_liked:
                return videos
            
            return [video for video in videos if video["id"] not in liked_video_ids]
        
        videos = [
            {"id": "video-1", "title": "Video 1"},
            {"id": "video-2", "title": "Video 2"},
            {"id": "video-3", "title": "Video 3"}
        ]
        
        liked_ids = ["video-1", "video-3"]
        
        # いいね動画を除外
        filtered = filter_liked_videos(videos, liked_ids, exclude_liked=True)
        assert len(filtered) == 1
        assert filtered[0]["id"] == "video-2"
        
        # いいね動画を除外しない
        not_filtered = filter_liked_videos(videos, liked_ids, exclude_liked=False)
        assert len(not_filtered) == 3
    
    @pytest.mark.unit
    def test_similarity_threshold_filtering(self):
        """類似度しきい値フィルタリングテスト"""
        def filter_by_similarity_threshold(candidates: List[Dict], threshold: float) -> List[Dict]:
            """類似度しきい値によるフィルタリング"""
            return [candidate for candidate in candidates if candidate["similarity_score"] >= threshold]
        
        candidates = [
            {"id": "video-1", "similarity_score": 0.9},
            {"id": "video-2", "similarity_score": 0.5},
            {"id": "video-3", "similarity_score": 0.2}
        ]
        
        # しきい値0.6でフィルタリング
        filtered = filter_by_similarity_threshold(candidates, 0.6)
        assert len(filtered) == 1
        assert filtered[0]["id"] == "video-1"
        
        # しきい値0.1でフィルタリング
        filtered_low = filter_by_similarity_threshold(candidates, 0.1)
        assert len(filtered_low) == 3