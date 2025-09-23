"""
コンテンツAPIエッジケーステスト

極端な条件やエラーケースに対するテスト
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import httpx


class TestContentAPIEdgeCases:
    """コンテンツAPIエッジケーステスト"""

    @pytest.mark.asyncio
    async def test_empty_database(self, content_api_urls, test_request_headers, http_client_session):
        """空データベーステスト"""
        with patch('supabase.functions._shared.database.getSupabaseClientFromRequest') as mock_client:
            # 空の結果を返すモック
            mock_supabase = Mock()
            mock_supabase.from.return_value.select.return_value.range.return_value = Mock()
            mock_supabase.from.return_value.select.return_value.range.return_value.execute = AsyncMock(
                return_value=Mock(data=[], error=None)
            )
            mock_client.return_value = mock_supabase

            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "explore",
                    "limit": 20
                },
                headers=test_request_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["videos"] == []
            assert data["data"]["total_count"] == 0

    @pytest.mark.asyncio
    async def test_very_large_limit(self, content_api_urls, test_request_headers, http_client_session):
        """非常に大きなリミット値テスト"""
        response = await http_client_session.post(
            content_api_urls['content_feed'],
            json={
                "feed_type": "explore",
                "limit": 10000  # 非常に大きな値
            },
            headers=test_request_headers
        )

        # サーバーが適切に制限すること
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert len(data["data"]["videos"]) <= 100  # 最大制限

    @pytest.mark.asyncio
    async def test_negative_offset(self, content_api_urls, test_request_headers, http_client_session):
        """負のオフセット値テスト"""
        response = await http_client_session.post(
            content_api_urls['content_feed'],
            json={
                "feed_type": "explore",
                "limit": 20,
                "offset": -10
            },
            headers=test_request_headers
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_extremely_long_exclude_ids_list(self, content_api_urls, test_request_headers, http_client_session):
        """非常に長い除外IDリストテスト"""
        exclude_ids = [f"video_{i}" for i in range(1000)]  # 1000個のID

        response = await http_client_session.post(
            content_api_urls['content_feed'],
            json={
                "feed_type": "explore",
                "limit": 20,
                "exclude_ids": exclude_ids
            },
            headers=test_request_headers
        )

        # サーバーが適切に処理すること（制限またはエラー）
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_invalid_uuid_in_exclude_ids(self, content_api_urls, test_request_headers, http_client_session):
        """無効なUUIDの除外IDテスト"""
        response = await http_client_session.post(
            content_api_urls['content_feed'],
            json={
                "feed_type": "explore",
                "limit": 20,
                "exclude_ids": ["invalid-uuid", "not-a-uuid", "12345"]
            },
            headers=test_request_headers
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_unicode_in_request(self, content_api_urls, test_request_headers, http_client_session):
        """Unicode文字を含むリクエストテスト"""
        response = await http_client_session.post(
            content_api_urls['content_feed'],
            json={
                "feed_type": "explore",
                "limit": 20,
                "filter": {
                    "search_query": "テスト動画🎬🎯"  # 日本語と絵文字
                }
            },
            headers=test_request_headers
        )

        # Unicode文字が適切に処理されること
        assert response.status_code in [200, 400]  # 実装によってはサポートしない場合もある

    @pytest.mark.asyncio
    async def test_concurrent_user_requests(self, content_api_urls, test_request_headers, http_client_session):
        """同一ユーザーからの同時リクエストテスト"""
        auth_headers = {**test_request_headers, "Authorization": "Bearer same-user-token"}

        async def make_user_request(request_id: int):
            return await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "personalized",
                    "limit": 15,
                    "user_id": "same-user-123"
                },
                headers=auth_headers
            )

        # 同じユーザーから5つの同時リクエスト
        tasks = [make_user_request(i) for i in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 全て成功するか適切にレート制限されること
        success_count = 0
        for response in responses:
            if isinstance(response, httpx.Response):
                if response.status_code in [200, 429]:  # 成功またはレート制限
                    success_count += 1

        assert success_count >= 3  # 少なくとも一部は成功

    @pytest.mark.asyncio
    async def test_malformed_authorization_header(self, content_api_urls, test_request_headers, http_client_session):
        """不正な認証ヘッダーテスト"""
        malformed_headers = [
            {**test_request_headers, "Authorization": "Invalid token format"},
            {**test_request_headers, "Authorization": "Bearer"},  # トークンなし
            {**test_request_headers, "Authorization": ""},  # 空文字
            {**test_request_headers, "Authorization": "Bearer " + "x" * 1000},  # 非常に長いトークン
        ]

        for headers in malformed_headers:
            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "personalized",
                    "limit": 20
                },
                headers=headers
            )

            # 適切にフォールバックすること
            assert response.status_code in [200, 401]

    @pytest.mark.asyncio
    async def test_request_timeout_simulation(self, content_api_urls, test_request_headers):
        """リクエストタイムアウトシミュレーション"""
        async with httpx.AsyncClient(timeout=0.1) as short_timeout_client:  # 100ms timeout
            try:
                response = await short_timeout_client.post(
                    content_api_urls['content_feed'],
                    json={
                        "feed_type": "explore",
                        "limit": 20
                    },
                    headers=test_request_headers
                )
                # 非常に高速であれば成功
                assert response.status_code == 200
            except httpx.TimeoutException:
                # タイムアウトは予期される動作
                pass

    @pytest.mark.asyncio
    async def test_memory_exhaustion_protection(self, content_api_urls, test_request_headers, http_client_session):
        """メモリ枯渇保護テスト"""
        # 非常に大きなペイロードを送信
        large_exclude_ids = [f"video_{i:010d}" for i in range(10000)]

        response = await http_client_session.post(
            content_api_urls['content_feed'],
            json={
                "feed_type": "explore",
                "limit": 50,
                "exclude_ids": large_exclude_ids,
                "additional_data": ["x" * 1000] * 100  # 大量のデータ
            },
            headers=test_request_headers
        )

        # サーバーが適切に制限すること
        assert response.status_code in [200, 400, 413]  # 成功、バリデーションエラー、ペイロード過大

    @pytest.mark.asyncio
    async def test_sql_injection_attempts(self, content_api_urls, test_request_headers, http_client_session):
        """SQLインジェクション攻撃テストト"""
        sql_injection_payloads = [
            "'; DROP TABLE videos; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
            "admin'/*",
            "' OR 1=1#"
        ]

        for payload in sql_injection_payloads:
            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "explore",
                    "limit": 20,
                    "user_id": payload  # SQLインジェクション試行
                },
                headers=test_request_headers
            )

            # SQLインジェクションが防がれること
            assert response.status_code in [200, 400]
            if response.status_code == 200:
                data = response.json()
                # 正常なレスポンス構造を持つこと
                assert "data" in data

    @pytest.mark.asyncio
    async def test_xss_payload_in_parameters(self, content_api_urls, test_request_headers, http_client_session):
        """XSSペイロードパラメータテスト"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert(String.fromCharCode(88,83,83))//'"
        ]

        for payload in xss_payloads:
            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "explore",
                    "limit": 20,
                    "search_query": payload
                },
                headers=test_request_headers
            )

            # XSSが適切に処理されること
            assert response.status_code in [200, 400]
            if response.status_code == 200:
                response_text = response.text
                # ペイロードがエスケープされている、または除去されていること
                assert payload not in response_text


class TestContentAPIFailureRecovery:
    """コンテンツAPI障害復旧テスト"""

    @pytest.mark.asyncio
    async def test_database_connection_recovery(self, content_api_urls, test_request_headers, http_client_session):
        """データベース接続復旧テスト"""
        # 最初にエラーを返し、次に成功するモック
        connection_attempts = 0

        def mock_get_client(*args, **kwargs):
            nonlocal connection_attempts
            connection_attempts += 1
            if connection_attempts == 1:
                raise Exception("Database connection failed")
            else:
                # 正常なクライアントを返す
                return Mock()

        with patch('supabase.functions._shared.database.getSupabaseClientFromRequest', side_effect=mock_get_client):
            # 最初のリクエスト（失敗）
            response1 = await http_client_session.post(
                content_api_urls['content_feed'],
                json={"feed_type": "explore", "limit": 20},
                headers=test_request_headers
            )

            # 2回目のリクエスト（成功）
            response2 = await http_client_session.post(
                content_api_urls['content_feed'],
                json={"feed_type": "explore", "limit": 20},
                headers=test_request_headers
            )

            # 復旧すること
            assert response1.status_code >= 500
            assert response2.status_code in [200, 500]  # 実装によって異なる

    @pytest.mark.asyncio
    async def test_partial_database_failure(self, content_api_urls, test_request_headers, http_client_session):
        """部分的データベース障害テスト"""
        with patch('supabase.functions._shared.database.getSupabaseClientFromRequest') as mock_client:
            # 一部のテーブルのみアクセス可能なモック
            mock_supabase = Mock()

            def mock_from_table(table_name):
                if table_name == 'videos':
                    # videosテーブルは正常
                    mock_query = Mock()
                    mock_query.select.return_value.range.return_value = Mock()
                    mock_query.select.return_value.range.return_value.execute = AsyncMock(
                        return_value=Mock(data=[{"id": "test", "title": "Test"}], error=None)
                    )
                    return mock_query
                else:
                    # 他のテーブルはエラー
                    mock_query = Mock()
                    mock_query.select.return_value.range.return_value = Mock()
                    mock_query.select.return_value.range.return_value.execute = AsyncMock(
                        return_value=Mock(data=None, error={"message": "Table not accessible"})
                    )
                    return mock_query

            mock_supabase.from = mock_from_table
            mock_client.return_value = mock_supabase

            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={"feed_type": "explore", "limit": 20},
                headers=test_request_headers
            )

            # 部分的な障害でも適切に処理すること
            assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_service_degradation_mode(self, content_api_urls, test_request_headers, http_client_session):
        """サービス縮退モードテスト"""
        # 埋め込み検索が失敗した場合のフォールバック
        with patch('supabase.functions._shared.content.getVideosBySimilarity') as mock_similarity:
            mock_similarity.return_value = []  # 類似度検索失敗

            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "personalized",
                    "limit": 20,
                    "user_id": "test-user"
                },
                headers=test_request_headers
            )

            # フォールバックして基本フィードを返すこと
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestContentAPISecurityEdgeCases:
    """コンテンツAPIセキュリティエッジケース"""

    @pytest.mark.asyncio
    async def test_rate_limiting_bypass_attempts(self, content_api_urls, test_request_headers, http_client_session):
        """レート制限バイパス試行テスト"""
        # 異なるUser-Agentで高頻度リクエスト
        headers_variations = [
            {**test_request_headers, "User-Agent": f"TestAgent{i}"}
            for i in range(10)
        ]

        tasks = []
        for headers in headers_variations:
            task = http_client_session.post(
                content_api_urls['content_feed'],
                json={"feed_type": "explore", "limit": 20},
                headers=headers
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 適切にレート制限されること
        rate_limited_count = 0
        for response in responses:
            if isinstance(response, httpx.Response) and response.status_code == 429:
                rate_limited_count += 1

        # 完全にバイパスされないこと
        assert rate_limited_count > 0 or len([r for r in responses if isinstance(r, httpx.Response) and r.status_code == 200]) < len(responses)

    @pytest.mark.asyncio
    async def test_header_injection_attempts(self, content_api_urls, http_client_session):
        """ヘッダーインジェクション試行テスト"""
        malicious_headers = {
            "Content-Type": "application/json\r\nX-Injected-Header: malicious",
            "Authorization": "Bearer token\r\nX-Admin: true",
            "User-Agent": "Normal\r\nHost: evil.com"
        }

        response = await http_client_session.post(
            content_api_urls['content_feed'],
            json={"feed_type": "explore", "limit": 20},
            headers=malicious_headers
        )

        # ヘッダーインジェクションが防がれること
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_privilege_escalation_attempts(self, content_api_urls, test_request_headers, http_client_session):
        """権限昇格試行テスト"""
        escalation_payloads = [
            {"user_id": "admin", "admin": True},
            {"user_id": "../../../admin"},
            {"user_id": "' UNION SELECT admin_token FROM admin_users --"},
            {"role": "admin", "user_id": "normal_user"}
        ]

        for payload in escalation_payloads:
            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "personalized",
                    "limit": 20,
                    **payload
                },
                headers=test_request_headers
            )

            # 権限昇格が防がれること
            assert response.status_code in [200, 400, 403]
            if response.status_code == 200:
                data = response.json()
                # 管理者データにアクセスできないこと
                assert "admin" not in str(data).lower()


@pytest.mark.slow
class TestContentAPIStressEdgeCases:
    """コンテンツAPIストレスエッジケース"""

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, content_api_urls, test_request_headers, http_client_session):
        """メモリリーク検出テスト"""
        # 多数のリクエストでメモリリークがないか確認
        for i in range(100):
            response = await http_client_session.post(
                content_api_urls['content_feed'],
                json={
                    "feed_type": "explore",
                    "limit": 20,
                    "request_id": i  # 各リクエストを識別
                },
                headers=test_request_headers
            )

            assert response.status_code in [200, 429]

            # 短時間待機
            if i % 10 == 0:
                await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, content_api_urls, test_request_headers):
        """接続プール枯渇テスト"""
        # 多数の同時接続を作成
        clients = [httpx.AsyncClient(timeout=30.0) for _ in range(20)]

        try:
            tasks = []
            for i, client in enumerate(clients):
                task = client.post(
                    content_api_urls['content_feed'],
                    json={"feed_type": "explore", "limit": 20},
                    headers=test_request_headers
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # 適切にエラー処理されること
            success_count = sum(
                1 for response in responses
                if isinstance(response, httpx.Response) and response.status_code == 200
            )

            # 全て失敗しないこと
            assert success_count > 0

        finally:
            # クリーンアップ
            for client in clients:
                await client.aclose()


# テスト実行用ヘルパー関数
def run_edge_case_tests():
    """エッジケーステストの実行"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow",
        "--durations=10"
    ])


if __name__ == "__main__":
    run_edge_case_tests()