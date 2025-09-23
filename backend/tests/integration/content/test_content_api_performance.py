"""
コンテンツAPIパフォーマンステスト

統合コンテンツAPIのパフォーマンス要件検証
"""

import pytest
import asyncio
import time
import statistics
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch
import httpx
import psutil
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    response_times: List[float]
    success_count: int
    error_count: int
    throughput_rps: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float


class PerformanceTestRunner:
    """パフォーマンステスト実行クラス"""

    def __init__(self, base_url: str = "http://localhost:54321"):
        self.base_url = base_url
        self.functions_url = f"{base_url}/functions/v1"

    async def measure_single_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Tuple[bool, float, int]:
        """単一リクエストの測定"""
        start_time = time.time()

        try:
            response = await client.post(endpoint, json=payload, headers=headers)
            end_time = time.time()

            response_time = (end_time - start_time) * 1000  # ms
            success = response.status_code == 200

            return success, response_time, response.status_code

        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            return False, response_time, 0

    async def run_load_test(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        concurrent_users: int,
        requests_per_user: int,
        ramp_up_seconds: float = 0.0
    ) -> PerformanceMetrics:
        """ロードテスト実行"""
        response_times = []
        success_count = 0
        error_count = 0

        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024

        async def user_session(user_id: int):
            """ユーザーセッション"""
            session_results = []

            # ランプアップ遅延
            if ramp_up_seconds > 0:
                delay = (user_id / concurrent_users) * ramp_up_seconds
                await asyncio.sleep(delay)

            async with httpx.AsyncClient(timeout=30.0) as client:
                for request_id in range(requests_per_user):
                    success, response_time, status_code = await self.measure_single_request(
                        client, endpoint, payload, headers
                    )
                    session_results.append((success, response_time, status_code))

                    # リクエスト間の短い遅延
                    await asyncio.sleep(0.1)

            return session_results

        # 並行ユーザーセッション実行
        tasks = [user_session(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)

        # 結果集計
        for user_results in all_results:
            for success, response_time, status_code in user_results:
                response_times.append(response_time)
                if success:
                    success_count += 1
                else:
                    error_count += 1

        end_time = time.time()
        total_duration = end_time - start_time
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024

        # メトリクス計算
        total_requests = len(response_times)
        throughput_rps = total_requests / total_duration if total_duration > 0 else 0

        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
        p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else 0

        return PerformanceMetrics(
            response_times=response_times,
            success_count=success_count,
            error_count=error_count,
            throughput_rps=throughput_rps,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            memory_usage_mb=memory_end - memory_start,
            cpu_usage_percent=psutil.cpu_percent(interval=1)
        )


@pytest.fixture(scope="session")
def performance_test_runner():
    """パフォーマンステスト実行器"""
    return PerformanceTestRunner()


@pytest.fixture
def performance_requirements():
    """パフォーマンス要件定義"""
    return {
        'max_response_time_ms': 500,
        'p95_response_time_ms': 300,
        'p99_response_time_ms': 800,
        'min_throughput_rps': 50,
        'max_error_rate': 0.05,  # 5%
        'max_memory_increase_mb': 100,
        'max_cpu_usage_percent': 80
    }


class TestContentAPIResponseTime:
    """コンテンツAPIレスポンス時間テスト"""

    @pytest.mark.asyncio
    async def test_explore_feed_response_time(self, performance_test_runner, performance_requirements):
        """探索フィードレスポンス時間テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "explore", "limit": 20}
        headers = {"Content-Type": "application/json"}

        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=1,
            requests_per_user=50
        )

        # レスポンス時間要件確認
        assert metrics.avg_response_time <= performance_requirements['max_response_time_ms']
        assert metrics.p95_response_time <= performance_requirements['p95_response_time_ms']
        assert metrics.success_count >= 45  # 90%以上成功

    @pytest.mark.asyncio
    async def test_personalized_feed_response_time(self, performance_test_runner, performance_requirements):
        """パーソナライズドフィードレスポンス時間テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {
            "feed_type": "personalized",
            "limit": 20,
            "user_id": "test-user-123"
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer mock-jwt-token"
        }

        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=1,
            requests_per_user=30
        )

        # パーソナライズドフィードは計算コストが高いため、少し緩い要件
        assert metrics.avg_response_time <= performance_requirements['max_response_time_ms'] * 1.5
        assert metrics.success_count >= 25  # 80%以上成功

    @pytest.mark.asyncio
    async def test_random_feed_response_time(self, performance_test_runner, performance_requirements):
        """ランダムフィードレスポンス時間テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "random", "limit": 20}
        headers = {"Content-Type": "application/json"}

        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=1,
            requests_per_user=40
        )

        # ランダムフィードは高速であるべき
        assert metrics.avg_response_time <= performance_requirements['max_response_time_ms'] * 0.8
        assert metrics.success_count >= 35  # 85%以上成功


class TestContentAPIThroughput:
    """コンテンツAPIスループットテスト"""

    @pytest.mark.asyncio
    async def test_concurrent_users_throughput(self, performance_test_runner, performance_requirements):
        """同時ユーザースループットテスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "explore", "limit": 15}
        headers = {"Content-Type": "application/json"}

        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=10,
            requests_per_user=5,
            ramp_up_seconds=2.0
        )

        # スループット要件確認
        assert metrics.throughput_rps >= performance_requirements['min_throughput_rps']
        assert metrics.success_count >= 40  # 80%以上成功

        # エラー率確認
        total_requests = metrics.success_count + metrics.error_count
        error_rate = metrics.error_count / total_requests if total_requests > 0 else 0
        assert error_rate <= performance_requirements['max_error_rate']

    @pytest.mark.asyncio
    async def test_high_concurrency_throughput(self, performance_test_runner, performance_requirements):
        """高同時接続スループットテスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "latest", "limit": 10}
        headers = {"Content-Type": "application/json"}

        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=25,
            requests_per_user=3,
            ramp_up_seconds=5.0
        )

        # 高負荷でも一定のパフォーマンスを維持
        assert metrics.throughput_rps >= performance_requirements['min_throughput_rps'] * 0.7
        assert metrics.p95_response_time <= performance_requirements['p95_response_time_ms'] * 2

    @pytest.mark.asyncio
    async def test_different_feed_types_mixed_load(self, performance_test_runner, performance_requirements):
        """異なるフィードタイプ混合負荷テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        headers = {"Content-Type": "application/json"}

        feed_types = ["explore", "latest", "popular", "random"]
        all_metrics = []

        for feed_type in feed_types:
            payload = {"feed_type": feed_type, "limit": 20}

            metrics = await performance_test_runner.run_load_test(
                endpoint=endpoint,
                payload=payload,
                headers=headers,
                concurrent_users=5,
                requests_per_user=4
            )

            all_metrics.append((feed_type, metrics))

        # 全フィードタイプが要件を満たすこと
        for feed_type, metrics in all_metrics:
            assert metrics.success_count >= 15, f"{feed_type} feed failed performance test"
            assert metrics.avg_response_time <= performance_requirements['max_response_time_ms'], \
                f"{feed_type} feed exceeded response time limit"


class TestContentAPIScalability:
    """コンテンツAPIスケーラビリティテスト"""

    @pytest.mark.asyncio
    async def test_gradual_load_increase(self, performance_test_runner, performance_requirements):
        """段階的負荷増加テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "explore", "limit": 20}
        headers = {"Content-Type": "application/json"}

        user_counts = [1, 5, 10, 15, 20]
        results = []

        for user_count in user_counts:
            metrics = await performance_test_runner.run_load_test(
                endpoint=endpoint,
                payload=payload,
                headers=headers,
                concurrent_users=user_count,
                requests_per_user=3
            )

            results.append((user_count, metrics))

            # 短時間休憩
            await asyncio.sleep(1)

        # スケーラビリティ確認
        for i, (user_count, metrics) in enumerate(results):
            # 負荷増加に伴う劣化が許容範囲内であること
            baseline_multiplier = 1 + (i * 0.3)  # 負荷に応じて30%ずつ許容劣化
            max_allowed_response_time = performance_requirements['max_response_time_ms'] * baseline_multiplier

            assert metrics.avg_response_time <= max_allowed_response_time, \
                f"User count {user_count}: response time {metrics.avg_response_time}ms exceeded limit {max_allowed_response_time}ms"

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, performance_test_runner, performance_requirements):
        """持続負荷パフォーマンステスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "explore", "limit": 20}
        headers = {"Content-Type": "application/json"}

        # 5分間の持続負荷シミュレーション（短縮版）
        duration_seconds = 30  # 実際のテストでは300秒（5分）
        requests_per_second = 10

        total_requests = duration_seconds * requests_per_second
        concurrent_users = 10
        requests_per_user = total_requests // concurrent_users

        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=concurrent_users,
            requests_per_user=requests_per_user
        )

        # 持続負荷でのパフォーマンス要件
        assert metrics.avg_response_time <= performance_requirements['max_response_time_ms']
        assert metrics.memory_usage_mb <= performance_requirements['max_memory_increase_mb']

        # メモリリークがないことの確認（レスポンス時間が安定）
        if len(metrics.response_times) >= 10:
            first_half = metrics.response_times[:len(metrics.response_times)//2]
            second_half = metrics.response_times[len(metrics.response_times)//2:]

            first_half_avg = statistics.mean(first_half)
            second_half_avg = statistics.mean(second_half)

            # 後半のレスポンス時間が前半の2倍を超えないこと
            assert second_half_avg <= first_half_avg * 2


class TestContentAPIResourceUsage:
    """コンテンツAPIリソース使用量テスト"""

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, performance_test_runner, performance_requirements):
        """メモリ使用量安定性テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "explore", "limit": 50}  # 大きめのペイロード
        headers = {"Content-Type": "application/json"}

        # 複数回のテスト実行でメモリ使用量を監視
        memory_usage_samples = []

        for iteration in range(5):
            metrics = await performance_test_runner.run_load_test(
                endpoint=endpoint,
                payload=payload,
                headers=headers,
                concurrent_users=5,
                requests_per_user=10
            )

            memory_usage_samples.append(metrics.memory_usage_mb)

            # 短時間待機
            await asyncio.sleep(2)

        # メモリ使用量が安定していること
        max_memory_increase = max(memory_usage_samples)
        assert max_memory_increase <= performance_requirements['max_memory_increase_mb']

        # メモリリークがないこと（使用量が増加し続けない）
        if len(memory_usage_samples) >= 3:
            trend = statistics.linear_regression(
                range(len(memory_usage_samples)),
                memory_usage_samples
            ).slope

            # メモリ使用量の増加トレンドが緩やか
            assert trend < 10  # 10MB/iteration未満

    @pytest.mark.asyncio
    async def test_cpu_usage_efficiency(self, performance_test_runner, performance_requirements):
        """CPU使用効率テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "personalized", "limit": 30}
        headers = {"Content-Type": "application/json"}

        # CPU集約的な処理での効率テスト
        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=8,
            requests_per_user=5
        )

        # CPU使用率が許容範囲内であること
        assert metrics.cpu_usage_percent <= performance_requirements['max_cpu_usage_percent']

        # スループットとCPU使用率の効率性
        efficiency = metrics.throughput_rps / max(metrics.cpu_usage_percent, 1)
        assert efficiency >= 0.5  # 最低限の効率性


@pytest.mark.slow
class TestContentAPIStressTest:
    """コンテンツAPIストレステスト"""

    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self, performance_test_runner):
        """極端な同時負荷テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "explore", "limit": 20}
        headers = {"Content-Type": "application/json"}

        # 極端な負荷
        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=50,
            requests_per_user=2,
            ramp_up_seconds=10.0
        )

        # システムが完全に停止しないこと
        assert metrics.success_count > 0
        assert metrics.avg_response_time < 10000  # 10秒以下

    @pytest.mark.asyncio
    async def test_long_running_stress(self, performance_test_runner):
        """長時間ストレステスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "latest", "limit": 15}
        headers = {"Content-Type": "application/json"}

        # 長時間負荷（実際は短縮）
        metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=15,
            requests_per_user=10
        )

        # 長時間実行後も正常動作
        assert metrics.success_count >= 100  # 最低限の成功数
        assert metrics.avg_response_time < 2000  # 2秒以下

    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(self, performance_test_runner):
        """リソース枯渇回復テスト"""
        endpoint = f"{performance_test_runner.functions_url}/content/feed"
        payload = {"feed_type": "random", "limit": 100}  # 大きなペイロード
        headers = {"Content-Type": "application/json"}

        # リソース枯渇を誘発
        heavy_metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=payload,
            headers=headers,
            concurrent_users=30,
            requests_per_user=3
        )

        # 短時間休憩
        await asyncio.sleep(5)

        # 回復確認
        recovery_payload = {"feed_type": "explore", "limit": 20}
        recovery_metrics = await performance_test_runner.run_load_test(
            endpoint=endpoint,
            payload=recovery_payload,
            headers=headers,
            concurrent_users=5,
            requests_per_user=3
        )

        # システムが回復すること
        assert recovery_metrics.success_count >= 10
        assert recovery_metrics.avg_response_time < 1000


# テスト実行用ヘルパー関数
def run_performance_tests():
    """パフォーマンステストの実行"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow",
        "--durations=10"
    ])


def run_full_performance_tests():
    """完全パフォーマンステストの実行（ストレステスト含む）"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=20"
    ])


if __name__ == "__main__":
    run_performance_tests()