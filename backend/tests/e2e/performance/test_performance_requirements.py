"""
Performance Requirements End-to-End Tests

パフォーマンス要件エンドツーエンドテスト - システム性能とスケーラビリティテスト
"""
import pytest
import asyncio
import time
import concurrent.futures
from typing import Dict, List, Any, Tuple
import statistics
import psutil
from datetime import datetime, timezone


class TestPerformanceRequirements:
    """パフォーマンス要件テストスイート"""
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_recommendation_latency_requirement(
        self, 
        system_under_test, 
        performance_monitor,
        system_config,
        test_user_profiles
    ):
        """推薦レスポンス時間要件テスト (<500ms)"""
        test_name = "recommendation_latency"
        performance_monitor.start_monitoring(test_name)
        
        target_latency_ms = system_config['performance_thresholds']['recommendation_latency_ms']
        sample_size = 100
        user_ids = list(test_user_profiles.keys())
        
        latencies = []
        failures = 0
        
        try:
            for i in range(sample_size):
                user_id = test_user_profiles[user_ids[i % len(user_ids)]]['user_id']
                
                start_time = time.time()
                result = await system_under_test.get_recommendations_for_user(user_id, num_recommendations=10)
                latency = (time.time() - start_time) * 1000
                
                success = 'error' not in result and len(result.get('recommendations', [])) > 0
                performance_monitor.record_operation(test_name, f"recommendation_{i}", latency, success)
                
                if success:
                    latencies.append(latency)
                else:
                    failures += 1
                
                # APIを過負荷にしないための短い間隔
                await asyncio.sleep(0.01)
            
            performance_monitor.stop_monitoring(test_name)
            summary = performance_monitor.get_summary(test_name)
            
            # 統計計算
            avg_latency = statistics.mean(latencies) if latencies else float('inf')
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else float('inf')
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else float('inf')
            max_latency = max(latencies) if latencies else float('inf')
            
            # 要件検証
            success_rate = len(latencies) / sample_size
            assert success_rate >= 0.95, f"Success rate too low: {success_rate:.1%}"
            assert avg_latency < target_latency_ms, f"Average latency too high: {avg_latency:.1f}ms > {target_latency_ms}ms"
            assert p95_latency < target_latency_ms * 1.5, f"P95 latency too high: {p95_latency:.1f}ms"
            assert p99_latency < target_latency_ms * 2.0, f"P99 latency too high: {p99_latency:.1f}ms"
            
            print(f"✅ Recommendation latency requirement test passed:")
            print(f"   Sample size: {sample_size}")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Average latency: {avg_latency:.1f}ms (target: <{target_latency_ms}ms)")
            print(f"   P95 latency: {p95_latency:.1f}ms")
            print(f"   P99 latency: {p99_latency:.1f}ms")
            print(f"   Max latency: {max_latency:.1f}ms")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_scalability(
        self, 
        system_under_test, 
        performance_monitor,
        system_config,
        load_test_config
    ):
        """並行ユーザースケーラビリティテスト"""
        test_name = "concurrent_scalability"
        performance_monitor.start_monitoring(test_name)
        
        concurrent_users = load_test_config['concurrent_users']
        target_throughput = system_config['performance_thresholds']['throughput_rps']
        
        scalability_results = {}
        
        try:
            for user_count in concurrent_users:
                print(f"Testing with {user_count} concurrent users...")
                
                # 並行ユーザーセッション
                user_tasks = []
                for i in range(user_count):
                    user_id = f"load_test_user_{i}"
                    task = self._simulate_user_load(
                        system_under_test, user_id, 
                        load_test_config['operations_per_user'],
                        performance_monitor, f"{test_name}_{user_count}users"
                    )
                    user_tasks.append(task)
                
                # 負荷テスト実行
                start_time = time.time()
                session_results = await asyncio.gather(*user_tasks, return_exceptions=True)
                total_time = time.time() - start_time
                
                # 結果分析
                successful_sessions = sum(1 for r in session_results if not isinstance(r, Exception))
                total_operations = user_count * load_test_config['operations_per_user']
                actual_throughput = total_operations / total_time
                
                scalability_results[user_count] = {
                    'successful_sessions': successful_sessions,
                    'total_sessions': user_count,
                    'total_time': total_time,
                    'throughput_rps': actual_throughput,
                    'success_rate': successful_sessions / user_count
                }
                
                performance_monitor.record_operation(
                    test_name, f"load_{user_count}users", total_time * 1000,
                    successful_sessions >= user_count * 0.9
                )
                
                # システム回復時間
                await asyncio.sleep(2)
            
            performance_monitor.stop_monitoring(test_name)
            
            # スケーラビリティ検証
            for user_count, result in scalability_results.items():
                assert result['success_rate'] >= 0.90, \
                    f"Success rate too low with {user_count} users: {result['success_rate']:.1%}"
                
                if user_count <= 10:
                    assert result['throughput_rps'] >= target_throughput * 0.8, \
                        f"Throughput too low with {user_count} users: {result['throughput_rps']:.1f} < {target_throughput * 0.8:.1f}"
            
            print(f"✅ Concurrent user scalability test passed:")
            for user_count, result in scalability_results.items():
                print(f"   {user_count} users: {result['success_rate']:.1%} success, {result['throughput_rps']:.1f} ops/sec")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """負荷下でのメモリ使用量テスト"""
        test_name = "memory_under_load"
        performance_monitor.start_monitoring(test_name)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        max_memory_increase_mb = 200  # 200MB制限
        
        try:
            # 高負荷操作実行
            heavy_load_tasks = []
            
            # 大量の並行推薦リクエスト
            for i in range(50):
                user_id = f"memory_test_user_{i}"
                task = system_under_test.get_recommendations_for_user(user_id, num_recommendations=50)
                heavy_load_tasks.append(task)
            
            # メモリ監視付きで実行
            start_time = time.time()
            monitoring_task = asyncio.create_task(
                self._monitor_memory_usage(process, memory_samples, duration=10)
            )
            
            # 並行負荷実行
            load_results = await asyncio.gather(*heavy_load_tasks, return_exceptions=True)
            
            # メモリ監視停止
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            total_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(memory_samples)
            memory_increase = peak_memory - initial_memory
            
            # 成功した操作数カウント
            successful_operations = sum(1 for r in load_results if not isinstance(r, Exception))
            
            performance_monitor.record_operation(
                test_name, "heavy_load_test", total_time * 1000,
                successful_operations >= 45 and memory_increase < max_memory_increase_mb
            )
            
            performance_monitor.stop_monitoring(test_name)
            
            # メモリ使用量検証
            assert memory_increase < max_memory_increase_mb, \
                f"Memory increase too high: {memory_increase:.1f}MB > {max_memory_increase_mb}MB"
            assert successful_operations >= 45, \
                f"Too many failed operations: {successful_operations}/50"
            
            print(f"✅ Memory usage under load test passed:")
            print(f"   Initial memory: {initial_memory:.1f}MB")
            print(f"   Peak memory: {peak_memory:.1f}MB")
            print(f"   Memory increase: {memory_increase:.1f}MB")
            print(f"   Successful operations: {successful_operations}/50")
            print(f"   Test duration: {total_time:.2f}s")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sustained_load_stability(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """継続負荷安定性テスト"""
        test_name = "sustained_load_stability"
        performance_monitor.start_monitoring(test_name)
        
        test_duration = 60  # 60秒間の継続テスト
        target_rps = 20  # 20 requests per second
        user_count = 10
        
        stability_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'error_rate_samples': [],
            'latency_samples': [],
            'throughput_samples': []
        }
        
        try:
            end_time = time.time() + test_duration
            sample_interval = 5  # 5秒ごとにサンプリング
            
            while time.time() < end_time:
                sample_start = time.time()
                sample_requests = 0
                sample_successes = 0
                sample_latencies = []
                
                # 5秒間のサンプル期間中の操作
                sample_end = sample_start + sample_interval
                
                while time.time() < sample_end:
                    # 並行リクエスト送信
                    batch_tasks = []
                    for i in range(user_count):
                        user_id = f"stability_user_{i}"
                        task = self._timed_recommendation_request(
                            system_under_test, user_id, performance_monitor, test_name
                        )
                        batch_tasks.append(task)
                    
                    batch_start = time.time()
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # バッチ結果処理
                    for result in batch_results:
                        sample_requests += 1
                        if not isinstance(result, Exception) and result.get('success', False):
                            sample_successes += 1
                            sample_latencies.append(result['latency_ms'])
                    
                    # 目標RPSに合わせた間隔調整
                    batch_time = time.time() - batch_start
                    target_interval = user_count / target_rps
                    if batch_time < target_interval:
                        await asyncio.sleep(target_interval - batch_time)
                
                # サンプル期間の統計
                sample_duration = time.time() - sample_start
                sample_error_rate = 1 - (sample_successes / sample_requests) if sample_requests > 0 else 1
                sample_throughput = sample_successes / sample_duration
                avg_latency = statistics.mean(sample_latencies) if sample_latencies else 0
                
                stability_metrics['total_requests'] += sample_requests
                stability_metrics['successful_requests'] += sample_successes
                stability_metrics['error_rate_samples'].append(sample_error_rate)
                stability_metrics['throughput_samples'].append(sample_throughput)
                stability_metrics['latency_samples'].append(avg_latency)
                
                print(f"Sample: {sample_duration:.1f}s, {sample_successes}/{sample_requests} success, "
                      f"{sample_throughput:.1f} rps, {avg_latency:.1f}ms avg latency")
            
            performance_monitor.stop_monitoring(test_name)
            
            # 安定性検証
            overall_success_rate = stability_metrics['successful_requests'] / stability_metrics['total_requests']
            avg_error_rate = statistics.mean(stability_metrics['error_rate_samples'])
            avg_throughput = statistics.mean(stability_metrics['throughput_samples'])
            avg_latency = statistics.mean(stability_metrics['latency_samples'])
            throughput_variance = statistics.variance(stability_metrics['throughput_samples'])
            
            assert overall_success_rate >= 0.95, f"Overall success rate too low: {overall_success_rate:.1%}"
            assert avg_error_rate <= 0.1, f"Average error rate too high: {avg_error_rate:.1%}"
            assert avg_throughput >= target_rps * 0.8, f"Average throughput too low: {avg_throughput:.1f} < {target_rps * 0.8:.1f}"
            assert throughput_variance < (target_rps * 0.2) ** 2, f"Throughput too unstable: {throughput_variance:.2f}"
            
            print(f"✅ Sustained load stability test passed:")
            print(f"   Test duration: {test_duration}s")
            print(f"   Total requests: {stability_metrics['total_requests']}")
            print(f"   Overall success rate: {overall_success_rate:.1%}")
            print(f"   Average throughput: {avg_throughput:.1f} rps")
            print(f"   Average latency: {avg_latency:.1f}ms")
            print(f"   Throughput stability (variance): {throughput_variance:.2f}")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    async def _simulate_user_load(
        self, 
        system_under_test, 
        user_id: str, 
        operations: int,
        performance_monitor,
        test_name: str
    ) -> Dict[str, Any]:
        """ユーザー負荷シミュレーション"""
        token = f"valid_token_{user_id}"
        completed_operations = 0
        
        try:
            # 認証
            auth_result = await system_under_test.authenticate_user(user_id, token)
            if not auth_result:
                return {'user_id': user_id, 'completed_operations': 0, 'error': 'auth_failed'}
            
            # 指定された操作数を実行
            for i in range(operations):
                operation_type = 'recommendation' if i % 2 == 0 else 'interaction'
                
                try:
                    if operation_type == 'recommendation':
                        result = await system_under_test.get_recommendations_for_user(user_id)
                        success = 'error' not in result
                    else:
                        video_id = f"load_video_{i}"
                        interaction_type = 'like' if i % 3 != 0 else 'dislike'
                        result = await system_under_test.record_user_interaction(user_id, video_id, interaction_type)
                        success = result.get('success', False)
                    
                    if success:
                        completed_operations += 1
                    
                    # 短い間隔
                    await asyncio.sleep(0.01)
                    
                except Exception:
                    pass
            
            return {'user_id': user_id, 'completed_operations': completed_operations}
            
        except Exception as e:
            return {'user_id': user_id, 'completed_operations': completed_operations, 'error': str(e)}
    
    async def _monitor_memory_usage(self, process, memory_samples: List[float], duration: int):
        """メモリ使用量監視"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            await asyncio.sleep(0.1)  # 100ms間隔
    
    async def _timed_recommendation_request(
        self, 
        system_under_test, 
        user_id: str, 
        performance_monitor,
        test_name: str
    ) -> Dict[str, Any]:
        """時間測定付き推薦リクエスト"""
        start_time = time.time()
        
        try:
            result = await system_under_test.get_recommendations_for_user(user_id)
            latency_ms = (time.time() - start_time) * 1000
            success = 'error' not in result and len(result.get('recommendations', [])) > 0
            
            performance_monitor.record_operation(test_name, f"timed_request_{user_id}", latency_ms, success)
            
            return {
                'success': success,
                'latency_ms': latency_ms,
                'user_id': user_id
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, f"timed_request_{user_id}", latency_ms, False)
            
            return {
                'success': False,
                'latency_ms': latency_ms,
                'error': str(e),
                'user_id': user_id
            }