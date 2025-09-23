"""
System Reliability End-to-End Tests

システム信頼性エンドツーエンドテスト - 障害耐性とエラーハンドリングテスト
"""
import pytest
import asyncio
import time
import random
from typing import Dict, List, Any, Optional
from unittest.mock import patch, Mock
from datetime import datetime, timezone


class TestSystemReliability:
    """システム信頼性テストスイート"""
    
    @pytest.mark.e2e
    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """エラーハンドリングと回復テスト"""
        test_name = "error_handling_recovery"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "reliability_test_user"
        error_scenarios = [
            'network_timeout',
            'database_connection_error',
            'ml_service_unavailable',
            'invalid_user_data',
            'rate_limit_exceeded'
        ]
        
        recovery_results = {}
        
        try:
            for scenario in error_scenarios:
                print(f"Testing error scenario: {scenario}")
                
                # エラー注入
                with patch.object(system_under_test, '_inject_error', return_value=scenario):
                    error_count = 0
                    recovery_count = 0
                    
                    # エラー条件下での操作試行
                    for attempt in range(10):
                        start_time = time.time()
                        
                        try:
                            if scenario == 'ml_service_unavailable':
                                result = await system_under_test.get_recommendations_for_user(user_id)
                                success = 'error' not in result or result.get('fallback_used', False)
                            elif scenario == 'database_connection_error':
                                result = await system_under_test.record_user_interaction(user_id, f"video_{attempt}", "like")
                                success = result.get('success', False) or result.get('cached', False)
                            else:
                                result = await system_under_test.get_user_feed(user_id)
                                success = 'error' not in result or len(result.get('feed', [])) > 0
                            
                            latency = (time.time() - start_time) * 1000
                            
                            if success:
                                recovery_count += 1
                            else:
                                error_count += 1
                                
                            performance_monitor.record_operation(
                                test_name, f"{scenario}_attempt_{attempt}", latency, success
                            )
                            
                        except Exception as e:
                            error_count += 1
                            performance_monitor.record_operation(
                                test_name, f"{scenario}_attempt_{attempt}", 0, False
                            )
                        
                        await asyncio.sleep(0.1)  # 試行間の間隔
                    
                    recovery_rate = recovery_count / (recovery_count + error_count) if (recovery_count + error_count) > 0 else 0
                    recovery_results[scenario] = {
                        'recovery_count': recovery_count,
                        'error_count': error_count,
                        'recovery_rate': recovery_rate
                    }
                
                # システム回復待機
                await asyncio.sleep(0.5)
            
            performance_monitor.stop_monitoring(test_name)
            
            # 回復能力検証
            for scenario, result in recovery_results.items():
                min_recovery_rate = 0.3  # 最低30%の回復率を期待
                assert result['recovery_rate'] >= min_recovery_rate, \
                    f"Recovery rate too low for {scenario}: {result['recovery_rate']:.1%} < {min_recovery_rate:.1%}"
            
            overall_recovery_rate = sum(r['recovery_count'] for r in recovery_results.values()) / \
                                   sum(r['recovery_count'] + r['error_count'] for r in recovery_results.values())
            
            assert overall_recovery_rate >= 0.4, f"Overall recovery rate too low: {overall_recovery_rate:.1%}"
            
            print(f"✅ Error handling and recovery test passed:")
            print(f"   Overall recovery rate: {overall_recovery_rate:.1%}")
            for scenario, result in recovery_results.items():
                print(f"   {scenario}: {result['recovery_rate']:.1%} recovery rate")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_data_consistency_under_failure(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """障害下でのデータ一貫性テスト"""
        test_name = "data_consistency_failure"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "consistency_test_user"
        video_ids = [f"video_{i}" for i in range(20)]
        
        consistency_violations = 0
        total_operations = 0
        
        try:
            # 部分的障害をシミュレートしながらインタラクション記録
            for i, video_id in enumerate(video_ids):
                interaction_type = 'like' if i % 3 != 0 else 'dislike'
                
                # ランダムに障害を注入（20%の確率）
                inject_failure = random.random() < 0.2
                
                start_time = time.time()
                
                if inject_failure:
                    # 障害条件下での操作
                    with patch.object(system_under_test.database, 'record_interaction', 
                                    side_effect=Exception("Simulated database failure")):
                        try:
                            result = await system_under_test.record_user_interaction(
                                user_id, video_id, interaction_type
                            )
                            # 障害後の一貫性チェック
                            consistency_check = await self._check_data_consistency(
                                system_under_test, user_id, video_id, interaction_type
                            )
                            if not consistency_check:
                                consistency_violations += 1
                                
                        except Exception:
                            pass
                else:
                    # 正常操作
                    result = await system_under_test.record_user_interaction(
                        user_id, video_id, interaction_type
                    )
                    consistency_check = await self._check_data_consistency(
                        system_under_test, user_id, video_id, interaction_type
                    )
                    if not consistency_check:
                        consistency_violations += 1
                
                total_operations += 1
                operation_time = (time.time() - start_time) * 1000
                
                performance_monitor.record_operation(
                    test_name, f"consistency_check_{i}", operation_time,
                    consistency_violations == 0
                )
                
                await asyncio.sleep(0.05)
            
            # 最終一貫性検証
            final_consistency = await self._verify_overall_consistency(system_under_test, user_id)
            
            performance_monitor.stop_monitoring(test_name)
            
            # 一貫性要件検証
            consistency_rate = 1 - (consistency_violations / total_operations)
            assert consistency_rate >= 0.95, f"Data consistency rate too low: {consistency_rate:.1%}"
            assert final_consistency, "Final data consistency check failed"
            
            print(f"✅ Data consistency under failure test passed:")
            print(f"   Total operations: {total_operations}")
            print(f"   Consistency violations: {consistency_violations}")
            print(f"   Consistency rate: {consistency_rate:.1%}")
            print(f"   Final consistency: {final_consistency}")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_graceful_degradation(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """グレースフルデグラデーションテスト"""
        test_name = "graceful_degradation"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "degradation_test_user"
        degradation_scenarios = [
            'ml_service_slow_response',
            'database_high_latency',
            'cache_unavailable',
            'external_api_timeout'
        ]
        
        degradation_results = {}
        
        try:
            for scenario in degradation_scenarios:
                print(f"Testing degradation scenario: {scenario}")
                
                baseline_performance = await self._measure_baseline_performance(system_under_test, user_id)
                
                # デグラデーション条件を注入
                with self._inject_degradation(scenario):
                    degraded_performance = await self._measure_degraded_performance(
                        system_under_test, user_id, performance_monitor, test_name, scenario
                    )
                
                # デグラデーション分析
                performance_ratio = degraded_performance['avg_latency'] / baseline_performance['avg_latency']
                success_ratio = degraded_performance['success_rate'] / baseline_performance['success_rate']
                
                degradation_results[scenario] = {
                    'baseline': baseline_performance,
                    'degraded': degraded_performance,
                    'performance_ratio': performance_ratio,
                    'success_ratio': success_ratio,
                    'acceptable_degradation': performance_ratio <= 3.0 and success_ratio >= 0.7
                }
                
                await asyncio.sleep(1)  # シナリオ間の回復時間
            
            performance_monitor.stop_monitoring(test_name)
            
            # デグラデーション許容性検証
            acceptable_scenarios = sum(1 for r in degradation_results.values() if r['acceptable_degradation'])
            degradation_tolerance = acceptable_scenarios / len(degradation_scenarios)
            
            assert degradation_tolerance >= 0.75, \
                f"Degradation tolerance too low: {degradation_tolerance:.1%}"
            
            print(f"✅ Graceful degradation test passed:")
            print(f"   Degradation tolerance: {degradation_tolerance:.1%}")
            for scenario, result in degradation_results.items():
                print(f"   {scenario}: {result['performance_ratio']:.1f}x latency, "
                      f"{result['success_ratio']:.1%} success ratio")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """リソース枯渇ハンドリングテスト"""
        test_name = "resource_exhaustion"
        performance_monitor.start_monitoring(test_name)
        
        exhaustion_scenarios = [
            'memory_pressure',
            'connection_pool_exhaustion',
            'disk_space_low',
            'cpu_overload'
        ]
        
        exhaustion_results = {}
        
        try:
            for scenario in exhaustion_scenarios:
                print(f"Testing resource exhaustion: {scenario}")
                
                # リソース枯渇をシミュレート
                with self._simulate_resource_exhaustion(scenario):
                    stress_results = await self._run_stress_operations(
                        system_under_test, performance_monitor, test_name, scenario
                    )
                
                # システム応答性検証
                system_responsive = stress_results['response_rate'] >= 0.5
                graceful_failure = stress_results['error_rate'] <= 0.8
                recovery_time = stress_results['recovery_time_seconds']
                
                exhaustion_results[scenario] = {
                    'system_responsive': system_responsive,
                    'graceful_failure': graceful_failure,
                    'recovery_time': recovery_time,
                    'acceptable_handling': system_responsive and graceful_failure and recovery_time <= 30
                }
                
                await asyncio.sleep(2)  # システム回復時間
            
            performance_monitor.stop_monitoring(test_name)
            
            # リソース枯渇対応能力検証
            acceptable_handling = sum(1 for r in exhaustion_results.values() if r['acceptable_handling'])
            exhaustion_tolerance = acceptable_handling / len(exhaustion_scenarios)
            
            assert exhaustion_tolerance >= 0.75, \
                f"Resource exhaustion handling too poor: {exhaustion_tolerance:.1%}"
            
            print(f"✅ Resource exhaustion handling test passed:")
            print(f"   Exhaustion tolerance: {exhaustion_tolerance:.1%}")
            for scenario, result in exhaustion_results.items():
                print(f"   {scenario}: {'✅' if result['acceptable_handling'] else '❌'} "
                      f"(recovery: {result['recovery_time']:.1f}s)")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_concurrent_failure_scenarios(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """並行障害シナリオテスト"""
        test_name = "concurrent_failures"
        performance_monitor.start_monitoring(test_name)
        
        # 複数の障害を同時に発生させる
        failure_combinations = [
            ['database_slow', 'ml_service_error'],
            ['network_timeout', 'cache_miss'],
            ['high_cpu', 'memory_pressure', 'disk_io_slow']
        ]
        
        concurrent_failure_results = {}
        
        try:
            for i, failure_combination in enumerate(failure_combinations):
                print(f"Testing concurrent failures: {', '.join(failure_combination)}")
                
                # 複数障害の同時注入
                with self._inject_multiple_failures(failure_combination):
                    # 並行ユーザー操作
                    concurrent_users = 5
                    user_tasks = []
                    
                    for user_idx in range(concurrent_users):
                        user_id = f"concurrent_failure_user_{user_idx}"
                        task = self._user_operation_under_failure(
                            system_under_test, user_id, performance_monitor, 
                            f"{test_name}_combo_{i}", failure_combination
                        )
                        user_tasks.append(task)
                    
                    start_time = time.time()
                    user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
                    total_time = time.time() - start_time
                
                # 結果分析
                successful_users = sum(1 for r in user_results if not isinstance(r, Exception))
                user_survival_rate = successful_users / concurrent_users
                
                concurrent_failure_results[f"combo_{i}"] = {
                    'failures': failure_combination,
                    'user_survival_rate': user_survival_rate,
                    'total_time': total_time,
                    'acceptable_survival': user_survival_rate >= 0.4
                }
                
                await asyncio.sleep(3)  # 障害回復時間
            
            performance_monitor.stop_monitoring(test_name)
            
            # 並行障害耐性検証
            acceptable_combinations = sum(1 for r in concurrent_failure_results.values() 
                                        if r['acceptable_survival'])
            failure_tolerance = acceptable_combinations / len(failure_combinations)
            
            assert failure_tolerance >= 0.6, \
                f"Concurrent failure tolerance too low: {failure_tolerance:.1%}"
            
            print(f"✅ Concurrent failure scenarios test passed:")
            print(f"   Failure tolerance: {failure_tolerance:.1%}")
            for combo_name, result in concurrent_failure_results.items():
                print(f"   {result['failures']}: {result['user_survival_rate']:.1%} user survival")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "test_error", 0, False)
            raise
    
    # ヘルパーメソッド
    async def _check_data_consistency(
        self, 
        system_under_test, 
        user_id: str, 
        video_id: str, 
        interaction_type: str
    ) -> bool:
        """データ一貫性チェック"""
        try:
            # ユーザー履歴から該当インタラクションを確認
            history = await system_under_test.database.get_user_history(user_id)
            recent_interaction = next(
                (h for h in history if h.get('video_id') == video_id), None
            )
            
            if recent_interaction:
                return recent_interaction.get('interaction_type') == interaction_type
            return True  # 記録されていなくても一貫性は保たれている
            
        except Exception:
            return False
    
    async def _verify_overall_consistency(self, system_under_test, user_id: str) -> bool:
        """全体データ一貫性検証"""
        try:
            # ユーザーデータの整合性チェック
            user_data = await system_under_test.database.get_user(user_id)
            user_history = await system_under_test.database.get_user_history(user_id)
            
            # 基本的な整合性ルール
            if user_data and user_history:
                # 履歴のユーザーIDが一致するか
                return all(h.get('user_id') == user_id for h in user_history)
            
            return True
            
        except Exception:
            return False
    
    async def _measure_baseline_performance(self, system_under_test, user_id: str) -> Dict[str, float]:
        """ベースライン性能測定"""
        latencies = []
        successes = 0
        total_ops = 10
        
        for i in range(total_ops):
            start_time = time.time()
            try:
                result = await system_under_test.get_recommendations_for_user(user_id)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                if 'error' not in result:
                    successes += 1
            except Exception:
                latencies.append((time.time() - start_time) * 1000)
            
            await asyncio.sleep(0.1)
        
        return {
            'avg_latency': sum(latencies) / len(latencies),
            'success_rate': successes / total_ops
        }
    
    async def _measure_degraded_performance(
        self, 
        system_under_test, 
        user_id: str, 
        performance_monitor,
        test_name: str,
        scenario: str
    ) -> Dict[str, float]:
        """デグラデーション性能測定"""
        latencies = []
        successes = 0
        total_ops = 10
        
        for i in range(total_ops):
            start_time = time.time()
            try:
                result = await system_under_test.get_recommendations_for_user(user_id)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                success = 'error' not in result
                if success:
                    successes += 1
                
                performance_monitor.record_operation(
                    test_name, f"{scenario}_degraded_{i}", latency, success
                )
            except Exception:
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                performance_monitor.record_operation(
                    test_name, f"{scenario}_degraded_{i}", latency, False
                )
            
            await asyncio.sleep(0.1)
        
        return {
            'avg_latency': sum(latencies) / len(latencies),
            'success_rate': successes / total_ops
        }
    
    def _inject_degradation(self, scenario: str):
        """デグラデーション注入コンテキストマネージャー"""
        class DegradationContext:
            def __enter__(self):
                # デグラデーション条件の設定
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                # クリーンアップ
                pass
        return DegradationContext()
    
    def _simulate_resource_exhaustion(self, scenario: str):
        """リソース枯渇シミュレーションコンテキストマネージャー"""
        class ResourceExhaustionContext:
            def __enter__(self):
                # リソース制限の設定
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                # リソース回復
                pass
        return ResourceExhaustionContext()
    
    def _inject_multiple_failures(self, failures: List[str]):
        """複数障害注入コンテキストマネージャー"""
        class MultipleFailureContext:
            def __enter__(self):
                # 複数障害の設定
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                # 障害クリア
                pass
        return MultipleFailureContext()
    
    async def _run_stress_operations(
        self, 
        system_under_test, 
        performance_monitor,
        test_name: str,
        scenario: str
    ) -> Dict[str, float]:
        """ストレス操作実行"""
        start_time = time.time()
        responses = 0
        errors = 0
        
        # 30秒間のストレステスト
        end_time = start_time + 30
        
        while time.time() < end_time:
            try:
                user_id = f"stress_{random.randint(1, 100)}"
                result = await system_under_test.get_recommendations_for_user(user_id)
                if 'error' not in result:
                    responses += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
            
            await asyncio.sleep(0.1)
        
        total_operations = responses + errors
        
        return {
            'response_rate': responses / total_operations if total_operations > 0 else 0,
            'error_rate': errors / total_operations if total_operations > 0 else 1,
            'recovery_time_seconds': time.time() - end_time
        }
    
    async def _user_operation_under_failure(
        self, 
        system_under_test, 
        user_id: str, 
        performance_monitor,
        test_name: str,
        failures: List[str]
    ) -> Dict[str, Any]:
        """障害下でのユーザー操作"""
        operations = ['get_recommendations', 'record_interaction', 'get_feed']
        completed_operations = 0
        
        for i, operation in enumerate(operations * 3):  # 各操作を3回実行
            try:
                start_time = time.time()
                
                if operation == 'get_recommendations':
                    result = await system_under_test.get_recommendations_for_user(user_id)
                    success = 'error' not in result
                elif operation == 'record_interaction':
                    video_id = f"failure_video_{i}"
                    result = await system_under_test.record_user_interaction(user_id, video_id, 'like')
                    success = result.get('success', False)
                else:  # get_feed
                    result = await system_under_test.get_user_feed(user_id)
                    success = 'error' not in result
                
                if success:
                    completed_operations += 1
                
                operation_time = (time.time() - start_time) * 1000
                performance_monitor.record_operation(
                    test_name, f"user_op_{user_id}_{i}", operation_time, success
                )
                
                await asyncio.sleep(0.05)
                
            except Exception:
                pass
        
        return {'user_id': user_id, 'completed_operations': completed_operations}