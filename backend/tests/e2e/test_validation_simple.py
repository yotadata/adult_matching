"""
Simple E2E Validation Tests

シンプルE2E検証テスト - 基本的なエンドツーエンドテスト機能の検証
"""
import pytest
import asyncio
import time
from datetime import datetime, timezone


class TestE2EValidation:
    """E2E検証テストスイート"""
    
    @pytest.mark.e2e
    @pytest.mark.validation
    @pytest.mark.asyncio
    async def test_system_integration_basic(self, system_under_test, performance_monitor):
        """基本システム統合テスト"""
        test_name = "basic_system_integration"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "validation_user"
        token = f"valid_token_{user_id}"
        
        try:
            # Step 1: 認証テスト
            start_time = time.time()
            auth_result = await system_under_test.authenticate_user(user_id, token)
            auth_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, "authentication", auth_time, auth_result)
            
            assert auth_result, "Authentication should succeed"
            
            # Step 2: 推薦取得テスト
            start_time = time.time()
            recommendations = await system_under_test.get_recommendations_for_user(user_id, num_recommendations=5)
            rec_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, "recommendations", rec_time, 'error' not in recommendations)
            
            assert 'error' not in recommendations, f"Recommendations should be generated: {recommendations.get('error', '')}"
            assert len(recommendations.get('recommendations', [])) > 0, "Should return recommendations"
            assert rec_time < 1000, f"Recommendation time too slow: {rec_time}ms"
            
            # Step 3: インタラクションテスト
            video_id = "validation_video"
            start_time = time.time()
            interaction_result = await system_under_test.record_user_interaction(user_id, video_id, "like")
            interaction_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, "interaction", interaction_time, interaction_result.get('success', False))
            
            assert interaction_result.get('success', False), f"Interaction should succeed: {interaction_result}"
            assert interaction_time < 500, f"Interaction time too slow: {interaction_time}ms"
            
            # Step 4: フィード取得テスト
            start_time = time.time()
            feed = await system_under_test.get_user_feed(user_id, page_size=10)
            feed_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, "feed", feed_time, 'error' not in feed)
            
            assert 'error' not in feed, f"Feed should be generated: {feed.get('error', '')}"
            assert len(feed.get('feed', [])) > 0, "Feed should contain videos"
            assert feed_time < 1000, f"Feed time too slow: {feed_time}ms"
            
            performance_monitor.stop_monitoring(test_name)
            summary = performance_monitor.get_summary(test_name)
            
            assert summary['success_rate'] >= 1.0, f"All operations should succeed: {summary['success_rate']}"
            assert summary['avg_duration_ms'] < 500, f"Average response time too slow: {summary['avg_duration_ms']}ms"
            
            print(f"✅ Basic system integration test passed:")
            print(f"   Operations: {summary['total_operations']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            print(f"   Average time: {summary['avg_duration_ms']:.1f}ms")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.validation
    @pytest.mark.asyncio
    async def test_user_workflow_simulation(self, system_under_test, performance_monitor):
        """ユーザーワークフローシミュレーション"""
        test_name = "user_workflow_simulation"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "workflow_user"
        token = f"valid_token_{user_id}"
        
        try:
            # 新規ユーザーのワークフローをシミュレート
            workflow_steps = [
                ("authenticate", lambda: system_under_test.authenticate_user(user_id, token)),
                ("get_initial_feed", lambda: system_under_test.get_user_feed(user_id, page_size=10)),
                ("like_video", lambda: system_under_test.record_user_interaction(user_id, "video_1", "like")),
                ("dislike_video", lambda: system_under_test.record_user_interaction(user_id, "video_2", "dislike")),
                ("get_updated_feed", lambda: system_under_test.get_user_feed(user_id, page_size=10)),
                ("get_recommendations", lambda: system_under_test.get_recommendations_for_user(user_id, 5))
            ]
            
            workflow_success = 0
            
            for step_name, step_function in workflow_steps:
                start_time = time.time()
                
                try:
                    result = await step_function()
                    step_time = (time.time() - start_time) * 1000
                    
                    # 結果の妥当性チェック
                    if step_name == "authenticate":
                        success = result == True
                    elif step_name in ["get_initial_feed", "get_updated_feed"]:
                        success = 'error' not in result and len(result.get('feed', [])) > 0
                    elif step_name in ["like_video", "dislike_video"]:
                        success = result.get('success', False)
                    elif step_name == "get_recommendations":
                        success = 'error' not in result and len(result.get('recommendations', [])) > 0
                    else:
                        success = True
                    
                    performance_monitor.record_operation(test_name, step_name, step_time, success)
                    
                    if success:
                        workflow_success += 1
                    
                    print(f"   {step_name}: {'✅' if success else '❌'} ({step_time:.1f}ms)")
                    
                except Exception as e:
                    performance_monitor.record_operation(test_name, step_name, 0, False)
                    print(f"   {step_name}: ❌ ERROR - {str(e)}")
                
                # ステップ間の短い間隔
                await asyncio.sleep(0.05)
            
            performance_monitor.stop_monitoring(test_name)
            summary = performance_monitor.get_summary(test_name)
            
            # ワークフロー検証
            workflow_completion_rate = workflow_success / len(workflow_steps)
            assert workflow_completion_rate >= 0.8, f"Workflow completion rate too low: {workflow_completion_rate:.1%}"
            
            print(f"✅ User workflow simulation passed:")
            print(f"   Workflow steps: {len(workflow_steps)}")
            print(f"   Completed successfully: {workflow_success}")
            print(f"   Completion rate: {workflow_completion_rate:.1%}")
            print(f"   Total duration: {summary['total_duration']:.2f}s")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "workflow_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.validation
    @pytest.mark.asyncio
    async def test_performance_baseline(self, system_under_test, performance_monitor, system_config):
        """パフォーマンスベースラインテスト"""
        test_name = "performance_baseline"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "performance_user"
        sample_size = 20
        latency_threshold = system_config['performance_thresholds']['recommendation_latency_ms']
        
        latencies = []
        successes = 0
        
        try:
            for i in range(sample_size):
                start_time = time.time()
                
                try:
                    result = await system_under_test.get_recommendations_for_user(user_id, num_recommendations=10)
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                    
                    success = 'error' not in result and len(result.get('recommendations', [])) > 0
                    if success:
                        successes += 1
                    
                    performance_monitor.record_operation(test_name, f"recommendation_{i}", latency, success)
                    
                except Exception as e:
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                    performance_monitor.record_operation(test_name, f"recommendation_{i}", latency, False)
                
                await asyncio.sleep(0.05)  # レート制限考慮
            
            performance_monitor.stop_monitoring(test_name)
            
            # パフォーマンス分析
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            success_rate = successes / sample_size
            
            # パフォーマンス要件検証
            assert success_rate >= 0.95, f"Success rate too low: {success_rate:.1%}"
            assert avg_latency < latency_threshold, f"Average latency too high: {avg_latency:.1f}ms > {latency_threshold}ms"
            assert max_latency < latency_threshold * 2, f"Max latency too high: {max_latency:.1f}ms"
            
            print(f"✅ Performance baseline test passed:")
            print(f"   Sample size: {sample_size}")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Average latency: {avg_latency:.1f}ms (threshold: {latency_threshold}ms)")
            print(f"   Max latency: {max_latency:.1f}ms")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "performance_error", 0, False)
            raise