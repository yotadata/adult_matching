"""
User Workflow End-to-End Tests

ユーザーワークフローエンドツーエンドテスト - 主要ユーザージャーニーの完全テスト
"""
import pytest
import asyncio
import time
from typing import Dict, List, Any
from datetime import datetime, timezone


class TestUserWorkflows:
    """ユーザーワークフローテストスイート"""
    
    @pytest.mark.e2e
    @pytest.mark.workflow
    @pytest.mark.asyncio
    async def test_new_user_onboarding_workflow(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """新規ユーザーオンボーディングワークフロー"""
        test_name = "new_user_onboarding"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "test_new_user_001"
        token = f"valid_token_{user_id}"
        
        try:
            # Step 1: ユーザー認証
            start_time = time.time()
            auth_result = await system_under_test.authenticate_user(user_id, token)
            auth_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, "authentication", auth_time, auth_result)
            
            assert auth_result, "Authentication should succeed for valid token"
            
            # Step 2: 初期推薦取得（コールドスタート）
            start_time = time.time()
            initial_feed = await system_under_test.get_user_feed(user_id, page_size=20)
            feed_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, "initial_feed", feed_time, 'error' not in initial_feed)
            
            assert 'error' not in initial_feed, f"Initial feed should be generated: {initial_feed.get('error', '')}"
            assert len(initial_feed['feed']) > 0, "Initial feed should contain recommendations"
            assert feed_time < system_config['performance_thresholds']['recommendation_latency_ms'], \
                f"Feed generation too slow: {feed_time}ms > {system_config['performance_thresholds']['recommendation_latency_ms']}ms"
            
            # Step 3: 初期インタラクション（いいね/パス）
            interactions = ['like', 'dislike', 'like', 'like', 'dislike']
            for i, interaction_type in enumerate(interactions):
                video_id = f"rec_video_{i + 1}"
                
                start_time = time.time()
                interaction_result = await system_under_test.record_user_interaction(
                    user_id, video_id, interaction_type
                )
                interaction_time = (time.time() - start_time) * 1000
                performance_monitor.record_operation(
                    test_name, f"interaction_{interaction_type}", interaction_time, 
                    interaction_result.get('success', False)
                )
                
                assert interaction_result['success'], f"Interaction recording should succeed: {interaction_result}"
                assert interaction_time < system_config['performance_thresholds']['api_response_time_ms'], \
                    f"Interaction recording too slow: {interaction_time}ms"
            
            # Step 4: 学習後の推薦取得
            start_time = time.time()
            improved_feed = await system_under_test.get_user_feed(user_id, page_size=20)
            improved_feed_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(test_name, "improved_feed", improved_feed_time, 'error' not in improved_feed)
            
            assert 'error' not in improved_feed, f"Improved feed should be generated: {improved_feed.get('error', '')}"
            assert len(improved_feed['feed']) > 0, "Improved feed should contain recommendations"
            
            # ワークフロー完了
            performance_monitor.stop_monitoring(test_name)
            summary = performance_monitor.get_summary(test_name)
            
            # パフォーマンス検証
            assert summary['success_rate'] >= 0.95, f"Workflow success rate too low: {summary['success_rate']}"
            assert summary['total_duration'] < 10.0, f"Total workflow too slow: {summary['total_duration']}s"
            
            print(f"✅ New user onboarding workflow completed:")
            print(f"   Total duration: {summary['total_duration']:.2f}s")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            print(f"   Average operation time: {summary['avg_duration_ms']:.1f}ms")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "workflow_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.workflow
    @pytest.mark.asyncio
    async def test_active_user_browsing_workflow(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """アクティブユーザーブラウジングワークフロー"""
        test_name = "active_user_browsing"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "test_user_active"
        token = f"valid_token_{user_id}"
        
        try:
            # Step 1: 認証
            auth_result = await system_under_test.authenticate_user(user_id, token)
            assert auth_result, "Authentication should succeed"
            
            # Step 2: 複数ページのブラウジングセッション
            session_interactions = 0
            total_videos_seen = 0
            
            for page in range(5):  # 5ページ分ブラウジング
                # フィード取得
                start_time = time.time()
                feed = await system_under_test.get_user_feed(user_id, page_size=10)
                feed_time = (time.time() - start_time) * 1000
                performance_monitor.record_operation(test_name, f"feed_page_{page}", feed_time, 'error' not in feed)
                
                assert 'error' not in feed, f"Feed page {page} should load successfully"
                assert len(feed['feed']) > 0, f"Feed page {page} should contain videos"
                
                total_videos_seen += len(feed['feed'])
                
                # ページ内でのインタラクション
                for i, video in enumerate(feed['feed'][:5]):  # 各ページで5つのビデオと相互作用
                    # ビデオ視聴をシミュレート
                    await asyncio.sleep(0.01)  # 視聴時間をシミュレート
                    
                    # インタラクション（80%の確率でlike）
                    interaction_type = 'like' if i < 4 else 'dislike'
                    
                    start_time = time.time()
                    interaction_result = await system_under_test.record_user_interaction(
                        user_id, video['video_id'], interaction_type
                    )
                    interaction_time = (time.time() - start_time) * 1000
                    performance_monitor.record_operation(
                        test_name, f"interaction_page_{page}", interaction_time,
                        interaction_result.get('success', False)
                    )
                    
                    session_interactions += 1
                
                # ページ間の間隔
                await asyncio.sleep(0.05)
            
            # Step 3: セッション終了時の推薦品質チェック
            final_feed = await system_under_test.get_user_feed(user_id, page_size=20)
            assert 'error' not in final_feed, "Final feed should be accessible"
            
            performance_monitor.stop_monitoring(test_name)
            summary = performance_monitor.get_summary(test_name)
            
            # セッション品質検証
            assert session_interactions >= 20, f"Insufficient interactions: {session_interactions}"
            assert total_videos_seen >= 40, f"Insufficient videos seen: {total_videos_seen}"
            assert summary['success_rate'] >= 0.95, f"Session success rate too low: {summary['success_rate']}"
            
            print(f"✅ Active user browsing workflow completed:")
            print(f"   Session duration: {summary['total_duration']:.2f}s")
            print(f"   Videos seen: {total_videos_seen}")
            print(f"   Interactions: {session_interactions}")
            print(f"   Average response time: {summary['avg_duration_ms']:.1f}ms")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "workflow_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.workflow
    @pytest.mark.asyncio
    async def test_heavy_user_session_workflow(
        self, 
        system_under_test, 
        performance_monitor,
        system_config
    ):
        """ヘビーユーザーセッションワークフロー"""
        test_name = "heavy_user_session"
        performance_monitor.start_monitoring(test_name)
        
        user_id = "test_user_heavy"
        token = f"valid_token_{user_id}"
        
        try:
            # Step 1: 認証
            auth_result = await system_under_test.authenticate_user(user_id, token)
            assert auth_result, "Authentication should succeed"
            
            # Step 2: 高頻度インタラクションセッション
            rapid_interactions = 0
            error_count = 0
            
            # 50個のビデオとの高速インタラクション
            for i in range(50):
                video_id = f"rec_video_{i + 1}"
                interaction_type = 'like' if i % 3 != 0 else 'dislike'
                
                start_time = time.time()
                try:
                    interaction_result = await system_under_test.record_user_interaction(
                        user_id, video_id, interaction_type
                    )
                    interaction_time = (time.time() - start_time) * 1000
                    
                    success = interaction_result.get('success', False)
                    performance_monitor.record_operation(
                        test_name, f"rapid_interaction_{i}", interaction_time, success
                    )
                    
                    if success:
                        rapid_interactions += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    performance_monitor.record_operation(test_name, f"rapid_interaction_{i}", 0, False)
                
                # 高頻度アクセス（短い間隔）
                await asyncio.sleep(0.001)
            
            # Step 3: 複数の並行推薦リクエスト
            concurrent_requests = 5
            recommendation_tasks = []
            
            for i in range(concurrent_requests):
                task = system_under_test.get_recommendations_for_user(user_id, num_recommendations=20)
                recommendation_tasks.append(task)
            
            start_time = time.time()
            concurrent_results = await asyncio.gather(*recommendation_tasks, return_exceptions=True)
            concurrent_time = (time.time() - start_time) * 1000
            
            successful_concurrent = sum(1 for result in concurrent_results if not isinstance(result, Exception))
            performance_monitor.record_operation(
                test_name, "concurrent_recommendations", concurrent_time,
                successful_concurrent == concurrent_requests
            )
            
            # Step 4: 大量データ取得
            start_time = time.time()
            large_feed = await system_under_test.get_user_feed(user_id, page_size=100)
            large_feed_time = (time.time() - start_time) * 1000
            performance_monitor.record_operation(
                test_name, "large_feed", large_feed_time, 'error' not in large_feed
            )
            
            performance_monitor.stop_monitoring(test_name)
            summary = performance_monitor.get_summary(test_name)
            
            # ヘビーユーザーセッション検証
            assert rapid_interactions >= 45, f"Too many interaction failures: {rapid_interactions}/50"
            assert error_count <= 5, f"Too many errors: {error_count}"
            assert successful_concurrent >= 4, f"Concurrent request failures: {successful_concurrent}/{concurrent_requests}"
            assert summary['success_rate'] >= 0.90, f"Heavy user session success rate too low: {summary['success_rate']}"
            
            print(f"✅ Heavy user session workflow completed:")
            print(f"   Session duration: {summary['total_duration']:.2f}s")
            print(f"   Rapid interactions: {rapid_interactions}/50")
            print(f"   Concurrent requests: {successful_concurrent}/{concurrent_requests}")
            print(f"   Error rate: {error_count/50:.1%}")
            print(f"   Throughput: {summary['throughput_ops_per_sec']:.1f} ops/sec")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "workflow_error", 0, False)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.workflow
    @pytest.mark.asyncio
    async def test_mixed_user_scenario_workflow(
        self, 
        system_under_test, 
        performance_monitor,
        system_config,
        test_user_profiles
    ):
        """混合ユーザーシナリオワークフロー"""
        test_name = "mixed_user_scenario"
        performance_monitor.start_monitoring(test_name)
        
        user_profiles = list(test_user_profiles.keys())
        
        try:
            # Step 1: 複数ユーザーの並行セッション
            user_tasks = []
            
            for user_type in user_profiles:
                user_id = test_user_profiles[user_type]['user_id']
                task = self._simulate_user_session(
                    system_under_test, user_id, user_type, performance_monitor, test_name
                )
                user_tasks.append(task)
            
            # 並行実行
            start_time = time.time()
            session_results = await asyncio.gather(*user_tasks, return_exceptions=True)
            total_session_time = (time.time() - start_time) * 1000
            
            successful_sessions = sum(1 for result in session_results if not isinstance(result, Exception))
            performance_monitor.record_operation(
                test_name, "concurrent_user_sessions", total_session_time,
                successful_sessions == len(user_profiles)
            )
            
            # Step 2: システム状態検証
            system_stats = system_under_test.get_system_stats()
            
            performance_monitor.stop_monitoring(test_name)
            summary = performance_monitor.get_summary(test_name)
            
            # 混合シナリオ検証
            assert successful_sessions == len(user_profiles), \
                f"Some user sessions failed: {successful_sessions}/{len(user_profiles)}"
            assert summary['success_rate'] >= 0.85, \
                f"Mixed scenario success rate too low: {summary['success_rate']}"
            assert system_stats['request_count'] > 50, \
                f"Insufficient system activity: {system_stats['request_count']}"
            
            print(f"✅ Mixed user scenario workflow completed:")
            print(f"   Total duration: {summary['total_duration']:.2f}s")
            print(f"   User sessions: {successful_sessions}/{len(user_profiles)}")
            print(f"   System requests: {system_stats['request_count']}")
            print(f"   Overall success rate: {summary['success_rate']:.1%}")
            
        except Exception as e:
            performance_monitor.record_operation(test_name, "workflow_error", 0, False)
            raise
    
    async def _simulate_user_session(
        self, 
        system_under_test, 
        user_id: str, 
        user_type: str, 
        performance_monitor, 
        test_name: str
    ) -> Dict[str, Any]:
        """ユーザーセッションシミュレーション"""
        session_id = f"{user_type}_{user_id}"
        token = f"valid_token_{user_id}"
        
        # 認証
        auth_result = await system_under_test.authenticate_user(user_id, token)
        if not auth_result:
            raise Exception(f"Authentication failed for {user_id}")
        
        # ユーザータイプ別の行動パターン
        if user_type == 'new_user':
            # 新規ユーザー: 慎重な探索行動
            actions = [
                ('get_feed', {'page_size': 10}),
                ('interact', {'video_id': 'rec_video_1', 'type': 'like'}),
                ('get_feed', {'page_size': 10}),
                ('interact', {'video_id': 'rec_video_2', 'type': 'dislike'}),
                ('get_feed', {'page_size': 15})
            ]
        elif user_type == 'active_user':
            # アクティブユーザー: 通常の利用パターン
            actions = [
                ('get_feed', {'page_size': 20}),
                ('interact', {'video_id': 'rec_video_1', 'type': 'like'}),
                ('interact', {'video_id': 'rec_video_2', 'type': 'like'}),
                ('get_feed', {'page_size': 20}),
                ('interact', {'video_id': 'rec_video_3', 'type': 'dislike'}),
                ('get_feed', {'page_size': 25}),
                ('interact', {'video_id': 'rec_video_4', 'type': 'like'})
            ]
        else:  # heavy_user
            # ヘビーユーザー: 高頻度利用
            actions = [
                ('get_feed', {'page_size': 50}),
                ('interact', {'video_id': 'rec_video_1', 'type': 'like'}),
                ('interact', {'video_id': 'rec_video_2', 'type': 'like'}),
                ('interact', {'video_id': 'rec_video_3', 'type': 'like'}),
                ('get_feed', {'page_size': 50}),
                ('interact', {'video_id': 'rec_video_4', 'type': 'dislike'}),
                ('interact', {'video_id': 'rec_video_5', 'type': 'like'}),
                ('get_feed', {'page_size': 100})
            ]
        
        # アクション実行
        for action_type, params in actions:
            start_time = time.time()
            
            try:
                if action_type == 'get_feed':
                    result = await system_under_test.get_user_feed(user_id, **params)
                    success = 'error' not in result
                elif action_type == 'interact':
                    result = await system_under_test.record_user_interaction(
                        user_id, params['video_id'], params['type']
                    )
                    success = result.get('success', False)
                else:
                    success = False
                
                action_time = (time.time() - start_time) * 1000
                performance_monitor.record_operation(
                    test_name, f"{session_id}_{action_type}", action_time, success
                )
                
                # ユーザー行動間の自然な間隔
                await asyncio.sleep(0.01)
                
            except Exception as e:
                performance_monitor.record_operation(
                    test_name, f"{session_id}_{action_type}", 0, False
                )
        
        return {'user_id': user_id, 'user_type': user_type, 'completed': True}