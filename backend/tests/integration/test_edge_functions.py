#!/usr/bin/env python3
"""
Edge Functions 統合テストスイート

リファクタリング後のEdge Functionsの動作を包括的にテスト
"""

import pytest
import requests
import json
import time
import os
from typing import Dict, Any, List
from datetime import datetime

# テスト設定
SUPABASE_URL = os.getenv('SUPABASE_URL', 'http://localhost:54321')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')
TEST_USER_TOKEN = os.getenv('TEST_USER_TOKEN', '')

class EdgeFunctionsTestSuite:
    """Edge Functions統合テストクラス"""
    
    def __init__(self):
        self.base_url = f"{SUPABASE_URL}/functions/v1"
        self.headers = {
            'Authorization': f'Bearer {TEST_USER_TOKEN}',
            'apikey': SUPABASE_ANON_KEY,
            'Content-Type': 'application/json'
        }
        self.test_results = {
            'start_time': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'function_tests': {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全テストの実行"""
        print("=== Edge Functions 統合テスト開始 ===")
        
        # 各関数グループのテスト
        self._test_user_management_functions()
        self._test_content_functions()
        self._test_recommendation_functions()
        self._test_backward_compatibility()
        self._test_performance_benchmarks()
        
        # 結果サマリー
        self.test_results['end_time'] = datetime.now().isoformat()
        success_rate = (self.test_results['tests_passed'] / 
                       max(self.test_results['tests_run'], 1)) * 100
        
        print(f"\n=== テスト結果サマリー ===")
        print(f"実行: {self.test_results['tests_run']}")
        print(f"成功: {self.test_results['tests_passed']}")
        print(f"失敗: {self.test_results['tests_failed']}")
        print(f"成功率: {success_rate:.1f}%")
        
        return self.test_results
    
    def _test_user_management_functions(self):
        """ユーザー管理関数のテスト"""
        print("\n--- ユーザー管理関数テスト ---")
        
        function_name = 'user-management'
        self.test_results['function_tests'][function_name] = {
            'tests': [],
            'passed': 0,
            'failed': 0
        }
        
        # いいね機能テスト
        self._run_test(
            function_name,
            "いいね一覧取得",
            self._test_likes_list,
            '/user-management/likes',
            'GET'
        )
        
        self._run_test(
            function_name,
            "いいね追加",
            self._test_add_like,
            '/user-management/likes',
            'POST'
        )
        
        # エンベディング更新テスト
        self._run_test(
            function_name,
            "エンベディング更新",
            self._test_embedding_update,
            '/user-management/embeddings',
            'POST'
        )
        
        # ヘルスチェック
        self._run_test(
            function_name,
            "ヘルスチェック",
            self._test_health_check,
            '/user-management/health',
            'GET'
        )
    
    def _test_content_functions(self):
        """コンテンツ関数のテスト"""
        print("\n--- コンテンツ関数テスト ---")
        
        function_name = 'content'
        self.test_results['function_tests'][function_name] = {
            'tests': [],
            'passed': 0,
            'failed': 0
        }
        
        # フィード取得テスト
        self._run_test(
            function_name,
            "探索フィード取得",
            self._test_feed_generation,
            '/content/feed',
            'POST'
        )
        
        # 検索機能テスト
        self._run_test(
            function_name,
            "コンテンツ検索",
            self._test_content_search,
            '/content/search',
            'POST'
        )
    
    def _test_recommendation_functions(self):
        """推薦関数のテスト"""
        print("\n--- 推薦関数テスト ---")
        
        function_name = 'recommendations'
        self.test_results['function_tests'][function_name] = {
            'tests': [],
            'passed': 0,
            'failed': 0
        }
        
        # 推薦生成テスト
        self._run_test(
            function_name,
            "パーソナライズ推薦",
            self._test_personalized_recommendations,
            '/recommendations/enhanced_two_tower',
            'POST'
        )
    
    def _test_backward_compatibility(self):
        """後方互換性テスト"""
        print("\n--- 後方互換性テスト ---")
        
        function_name = 'backward_compatibility'
        self.test_results['function_tests'][function_name] = {
            'tests': [],
            'passed': 0,
            'failed': 0
        }
        
        # 旧API エンドポイントテスト
        self._run_test(
            function_name,
            "旧いいねAPI",
            self._test_legacy_likes_api,
            '/likes',
            'GET'
        )
        
        self._run_test(
            function_name,
            "旧推薦API",
            self._test_legacy_recommendations_api,
            '/recommendations',
            'POST'
        )
    
    def _run_test(self, function_name: str, test_name: str, 
                  test_func, endpoint: str, method: str):
        """個別テストの実行"""
        self.test_results['tests_run'] += 1
        
        try:
            start_time = time.time()
            result = test_func(endpoint, method)
            duration = time.time() - start_time
            
            if result['success']:
                self.test_results['tests_passed'] += 1
                self.test_results['function_tests'][function_name]['passed'] += 1
                status = "✓ PASS"
            else:
                self.test_results['tests_failed'] += 1
                self.test_results['function_tests'][function_name]['failed'] += 1
                status = "✗ FAIL"
            
            test_result = {
                'name': test_name,
                'status': status,
                'duration_ms': duration * 1000,
                'details': result
            }
            
            self.test_results['function_tests'][function_name]['tests'].append(test_result)
            
            print(f"  {status} {test_name} ({duration*1000:.0f}ms)")
            
        except Exception as e:
            self.test_results['tests_failed'] += 1
            self.test_results['function_tests'][function_name]['failed'] += 1
            
            error_result = {
                'name': test_name,
                'status': "✗ ERROR",
                'duration_ms': 0,
                'error': str(e)
            }
            
            self.test_results['function_tests'][function_name]['tests'].append(error_result)
            print(f"  ✗ ERROR {test_name}: {str(e)}")
    
    # ============================================================================
    # 個別テスト実装
    # ============================================================================
    
    def _test_likes_list(self, endpoint: str, method: str) -> Dict[str, Any]:
        """いいね一覧取得テスト"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={'limit': 10, 'offset': 0},
                timeout=30
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_data': response.json() if response.status_code == 200 else None,
                'error': response.text if response.status_code != 200 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_add_like(self, endpoint: str, method: str) -> Dict[str, Any]:
        """いいね追加テスト"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={'video_id': 'test_video_id'},
                timeout=30
            )
            
            return {
                'success': response.status_code in [200, 409],  # 既存いいねの場合409も許可
                'status_code': response.status_code,
                'response_data': response.json() if response.status_code in [200, 409] else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_embedding_update(self, endpoint: str, method: str) -> Dict[str, Any]:
        """エンベディング更新テスト"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={'force_update': False},
                timeout=60  # エンベディング処理は時間がかかる
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_data': response.json() if response.status_code == 200 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_feed_generation(self, endpoint: str, method: str) -> Dict[str, Any]:
        """フィード生成テスト"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={'limit': 20, 'exclude_ids': []},
                timeout=30
            )
            
            success = (response.status_code == 200 and 
                      'videos' in response.json() and
                      'diversity_metrics' in response.json())
            
            return {
                'success': success,
                'status_code': response.status_code,
                'response_data': response.json() if response.status_code == 200 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_content_search(self, endpoint: str, method: str) -> Dict[str, Any]:
        """コンテンツ検索テスト"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={
                    'query': 'test',
                    'limit': 10,
                    'filters': {'genres': ['test_genre']}
                },
                timeout=30
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_data': response.json() if response.status_code == 200 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_personalized_recommendations(self, endpoint: str, method: str) -> Dict[str, Any]:
        """パーソナライズ推薦テスト"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={
                    'limit': 20,
                    'algorithm': 'enhanced',
                    'include_reasons': True
                },
                timeout=60
            )
            
            success = (response.status_code == 200 and 
                      'videos' in response.json())
            
            return {
                'success': success,
                'status_code': response.status_code,
                'response_data': response.json() if response.status_code == 200 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_legacy_likes_api(self, endpoint: str, method: str) -> Dict[str, Any]:
        """旧いいねAPI テスト"""
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                timeout=30
            )
            
            return {
                'success': response.status_code in [200, 302],  # リダイレクトも許可
                'status_code': response.status_code
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_legacy_recommendations_api(self, endpoint: str, method: str) -> Dict[str, Any]:
        """旧推薦API テスト"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={'max_results': 10},  # 旧パラメータ形式
                timeout=60
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_health_check(self, endpoint: str, method: str) -> Dict[str, Any]:
        """ヘルスチェックテスト"""
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                timeout=10
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_performance_benchmarks(self):
        """パフォーマンスベンチマークテスト"""
        print("\n--- パフォーマンステスト ---")
        
        function_name = 'performance'
        self.test_results['function_tests'][function_name] = {
            'tests': [],
            'passed': 0,
            'failed': 0
        }
        
        # レスポンス時間テスト
        self._run_performance_test(
            function_name,
            "レスポンス時間",
            '/content/feed',
            target_ms=3000
        )
        
        # 同時リクエストテスト
        self._run_concurrent_test(
            function_name,
            "同時リクエスト処理",
            '/user-management/likes',
            concurrent_requests=10
        )
    
    def _run_performance_test(self, function_name: str, test_name: str,
                             endpoint: str, target_ms: int):
        """パフォーマンステスト実行"""
        self.test_results['tests_run'] += 1
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json={'limit': 20},
                timeout=30
            )
            duration_ms = (time.time() - start_time) * 1000
            
            success = response.status_code == 200 and duration_ms <= target_ms
            
            if success:
                self.test_results['tests_passed'] += 1
                self.test_results['function_tests'][function_name]['passed'] += 1
                status = "✓ PASS"
            else:
                self.test_results['tests_failed'] += 1
                self.test_results['function_tests'][function_name]['failed'] += 1
                status = "✗ FAIL"
            
            test_result = {
                'name': test_name,
                'status': status,
                'duration_ms': duration_ms,
                'target_ms': target_ms,
                'within_target': duration_ms <= target_ms
            }
            
            self.test_results['function_tests'][function_name]['tests'].append(test_result)
            print(f"  {status} {test_name}: {duration_ms:.0f}ms (目標: {target_ms}ms)")
            
        except Exception as e:
            self.test_results['tests_failed'] += 1
            print(f"  ✗ ERROR {test_name}: {str(e)}")
    
    def _run_concurrent_test(self, function_name: str, test_name: str,
                            endpoint: str, concurrent_requests: int):
        """同時リクエストテスト"""
        import concurrent.futures
        
        self.test_results['tests_run'] += 1
        
        def make_request():
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    json={'limit': 10},
                    timeout=30
                )
                return response.status_code == 200
            except:
                return False
        
        try:
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            duration_ms = (time.time() - start_time) * 1000
            success_count = sum(results)
            success_rate = success_count / concurrent_requests
            
            success = success_rate >= 0.8  # 80%成功で合格
            
            if success:
                self.test_results['tests_passed'] += 1
                self.test_results['function_tests'][function_name]['passed'] += 1
                status = "✓ PASS"
            else:
                self.test_results['tests_failed'] += 1
                self.test_results['function_tests'][function_name]['failed'] += 1
                status = "✗ FAIL"
            
            test_result = {
                'name': test_name,
                'status': status,
                'duration_ms': duration_ms,
                'concurrent_requests': concurrent_requests,
                'successful_requests': success_count,
                'success_rate': success_rate
            }
            
            self.test_results['function_tests'][function_name]['tests'].append(test_result)
            print(f"  {status} {test_name}: {success_count}/{concurrent_requests} 成功 ({success_rate*100:.1f}%)")
            
        except Exception as e:
            self.test_results['tests_failed'] += 1
            print(f"  ✗ ERROR {test_name}: {str(e)}")
    
    def save_test_report(self, file_path: str):
        """テストレポートの保存"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Edge Functions統合テスト')
    parser.add_argument('--output', '-o', help='結果保存ファイル')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細出力')
    
    args = parser.parse_args()
    
    # テスト実行
    test_suite = EdgeFunctionsTestSuite()
    results = test_suite.run_all_tests()
    
    # 結果保存
    if args.output:
        test_suite.save_test_report(args.output)
        print(f"\nテスト結果を保存しました: {args.output}")
    
    # 詳細出力
    if args.verbose:
        print("\n=== 詳細結果 ===")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # 終了コード
    return 0 if results['tests_failed'] == 0 else 1

if __name__ == "__main__":
    exit(main())