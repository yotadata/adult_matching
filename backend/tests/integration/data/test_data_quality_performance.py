"""
Data Quality and Performance Integration Tests

データ品質・パフォーマンス統合テスト - 大規模データでの品質・性能検証
"""

import pytest
import time
import psutil
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from tests.integration.data.mock_processors import UnifiedDataProcessor, DataQualityMonitor


@pytest.fixture
def large_dataset():
    """大規模テストデータセット"""
    np.random.seed(42)
    
    # 10万件の動画データ
    videos = []
    for i in range(100000):
        videos.append({
            'external_id': f'video_{i:06d}',
            'source': 'dmm',
            'title': f'テスト動画 {i}',
            'description': f'これは動画{i}の説明です。' * np.random.randint(1, 5),
            'maker': f'メーカー{np.random.randint(1, 100)}',
            'genre': np.random.choice(['アクション', 'ドラマ', 'コメディ', 'ホラー', 'SF']),
            'price': np.random.randint(100, 5000),
            'thumbnail_url': f'https://example.com/thumb_{i}.jpg',
            'duration_seconds': np.random.randint(600, 7200),  # 10分-2時間
            'release_date': (datetime.now() - timedelta(days=np.random.randint(1, 1000))).isoformat(),
            'rating': np.random.uniform(1.0, 5.0),
            'review_count': np.random.randint(0, 1000),
            'tags': [f'tag{j}' for j in np.random.choice(range(1, 50), size=np.random.randint(1, 8))]
        })
    
    # 1万件のユーザーデータ
    users = []
    for i in range(10000):
        users.append({
            'user_id': f'user_{i:06d}',
            'age': np.random.randint(18, 70),
            'gender': np.random.choice(['M', 'F']),
            'prefecture': np.random.choice(['東京都', '大阪府', '神奈川県', '愛知県', '福岡県']),
            'occupation': np.random.choice(['会社員', '学生', '自営業', 'その他']),
            'interests': np.random.choice(['アクション', 'ドラマ', 'コメディ'], size=np.random.randint(1, 3)).tolist(),
            'signup_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat()
        })
    
    return {
        'videos': videos,
        'users': users
    }


@pytest.fixture
def performance_config():
    """パフォーマンステスト設定"""
    return {
        'max_processing_time_seconds': 300,  # 5分以内
        'max_memory_usage_mb': 2048,  # 2GB以内
        'min_throughput_records_per_second': 100,
        'quality_threshold': 0.85,
        'batch_size': 1000,
        'max_workers': 4
    }


@pytest.fixture
def temp_performance_dir():
    """一時パフォーマンステストディレクトリ"""
    temp_dir = tempfile.mkdtemp(prefix="performance_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestDataQualityPerformance:
    """データ品質・パフォーマンス統合テストクラス"""
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.performance
    def test_large_dataset_processing_performance(self, large_dataset, performance_config, temp_performance_dir):
        """大規模データセット処理パフォーマンステスト"""
        processor = UnifiedDataProcessor()
        
        # メモリ使用量測定開始
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 処理時間測定開始
        start_time = time.time()
        
        # 大規模データ処理実行
        processing_result = processor.process_large_dataset(
            data=large_dataset,
            batch_size=performance_config['batch_size'],
            max_workers=performance_config['max_workers']
        )
        
        # 処理時間測定終了
        processing_time = time.time() - start_time
        
        # メモリ使用量測定終了
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        # パフォーマンス要件検証
        assert processing_time <= performance_config['max_processing_time_seconds'], \
            f"処理時間が上限超過: {processing_time:.2f}秒 > {performance_config['max_processing_time_seconds']}秒"
        
        assert memory_usage <= performance_config['max_memory_usage_mb'], \
            f"メモリ使用量が上限超過: {memory_usage:.2f}MB > {performance_config['max_memory_usage_mb']}MB"
        
        # スループット計算
        total_records = len(large_dataset['videos']) + len(large_dataset['users'])
        throughput = total_records / processing_time
        
        assert throughput >= performance_config['min_throughput_records_per_second'], \
            f"スループットが下限未満: {throughput:.2f} records/sec < {performance_config['min_throughput_records_per_second']}"
        
        # 処理結果検証
        assert processing_result['success'] == True
        assert processing_result['total_processed'] == total_records
        assert processing_result['processing_errors'] == 0
        
        print(f"パフォーマンステスト結果:")
        print(f"  処理時間: {processing_time:.2f}秒")
        print(f"  メモリ使用量: {memory_usage:.2f}MB")
        print(f"  スループット: {throughput:.2f} records/sec")
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.quality
    def test_data_quality_at_scale(self, large_dataset, performance_config):
        """大規模データでの品質検証テスト"""
        quality_monitor = QualityMonitor()
        validator = DataValidator()
        
        # 品質検証実行
        start_time = time.time()
        
        quality_report = quality_monitor.assess_dataset_quality(
            dataset=large_dataset,
            sampling_rate=0.1  # 10%サンプリングで高速化
        )
        
        validation_time = time.time() - start_time
        
        # 品質スコア検証
        overall_quality = quality_report['overall_score']
        assert overall_quality >= performance_config['quality_threshold'], \
            f"データ品質が基準未満: {overall_quality:.3f} < {performance_config['quality_threshold']}"
        
        # 品質メトリクス詳細検証
        metrics = quality_report['metrics']
        
        # 完全性（必須フィールドの存在率）
        assert metrics['completeness'] >= 0.95, f"完全性が低い: {metrics['completeness']:.3f}"
        
        # 正確性（データ形式の正しさ）
        assert metrics['accuracy'] >= 0.90, f"正確性が低い: {metrics['accuracy']:.3f}"
        
        # 一意性（重複データの少なさ）
        assert metrics['uniqueness'] >= 0.98, f"一意性が低い: {metrics['uniqueness']:.3f}"
        
        # 一貫性（データ間の整合性）
        assert metrics['consistency'] >= 0.85, f"一貫性が低い: {metrics['consistency']:.3f}"
        
        # 妥当性（ビジネスルール適合）
        assert metrics['validity'] >= 0.90, f"妥当性が低い: {metrics['validity']:.3f}"
        
        # 品質検証時間の確認
        assert validation_time < 60.0, f"品質検証時間が長すぎ: {validation_time:.2f}秒"
        
        print(f"品質検証結果:")
        print(f"  総合品質スコア: {overall_quality:.3f}")
        print(f"  完全性: {metrics['completeness']:.3f}")
        print(f"  正確性: {metrics['accuracy']:.3f}")
        print(f"  一意性: {metrics['uniqueness']:.3f}")
        print(f"  一貫性: {metrics['consistency']:.3f}")
        print(f"  妥当性: {metrics['validity']:.3f}")
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.performance
    def test_concurrent_data_processing(self, large_dataset, performance_config, temp_performance_dir):
        """並行データ処理パフォーマンステスト"""
        processor = UnifiedDataProcessor()
        
        # データセットを分割
        chunk_size = len(large_dataset['videos']) // 4
        video_chunks = [
            large_dataset['videos'][i:i+chunk_size] 
            for i in range(0, len(large_dataset['videos']), chunk_size)
        ]
        
        user_chunk_size = len(large_dataset['users']) // 4
        user_chunks = [
            large_dataset['users'][i:i+user_chunk_size]
            for i in range(0, len(large_dataset['users']), user_chunk_size)
        ]
        
        # 並行処理実行
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=performance_config['max_workers']) as executor:
            # 動画処理タスク
            video_futures = [
                executor.submit(processor.process_video_chunk, chunk)
                for chunk in video_chunks
            ]
            
            # ユーザー処理タスク
            user_futures = [
                executor.submit(processor.process_user_chunk, chunk)
                for chunk in user_chunks
            ]
            
            # 結果収集
            video_results = [future.result() for future in video_futures]
            user_results = [future.result() for future in user_futures]
        
        concurrent_time = time.time() - start_time
        
        # 並行処理結果検証
        total_video_processed = sum(result['processed_count'] for result in video_results)
        total_user_processed = sum(result['processed_count'] for result in user_results)
        
        assert total_video_processed == len(large_dataset['videos'])
        assert total_user_processed == len(large_dataset['users'])
        
        # 並行処理効率検証
        total_records = total_video_processed + total_user_processed
        concurrent_throughput = total_records / concurrent_time
        
        assert concurrent_throughput >= performance_config['min_throughput_records_per_second'] * 2, \
            f"並行処理スループットが不十分: {concurrent_throughput:.2f} records/sec"
        
        # エラー率確認
        total_errors = sum(result['error_count'] for result in video_results + user_results)
        error_rate = total_errors / total_records
        
        assert error_rate <= 0.01, f"エラー率が高すぎ: {error_rate:.3f}"
        
        print(f"並行処理パフォーマンス:")
        print(f"  処理時間: {concurrent_time:.2f}秒")
        print(f"  スループット: {concurrent_throughput:.2f} records/sec")
        print(f"  エラー率: {error_rate:.3f}")
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.stress
    def test_memory_efficiency_stress(self, performance_config, temp_performance_dir):
        """メモリ効率ストレステスト"""
        processor = UnifiedDataProcessor()
        
        # メモリ使用量監視
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        max_memory_used = initial_memory
        
        memory_samples = []
        
        def monitor_memory():
            nonlocal max_memory_used
            while True:
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory_used = max(max_memory_used, current_memory)
                memory_samples.append(current_memory)
                time.sleep(0.1)  # 100ms間隔で監視
        
        # メモリ監視開始
        import threading
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            # 段階的にデータサイズを増加
            data_sizes = [10000, 50000, 100000, 200000]
            
            for size in data_sizes:
                print(f"データサイズ {size} でのテスト開始")
                
                # 大量データ生成
                large_data = {
                    'videos': [
                        {
                            'external_id': f'stress_video_{i}',
                            'title': f'ストレステスト動画 {i}',
                            'description': 'ストレステスト用のサンプルデータ' * 10,
                            'data': 'x' * 1000  # 1KB のダミーデータ
                        }
                        for i in range(size)
                    ]
                }
                
                # 処理実行
                start_time = time.time()
                result = processor.process_streaming_data(large_data)
                processing_time = time.time() - start_time
                
                # メモリ使用量確認
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                assert memory_increase <= performance_config['max_memory_usage_mb'], \
                    f"メモリ使用量上限超過 (サイズ{size}): {memory_increase:.2f}MB"
                
                print(f"  処理時間: {processing_time:.2f}秒")
                print(f"  メモリ増加: {memory_increase:.2f}MB")
                
                # メモリ解放確認
                del large_data
                import gc
                gc.collect()
                
                time.sleep(1)  # メモリ解放待機
        
        finally:
            # メモリ監視終了
            monitor_thread.join(timeout=1)
        
        # メモリ使用パターン分析
        if memory_samples:
            avg_memory = np.mean(memory_samples)
            peak_memory = np.max(memory_samples)
            memory_variance = np.var(memory_samples)
            
            print(f"メモリ使用統計:")
            print(f"  平均使用量: {avg_memory:.2f}MB")
            print(f"  ピーク使用量: {peak_memory:.2f}MB")
            print(f"  使用量分散: {memory_variance:.2f}")
            
            # メモリリーク検出
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            assert memory_trend <= 0.1, f"メモリリークの可能性: 傾向{memory_trend:.3f}MB/sample"
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.reliability
    def test_data_processing_reliability(self, large_dataset, performance_config):
        """データ処理信頼性テスト"""
        processor = UnifiedDataProcessor()
        
        # 異常データの注入
        corrupted_dataset = large_dataset.copy()
        
        # 5%のデータを破損
        corruption_count = len(corrupted_dataset['videos']) // 20
        corruption_indices = np.random.choice(
            len(corrupted_dataset['videos']), 
            size=corruption_count, 
            replace=False
        )
        
        for idx in corruption_indices:
            video = corrupted_dataset['videos'][idx]
            
            # ランダムな破損パターン
            corruption_type = np.random.choice(['missing_field', 'invalid_type', 'extreme_value'])
            
            if corruption_type == 'missing_field':
                video.pop('title', None)
            elif corruption_type == 'invalid_type':
                video['price'] = 'invalid_price'
            elif corruption_type == 'extreme_value':
                video['duration_seconds'] = -999999
        
        # 破損データでの処理実行
        start_time = time.time()
        
        try:
            processing_result = processor.process_with_error_handling(
                data=corrupted_dataset,
                continue_on_error=True
            )
            
            processing_time = time.time() - start_time
            
            # 信頼性検証
            assert processing_result['success'] == True
            
            # 処理されたレコード数の確認
            total_input = len(corrupted_dataset['videos']) + len(corrupted_dataset['users'])
            processed_count = processing_result['processed_count']
            
            # 90%以上が正常に処理されることを確認
            success_rate = processed_count / total_input
            assert success_rate >= 0.9, f"処理成功率が低い: {success_rate:.3f}"
            
            # エラーハンドリングの確認
            error_count = processing_result['error_count']
            assert error_count > 0, "破損データがあるのにエラーが検出されていない"
            assert error_count <= corruption_count * 1.1, "予想以上のエラー数"
            
            # エラー分類の確認
            error_types = processing_result['error_breakdown']
            assert 'missing_field' in error_types
            assert 'invalid_type' in error_types
            assert 'validation_error' in error_types
            
            print(f"信頼性テスト結果:")
            print(f"  処理成功率: {success_rate:.3f}")
            print(f"  エラー数: {error_count}")
            print(f"  処理時間: {processing_time:.2f}秒")
            
        except Exception as e:
            pytest.fail(f"信頼性テストで予期しないエラー: {str(e)}")
    
    @pytest.mark.integration
    @pytest.mark.data
    @pytest.mark.benchmark
    def test_performance_benchmark_comparison(self, large_dataset, performance_config, temp_performance_dir):
        """パフォーマンスベンチマーク比較テスト"""
        processor = UnifiedDataProcessor()
        
        # 異なる処理方式でのベンチマーク
        benchmark_results = {}
        
        # 1. シングルスレッド処理
        start_time = time.time()
        single_result = processor.process_single_threaded(large_dataset)
        benchmark_results['single_threaded'] = {
            'time': time.time() - start_time,
            'throughput': single_result['processed_count'] / (time.time() - start_time),
            'memory_peak': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        # 2. マルチスレッド処理
        start_time = time.time()
        multi_result = processor.process_multi_threaded(large_dataset, threads=4)
        benchmark_results['multi_threaded'] = {
            'time': time.time() - start_time,
            'throughput': multi_result['processed_count'] / (time.time() - start_time),
            'memory_peak': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        # 3. バッチ処理
        start_time = time.time()
        batch_result = processor.process_batched(large_dataset, batch_size=1000)
        benchmark_results['batched'] = {
            'time': time.time() - start_time,
            'throughput': batch_result['processed_count'] / (time.time() - start_time),
            'memory_peak': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        # パフォーマンス比較分析
        print("パフォーマンスベンチマーク結果:")
        
        fastest_method = min(benchmark_results.keys(), key=lambda k: benchmark_results[k]['time'])
        highest_throughput = max(benchmark_results.keys(), key=lambda k: benchmark_results[k]['throughput'])
        lowest_memory = min(benchmark_results.keys(), key=lambda k: benchmark_results[k]['memory_peak'])
        
        for method, metrics in benchmark_results.items():
            print(f"  {method}:")
            print(f"    時間: {metrics['time']:.2f}秒")
            print(f"    スループット: {metrics['throughput']:.2f} records/sec")
            print(f"    メモリピーク: {metrics['memory_peak']:.2f}MB")
        
        print(f"最速方式: {fastest_method}")
        print(f"最高スループット: {highest_throughput}")
        print(f"最低メモリ使用: {lowest_memory}")
        
        # 期待パフォーマンスの確認
        best_throughput = benchmark_results[highest_throughput]['throughput']
        assert best_throughput >= performance_config['min_throughput_records_per_second'], \
            f"最高スループットが基準未満: {best_throughput:.2f}"
        
        # マルチスレッドがシングルスレッドより高速であることを確認
        multi_time = benchmark_results['multi_threaded']['time']
        single_time = benchmark_results['single_threaded']['time']
        speedup_ratio = single_time / multi_time
        
        assert speedup_ratio >= 1.5, f"マルチスレッド処理の効果が不十分: {speedup_ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])