"""
ML Performance Tests

機械学習コンポーネントのパフォーマンステスト
"""

import pytest
import time
import psutil
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from memory_profiler import profile
import threading

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml.preprocessing.embeddings.embedding_manager import EmbeddingManager
from backend.ml.preprocessing.features.user_feature_processor import UserFeatureProcessor
from backend.ml.preprocessing.features.item_feature_processor import ItemFeatureProcessor


@pytest.mark.performance
class TestEmbeddingManagerPerformance:
    """EmbeddingManager パフォーマンステスト"""
    
    def test_similarity_computation_speed(self):
        """類似度計算速度テスト"""
        manager = EmbeddingManager()
        
        # Large embeddings for performance test
        user_embeddings = np.random.randn(1000, 768).astype(np.float32)
        item_embeddings = np.random.randn(5000, 768).astype(np.float32)
        
        start_time = time.time()
        
        # Compute similarities for all user-item pairs
        similarities = manager.batch_compute_similarity(user_embeddings, item_embeddings)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        assert similarities.shape == (1000, 5000)
        assert computation_time < 5.0  # Should complete within 5 seconds
        
        # Calculate throughput
        total_computations = 1000 * 5000
        throughput = total_computations / computation_time
        
        print(f"Similarity computation throughput: {throughput:,.0f} pairs/second")
        assert throughput > 100000  # At least 100k pairs per second
    
    def test_embedding_update_performance(self):
        """埋め込み更新パフォーマンステスト"""
        manager = EmbeddingManager()
        
        # Initialize with large embeddings
        user_data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(10000)],
            'preferences': [{'genre': 'action'}] * 10000
        })
        item_data = pd.DataFrame({
            'external_id': [f'item_{i}' for i in range(50000)],
            'title': [f'Item {i}'] * 50000,
            'genre': ['action'] * 50000
        })
        
        start_time = time.time()
        manager.initialize_embeddings(user_data, item_data)
        init_time = time.time() - start_time
        
        assert init_time < 30.0  # Should initialize within 30 seconds
        
        # Test batch updates
        batch_size = 1000
        new_embeddings = np.random.randn(batch_size, 768).astype(np.float32)
        user_ids = [f'user_{i}' for i in range(batch_size)]
        
        start_time = time.time()
        for i, user_id in enumerate(user_ids):
            manager.update_user_embedding(user_id, new_embeddings[i])
        update_time = time.time() - start_time
        
        update_rate = batch_size / update_time
        print(f"Embedding update rate: {update_rate:,.0f} updates/second")
        assert update_rate > 100  # At least 100 updates per second
    
    def test_memory_usage(self):
        """メモリ使用量テスト"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        manager = EmbeddingManager()
        
        # Large dataset
        num_users = 50000
        num_items = 100000
        
        user_data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(num_users)],
            'preferences': [{'genre': 'action'}] * num_users
        })
        item_data = pd.DataFrame({
            'external_id': [f'item_{i}' for i in range(num_items)],
            'title': [f'Item {i}'] * num_items,
            'genre': ['action'] * num_items
        })
        
        manager.initialize_embeddings(user_data, item_data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Memory usage: {memory_increase:.1f} MB for {num_users} users and {num_items} items")
        
        # Memory usage should be reasonable (less than 2GB for large dataset)
        assert memory_increase < 2048
        
        # Calculate memory efficiency
        expected_memory = (num_users + num_items) * 768 * 4 / 1024 / 1024  # 4 bytes per float32
        efficiency = expected_memory / memory_increase
        
        print(f"Memory efficiency: {efficiency:.2f}")
        assert efficiency > 0.5  # Should be at least 50% efficient
    
    @pytest.mark.slow
    def test_concurrent_similarity_computation(self):
        """並行類似度計算テスト"""
        manager = EmbeddingManager()
        
        # Setup embeddings
        num_users = 1000
        num_items = 5000
        user_embeddings = np.random.randn(num_users, 768).astype(np.float32)
        item_embeddings = np.random.randn(num_items, 768).astype(np.float32)
        
        manager.user_embeddings = user_embeddings
        manager.item_embeddings = item_embeddings
        manager.user_ids = [f'user_{i}' for i in range(num_users)]
        manager.item_ids = [f'item_{i}' for i in range(num_items)]
        
        def compute_similarities_for_user(user_idx):
            """単一ユーザーの類似度計算"""
            user_emb = user_embeddings[user_idx]
            similarities = []
            for item_emb in item_embeddings:
                sim = manager.compute_similarity(user_emb, item_emb)
                similarities.append(sim)
            return similarities
        
        # Sequential computation
        start_time = time.time()
        sequential_results = []
        for i in range(100):  # First 100 users
            result = compute_similarities_for_user(i)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Parallel computation
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(compute_similarities_for_user, range(100)))
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time
        print(f"Parallel speedup: {speedup:.2f}x")
        
        # Should achieve meaningful speedup with multiple cores
        assert speedup > 1.5
        assert len(parallel_results) == 100


@pytest.mark.performance
class TestFeatureProcessingPerformance:
    """特徴量処理パフォーマンステスト"""
    
    def test_user_feature_processing_speed(self):
        """ユーザー特徴量処理速度テスト"""
        processor = UserFeatureProcessor()
        
        # Large user dataset
        num_users = 10000
        user_data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(num_users)],
            'age': np.random.randint(18, 65, num_users),
            'preferences': [{'genre': np.random.choice(['action', 'comedy', 'drama'])} for _ in range(num_users)],
            'total_likes': np.random.randint(0, 1000, num_users)
        })
        
        start_time = time.time()
        result = processor.process_user_data(user_data)
        processing_time = time.time() - start_time
        
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        processing_rate = num_users / processing_time
        print(f"User processing rate: {processing_rate:,.0f} users/second")
        assert processing_rate > 500  # At least 500 users per second
    
    def test_item_feature_processing_speed(self):
        """アイテム特徴量処理速度テスト"""
        processor = ItemFeatureProcessor()
        
        # Large item dataset
        num_items = 50000
        item_data = pd.DataFrame({
            'external_id': [f'item_{i}' for i in range(num_items)],
            'title': [f'Item Title {i}' for i in range(num_items)],
            'duration': np.random.randint(300, 7200, num_items),
            'price': np.random.randint(100, 5000, num_items),
            'genre': np.random.choice(['action', 'comedy', 'drama'], num_items),
            'source': ['dmm'] * num_items
        })
        
        start_time = time.time()
        result = processor.process_item_data(item_data)
        processing_time = time.time() - start_time
        
        assert processing_time < 15.0  # Should complete within 15 seconds
        
        processing_rate = num_items / processing_time
        print(f"Item processing rate: {processing_rate:,.0f} items/second")
        assert processing_rate > 2000  # At least 2000 items per second
    
    @pytest.mark.slow
    def test_batch_processing_scalability(self):
        """バッチ処理スケーラビリティテスト"""
        user_processor = UserFeatureProcessor()
        item_processor = ItemFeatureProcessor()
        
        # Test different batch sizes
        batch_sizes = [1000, 5000, 10000, 20000]
        processing_times = []
        
        for batch_size in batch_sizes:
            # Generate test data
            user_data = pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(batch_size)],
                'age': np.random.randint(18, 65, batch_size),
                'preferences': [{'genre': 'action'}] * batch_size
            })
            
            item_data = pd.DataFrame({
                'external_id': [f'item_{i}' for i in range(batch_size)],
                'title': [f'Item {i}'] * batch_size,
                'genre': ['action'] * batch_size
            })
            
            # Measure processing time
            start_time = time.time()
            user_processor.process_user_data(user_data)
            item_processor.process_item_data(item_data)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            print(f"Batch size {batch_size}: {processing_time:.2f} seconds")
        
        # Check scalability - processing time should scale reasonably with data size
        for i in range(1, len(batch_sizes)):
            size_ratio = batch_sizes[i] / batch_sizes[i-1]
            time_ratio = processing_times[i] / processing_times[i-1]
            
            # Time ratio should not exceed size ratio by more than 50%
            assert time_ratio < size_ratio * 1.5
    
    def test_memory_efficiency(self):
        """メモリ効率性テスト"""
        processor = UserFeatureProcessor()
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process increasingly large datasets
        for size in [1000, 5000, 10000]:
            user_data = pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(size)],
                'age': np.random.randint(18, 65, size),
                'preferences': [{'genre': 'action'}] * size
            })
            
            processor.process_user_data(user_data)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_per_record = (current_memory - initial_memory) / size
            
            print(f"Memory per record ({size} records): {memory_per_record:.3f} MB")
            
            # Memory per record should be reasonable and not grow excessively
            assert memory_per_record < 0.1  # Less than 100KB per record


@pytest.mark.performance
class TestConcurrencyPerformance:
    """並行処理パフォーマンステスト"""
    
    @pytest.mark.asyncio
    async def test_async_processing_performance(self):
        """非同期処理パフォーマンステスト"""
        embedding_manager = EmbeddingManager()
        
        # Setup test data
        num_users = 100
        num_items = 1000
        user_embeddings = np.random.randn(num_users, 768).astype(np.float32)
        item_embeddings = np.random.randn(num_items, 768).astype(np.float32)
        
        embedding_manager.user_embeddings = user_embeddings
        embedding_manager.item_embeddings = item_embeddings
        embedding_manager.user_ids = [f'user_{i}' for i in range(num_users)]
        
        async def get_recommendations_for_user(user_id: str):
            """ユーザーの推薦を非同期で取得"""
            await asyncio.sleep(0.01)  # Simulate async operation
            return embedding_manager.get_similar_items(user_id, top_k=10)
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for i in range(50):  # First 50 users
            result = await get_recommendations_for_user(f'user_{i}')
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        tasks = [get_recommendations_for_user(f'user_{i}') for i in range(50)]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        speedup = sequential_time / concurrent_time
        print(f"Async speedup: {speedup:.2f}x")
        
        assert speedup > 2  # Should achieve significant speedup
        assert len(concurrent_results) == 50
    
    def test_thread_safety(self):
        """スレッドセーフティテスト"""
        embedding_manager = EmbeddingManager()
        
        # Setup shared data
        user_embeddings = np.random.randn(1000, 768).astype(np.float32)
        item_embeddings = np.random.randn(5000, 768).astype(np.float32)
        
        embedding_manager.user_embeddings = user_embeddings
        embedding_manager.item_embeddings = item_embeddings
        embedding_manager.user_ids = [f'user_{i}' for i in range(1000)]
        
        results = []
        errors = []
        
        def worker_thread(thread_id: int):
            """ワーカースレッド"""
            try:
                for i in range(100):
                    user_id = f'user_{(thread_id * 100 + i) % 1000}'
                    similar_items = embedding_manager.get_similar_items(user_id, top_k=10)
                    results.append(len(similar_items))
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads concurrently
        threads = []
        num_threads = 4
        
        start_time = time.time()
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == num_threads * 100
        
        print(f"Thread safety test completed in {end_time - start_time:.2f} seconds")
        print(f"Processed {len(results)} requests across {num_threads} threads")


@pytest.mark.performance
class TestResourceUtilization:
    """リソース使用率テスト"""
    
    def test_cpu_utilization(self):
        """CPU使用率テスト"""
        embedding_manager = EmbeddingManager()
        
        # CPU intensive operation
        large_user_embeddings = np.random.randn(5000, 768).astype(np.float32)
        large_item_embeddings = np.random.randn(10000, 768).astype(np.float32)
        
        # Monitor CPU usage during computation
        process = psutil.Process()
        cpu_percentages = []
        
        def monitor_cpu():
            """CPU使用率監視"""
            for _ in range(50):  # Monitor for 5 seconds
                cpu_percent = process.cpu_percent()
                cpu_percentages.append(cpu_percent)
                time.sleep(0.1)
        
        # Start CPU monitoring in background
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform intensive computation
        start_time = time.time()
        similarities = embedding_manager.batch_compute_similarity(
            large_user_embeddings, large_item_embeddings
        )
        computation_time = time.time() - start_time
        
        monitor_thread.join()
        
        # Analyze CPU usage
        max_cpu = max(cpu_percentages) if cpu_percentages else 0
        avg_cpu = np.mean(cpu_percentages) if cpu_percentages else 0
        
        print(f"Computation time: {computation_time:.2f} seconds")
        print(f"Max CPU usage: {max_cpu:.1f}%")
        print(f"Average CPU usage: {avg_cpu:.1f}%")
        
        # CPU should be efficiently utilized but not maxed out
        assert avg_cpu > 10  # Should use significant CPU
        assert max_cpu < 95   # But not completely max out system
    
    def test_memory_leak_detection(self):
        """メモリリーク検出テスト"""
        embedding_manager = EmbeddingManager()
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = []
        
        # Perform repeated operations that should not leak memory
        for iteration in range(10):
            # Create temporary data
            user_embeddings = np.random.randn(1000, 768).astype(np.float32)
            item_embeddings = np.random.randn(2000, 768).astype(np.float32)
            
            # Perform computations
            similarities = embedding_manager.batch_compute_similarity(
                user_embeddings, item_embeddings
            )
            
            # Delete references
            del similarities, user_embeddings, item_embeddings
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory - initial_memory)
            
            print(f"Iteration {iteration + 1}: Memory usage = {memory_samples[-1]:.1f} MB")
        
        # Check for memory leaks
        # Memory should not continuously increase
        if len(memory_samples) > 5:
            recent_avg = np.mean(memory_samples[-3:])
            early_avg = np.mean(memory_samples[:3])
            memory_growth = recent_avg - early_avg
            
            print(f"Memory growth over iterations: {memory_growth:.1f} MB")
            
            # Memory growth should be minimal (less than 100MB)
            assert memory_growth < 100, f"Potential memory leak detected: {memory_growth:.1f} MB growth"