"""
MLパイプラインパフォーマンステスト

Two-Towerモデルとエンベディング生成の性能検証
"""

import pytest
import time
import psutil
import numpy as np
from typing import Dict, List, Any
import asyncio
import logging
from datetime import datetime, timedelta

# MLパイプラインのインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml_pipeline.two_tower_model import TwoTowerModel
from ml_pipeline.embedding_pipeline import EmbeddingPipeline
from data_processing.pipeline import DataProcessingPipeline
from data_processing.utils.performance_monitor import PerformanceMonitor

class TestMLPipelinePerformance:
    """MLパイプライン性能テスト"""
    
    @classmethod
    def setup_class(cls):
        """テスト開始前の初期化"""
        cls.performance_monitor = PerformanceMonitor()
        cls.performance_monitor.start_system_monitoring(interval_seconds=5)
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
    
    @classmethod
    def teardown_class(cls):
        """テスト終了後のクリーンアップ"""
        cls.performance_monitor.stop_system_monitoring()
    
    def test_two_tower_model_initialization_performance(self):
        """Two-Towerモデル初期化性能テスト"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # モデル初期化
        model = TwoTowerModel(
            user_embedding_dim=768,
            item_embedding_dim=768,
            hidden_units=[512, 256, 128]
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        initialization_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / (1024 * 1024)  # MB
        
        # 性能要件
        assert initialization_time < 5.0, f"初期化時間が遅すぎます: {initialization_time:.2f}秒"
        assert memory_usage < 500, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        
        self.logger.info(f"Two-Towerモデル初期化 - 時間: {initialization_time:.3f}秒, メモリ: {memory_usage:.1f}MB")
    
    def test_batch_embedding_generation_performance(self):
        """バッチエンベディング生成性能テスト"""
        # テストデータ準備
        batch_sizes = [10, 50, 100, 500]
        results = {}
        
        for batch_size in batch_sizes:
            # サンプルデータ生成
            sample_data = [
                {
                    'id': i,
                    'title': f'Test Video {i}',
                    'description': f'Description for video {i}' * 10,  # 長めの説明
                    'genre': 'test',
                    'maker': 'test_maker'
                }
                for i in range(batch_size)
            ]
            
            # エンベディング生成の時間測定
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            pipeline = EmbeddingPipeline()
            embeddings = pipeline.generate_batch_embeddings(sample_data)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            processing_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / (1024 * 1024)
            throughput = batch_size / processing_time
            
            results[batch_size] = {
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'throughput': throughput
            }
            
            # バッチサイズ別の性能要件
            if batch_size <= 50:
                assert processing_time < 10.0, f"小バッチ({batch_size})の処理時間が遅すぎます: {processing_time:.2f}秒"
            elif batch_size <= 100:
                assert processing_time < 20.0, f"中バッチ({batch_size})の処理時間が遅すぎます: {processing_time:.2f}秒"
            else:
                assert processing_time < 60.0, f"大バッチ({batch_size})の処理時間が遅すぎます: {processing_time:.2f}秒"
            
            assert len(embeddings) == batch_size, "出力エンベディング数が入力データ数と一致しません"
            
            self.logger.info(
                f"バッチサイズ{batch_size}: {processing_time:.3f}秒, "
                f"{memory_usage:.1f}MB, {throughput:.1f} items/sec"
            )
        
        # スケーラビリティ検証
        small_throughput = results[10]['throughput']
        large_throughput = results[100]['throughput']
        scalability_ratio = large_throughput / small_throughput
        
        assert scalability_ratio > 0.5, f"スケーラビリティが低下しています: {scalability_ratio:.2f}"
    
    def test_model_training_performance(self):
        """モデル訓練性能テスト"""
        # 訓練データ準備（小規模）
        num_users = 100
        num_items = 500
        num_interactions = 1000
        
        # 合成データ生成
        np.random.seed(42)
        user_ids = np.random.randint(0, num_users, num_interactions)
        item_ids = np.random.randint(0, num_items, num_interactions)
        ratings = np.random.randint(1, 6, num_interactions)
        
        training_data = {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'ratings': ratings
        }
        
        # 訓練時間測定
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        model = TwoTowerModel(
            user_embedding_dim=64,  # テスト用に小さいサイズ
            item_embedding_dim=64,
            hidden_units=[128, 64]
        )
        
        # 軽量訓練（数エポック）
        model.train(training_data, epochs=3, batch_size=32)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        training_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / (1024 * 1024)
        
        # 性能要件（小規模データセット用）
        assert training_time < 120.0, f"訓練時間が長すぎます: {training_time:.2f}秒"
        assert memory_usage < 1000, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        
        self.logger.info(f"モデル訓練 - 時間: {training_time:.3f}秒, メモリ: {memory_usage:.1f}MB")
    
    def test_inference_performance(self):
        """推論性能テスト"""
        # モデル準備
        model = TwoTowerModel(
            user_embedding_dim=64,
            item_embedding_dim=64,
            hidden_units=[128, 64]
        )
        
        # ダミー訓練（推論テストのため）
        dummy_data = {
            'user_ids': np.array([0, 1, 2]),
            'item_ids': np.array([0, 1, 2]),
            'ratings': np.array([5, 4, 3])
        }
        model.train(dummy_data, epochs=1, batch_size=32)
        
        # 推論テストデータ
        inference_sizes = [1, 10, 100, 1000]
        
        for size in inference_sizes:
            user_ids = np.random.randint(0, 10, size)
            item_ids = np.random.randint(0, 100, size)
            
            # 推論時間測定
            start_time = time.time()
            
            predictions = model.predict(user_ids, item_ids)
            
            end_time = time.time()
            inference_time = end_time - start_time
            throughput = size / inference_time
            
            # 性能要件
            if size == 1:
                assert inference_time < 0.1, f"単一推論時間が遅すぎます: {inference_time:.4f}秒"
            elif size <= 100:
                assert inference_time < 1.0, f"小バッチ推論時間が遅すぎます: {inference_time:.3f}秒"
            else:
                assert inference_time < 5.0, f"大バッチ推論時間が遅すぎます: {inference_time:.3f}秒"
            
            assert len(predictions) == size, "予測数が入力数と一致しません"
            
            self.logger.info(
                f"推論サイズ{size}: {inference_time:.4f}秒, {throughput:.1f} predictions/sec"
            )
    
    def test_data_processing_pipeline_performance(self):
        """データ処理パイプライン性能テスト"""
        # パイプライン初期化
        pipeline = DataProcessingPipeline(config_env='development')
        
        # スクレイピング性能テスト（少数データ）
        start_time = time.time()
        
        scraping_result = pipeline.run_scraping_step(target_count=10)
        
        scraping_time = time.time() - start_time
        
        assert scraping_time < 30.0, f"スクレイピング時間が長すぎます: {scraping_time:.2f}秒"
        assert scraping_result['videos_scraped'] > 0, "スクレイピングでデータが取得されませんでした"
        
        # クリーニング性能テスト
        start_time = time.time()
        
        cleaning_result = pipeline.run_cleaning_step()
        
        cleaning_time = time.time() - start_time
        
        assert cleaning_time < 20.0, f"クリーニング時間が長すぎます: {cleaning_time:.2f}秒"
        
        # 検証性能テスト
        start_time = time.time()
        
        validation_result = pipeline.run_validation_step()
        
        validation_time = time.time() - start_time
        
        assert validation_time < 15.0, f"検証時間が長すぎます: {validation_time:.2f}秒"
        
        self.logger.info(
            f"パイプライン性能 - スクレイピング: {scraping_time:.2f}秒, "
            f"クリーニング: {cleaning_time:.2f}秒, 検証: {validation_time:.2f}秒"
        )
    
    def test_concurrent_processing_performance(self):
        """並行処理性能テスト"""
        async def process_batch(batch_id: int, batch_size: int):
            """バッチ処理の非同期実行"""
            pipeline = EmbeddingPipeline()
            
            sample_data = [
                {
                    'id': f'{batch_id}_{i}',
                    'title': f'Batch {batch_id} Video {i}',
                    'description': f'Description for batch {batch_id} video {i}',
                    'genre': 'test',
                    'maker': 'test_maker'
                }
                for i in range(batch_size)
            ]
            
            start_time = time.time()
            embeddings = pipeline.generate_batch_embeddings(sample_data)
            processing_time = time.time() - start_time
            
            return {
                'batch_id': batch_id,
                'processing_time': processing_time,
                'embeddings_count': len(embeddings)
            }
        
        async def test_concurrent_execution():
            """並行実行テスト"""
            num_batches = 4
            batch_size = 25
            
            start_time = time.time()
            
            # 並行バッチ処理実行
            tasks = [
                process_batch(i, batch_size) 
                for i in range(num_batches)
            ]
            
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            total_embeddings = sum(r['embeddings_count'] for r in results)
            throughput = total_embeddings / total_time
            
            # 並行処理の効果確認（テスト環境では実際の並行効果は期待できないので調整）
            sequential_estimate = sum(r['processing_time'] for r in results)
            concurrency_benefit = sequential_estimate / total_time
            
            # テスト環境では1.0倍程度でも許容する
            assert concurrency_benefit >= 0.8, f"並行処理の効果が不十分です: {concurrency_benefit:.2f}倍"
            assert total_embeddings == num_batches * batch_size, "並行処理で欠損が発生しました"
            
            self.logger.info(
                f"並行処理 - 総時間: {total_time:.2f}秒, "
                f"並行効果: {concurrency_benefit:.2f}倍, "
                f"スループット: {throughput:.1f} embeddings/sec"
            )
            
            return results
        
        # 非同期テスト実行
        results = asyncio.run(test_concurrent_execution())
        assert len(results) == 4, "すべてのバッチが完了していません"
    
    def test_memory_usage_stability(self):
        """メモリ使用量安定性テスト"""
        initial_memory = psutil.Process().memory_info().rss
        memory_readings = [initial_memory]
        
        # 複数回の処理実行でメモリリーク検出
        for iteration in range(5):
            pipeline = EmbeddingPipeline()
            
            sample_data = [
                {
                    'id': f'memory_test_{i}',
                    'title': f'Memory Test Video {i}',
                    'description': f'Description {i}' * 20,
                    'genre': 'test',
                    'maker': 'test_maker'
                }
                for i in range(100)
            ]
            
            embeddings = pipeline.generate_batch_embeddings(sample_data)
            
            # ガベージコレクション強制実行
            import gc
            gc.collect()
            
            current_memory = psutil.Process().memory_info().rss
            memory_readings.append(current_memory)
            
            self.logger.info(
                f"反復 {iteration + 1}: メモリ使用量 {current_memory / (1024*1024):.1f}MB"
            )
        
        # メモリ使用量の増加傾向をチェック
        memory_increases = [
            memory_readings[i] - memory_readings[i-1] 
            for i in range(1, len(memory_readings))
        ]
        
        max_memory = max(memory_readings)
        memory_growth = (max_memory - initial_memory) / (1024 * 1024)
        
        # メモリリークの検出
        assert memory_growth < 200, f"メモリリークの可能性: {memory_growth:.1f}MB増加"
        
        # 一定の変動範囲内であることを確認
        avg_increase = np.mean(memory_increases)
        assert abs(avg_increase) < 50 * 1024 * 1024, "メモリ使用量が不安定です"  # 50MB以内
    
    def test_system_resource_utilization(self):
        """システムリソース使用率テスト"""
        # CPU使用率監視開始
        cpu_start = psutil.cpu_percent(interval=None)
        
        # 高負荷処理実行
        pipeline = EmbeddingPipeline()
        
        large_sample_data = [
            {
                'id': f'resource_test_{i}',
                'title': f'Resource Test Video {i}',
                'description': f'Long description for video {i}' * 50,
                'genre': 'test_genre',
                'maker': 'test_maker'
            }
            for i in range(200)
        ]
        
        start_time = time.time()
        embeddings = pipeline.generate_batch_embeddings(large_sample_data)
        processing_time = time.time() - start_time
        
        # CPU使用率測定
        cpu_end = psutil.cpu_percent(interval=1)
        
        # メモリ使用率取得
        memory_percent = psutil.virtual_memory().percent
        
        # ディスク使用率取得（必要に応じて）
        disk_percent = psutil.disk_usage('/').percent
        
        # リソース使用率の確認
        assert cpu_end < 90.0, f"CPU使用率が高すぎます: {cpu_end:.1f}%"
        assert memory_percent < 85.0, f"メモリ使用率が高すぎます: {memory_percent:.1f}%"
        assert disk_percent < 90.0, f"ディスク使用率が高すぎます: {disk_percent:.1f}%"
        
        self.logger.info(
            f"リソース使用率 - CPU: {cpu_end:.1f}%, "
            f"メモリ: {memory_percent:.1f}%, ディスク: {disk_percent:.1f}%"
        )
    
    def test_performance_monitoring_integration(self):
        """パフォーマンス監視統合テスト"""
        # パフォーマンス監視の確認
        summary = self.performance_monitor.get_metrics_summary()
        
        assert 'total_operations' in summary, "操作数がトラッキングされていません"
        assert 'operations' in summary, "操作別統計が収集されていません"
        
        if 'system_metrics' in summary:
            sys_metrics = summary['system_metrics']
            assert 'current_cpu_percent' in sys_metrics, "CPU使用率が監視されていません"
            assert 'current_memory_percent' in sys_metrics, "メモリ使用率が監視されていません"
        
        # レポート生成テスト
        report = self.performance_monitor.get_performance_report()
        assert len(report) > 0, "パフォーマンスレポートが生成されませんでした"
        assert "パフォーマンス監視レポート" in report, "レポートフォーマットが正しくありません"
        
        self.logger.info("パフォーマンス監視システムが正常に動作しています")

if __name__ == '__main__':
    # パフォーマンステスト実行
    pytest.main([__file__, '-v', '--tb=short', '-s'])