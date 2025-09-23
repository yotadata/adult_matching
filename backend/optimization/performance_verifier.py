"""
パフォーマンス検証とベンチマークテスト

最適化されたコンポーネントの性能を検証し、
目標性能(<500ms推薦、<2時間トレーニング)の達成を確認
"""

import asyncio
import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor

from .recommendation_optimizer import create_recommendation_optimizer
from .ml_training_optimizer import create_ml_training_optimizer
from .database_optimizer import create_database_optimizer


@dataclass
class PerformanceTarget:
    """パフォーマンス目標"""
    recommendation_latency_ms: float = 500.0
    training_time_hours: float = 2.0
    database_query_ms: float = 100.0
    throughput_requests_per_second: float = 100.0
    memory_usage_mb: float = 2048.0


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    test_name: str
    target_value: float
    actual_value: float
    unit: str
    passed: bool
    improvement_ratio: Optional[float] = None
    details: Dict[str, Any] = None


@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""
    test_timestamp: str
    targets: PerformanceTarget
    results: List[BenchmarkResult]
    overall_passed: bool
    summary: Dict[str, Any]


class PerformanceVerifier:
    """パフォーマンス検証システム"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.targets = PerformanceTarget()
        
        # 最適化コンポーネント
        self.recommendation_optimizer = None
        self.ml_training_optimizer = None
        self.database_optimizer = None
        
    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'benchmark_iterations': 10,
            'warmup_iterations': 3,
            'concurrent_users': 50,
            'test_data_size': 1000,
            'training_sample_size': 10000,
            'timeout_seconds': 300,
            'database_url': 'postgresql://localhost:5432/postgres'
        }
    
    async def initialize_optimizers(self):
        """最適化コンポーネント初期化"""
        self.recommendation_optimizer = create_recommendation_optimizer(self.config)
        self.ml_training_optimizer = create_ml_training_optimizer(self.config)
        self.database_optimizer = create_database_optimizer(self.config)
        
        # データベース最適化の実行
        await self.database_optimizer.run_database_optimization()
        
        self.logger.info("Performance optimizers initialized")
    
    async def benchmark_recommendation_latency(self) -> BenchmarkResult:
        """推薦レスポンス時間ベンチマーク"""
        self.logger.info("Starting recommendation latency benchmark...")
        
        if not self.recommendation_optimizer:
            await self.initialize_optimizers()
        
        # ウォームアップ
        for _ in range(self.config['warmup_iterations']):
            await self.recommendation_optimizer.optimize_recommendation_request(
                "test_user", 10
            )
        
        # ベンチマーク実行
        latencies = []
        for i in range(self.config['benchmark_iterations']):
            start_time = time.time()
            
            results, optimization_time = await self.recommendation_optimizer.optimize_recommendation_request(
                f"test_user_{i}", 10
            )
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        passed = avg_latency <= self.targets.recommendation_latency_ms
        
        return BenchmarkResult(
            test_name="Recommendation Latency",
            target_value=self.targets.recommendation_latency_ms,
            actual_value=avg_latency,
            unit="ms",
            passed=passed,
            details={
                'average_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'all_latencies': latencies
            }
        )
    
    async def benchmark_training_time(self) -> BenchmarkResult:
        """MLトレーニング時間ベンチマーク"""
        self.logger.info("Starting ML training time benchmark...")
        
        if not self.ml_training_optimizer:
            await self.initialize_optimizers()
        
        # トレーニングデータサイズを調整
        training_data_size = self.config['training_sample_size']
        model_params = {
            'embedding_dim': 768,
            'batch_size': 512,
            'epochs': 10
        }
        
        start_time = time.time()
        
        # 最適化されたトレーニング実行
        training_result = await self.ml_training_optimizer.optimize_training_pipeline(
            training_data_size, model_params
        )
        
        training_time_hours = (time.time() - start_time) / 3600
        
        passed = training_time_hours <= self.targets.training_time_hours
        
        return BenchmarkResult(
            test_name="ML Training Time",
            target_value=self.targets.training_time_hours,
            actual_value=training_time_hours,
            unit="hours",
            passed=passed,
            details={
                'training_time_seconds': (time.time() - start_time),
                'data_loading_optimization': training_result.get('data_loading_time', 0),
                'batch_processing_optimization': training_result.get('batch_processing_time', 0),
                'memory_optimization': training_result.get('memory_usage_mb', 0),
                'parallel_processing_optimization': training_result.get('parallel_efficiency', 0)
            }
        )
    
    async def benchmark_database_performance(self) -> BenchmarkResult:
        """データベースパフォーマンスベンチマーク"""
        self.logger.info("Starting database performance benchmark...")
        
        if not self.database_optimizer:
            await self.initialize_optimizers()
        
        # ベクター検索ベンチマーク
        test_embedding = [0.1] * 768
        query_times = []
        
        # ウォームアップ
        for _ in range(self.config['warmup_iterations']):
            await self.database_optimizer.optimize_vector_queries(test_embedding, 10)
        
        # ベンチマーク実行
        for _ in range(self.config['benchmark_iterations']):
            results, query_time = await self.database_optimizer.optimize_vector_queries(
                test_embedding, 10
            )
            query_times.append(query_time * 1000)  # ms変換
        
        avg_query_time = statistics.mean(query_times)
        passed = avg_query_time <= self.targets.database_query_ms
        
        return BenchmarkResult(
            test_name="Database Query Performance",
            target_value=self.targets.database_query_ms,
            actual_value=avg_query_time,
            unit="ms",
            passed=passed,
            details={
                'average_query_time_ms': avg_query_time,
                'min_query_time_ms': min(query_times),
                'max_query_time_ms': max(query_times),
                'all_query_times': query_times
            }
        )
    
    async def benchmark_throughput(self) -> BenchmarkResult:
        """スループットベンチマーク"""
        self.logger.info("Starting throughput benchmark...")
        
        if not self.recommendation_optimizer:
            await self.initialize_optimizers()
        
        concurrent_users = self.config['concurrent_users']
        test_duration = 60  # 60秒
        
        async def make_recommendation_request(user_id: str):
            """推薦リクエスト実行"""
            try:
                results, _ = await self.recommendation_optimizer.optimize_recommendation_request(
                    user_id, 10
                )
                return 1  # 成功
            except Exception:
                return 0  # 失敗
        
        # 並行リクエスト実行
        start_time = time.time()
        completed_requests = 0
        
        while time.time() - start_time < test_duration:
            # 並行リクエスト生成
            tasks = [
                make_recommendation_request(f"user_{i}")
                for i in range(concurrent_users)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            completed_requests += sum(r for r in results if isinstance(r, int))
        
        actual_duration = time.time() - start_time
        throughput = completed_requests / actual_duration
        
        passed = throughput >= self.targets.throughput_requests_per_second
        
        return BenchmarkResult(
            test_name="System Throughput",
            target_value=self.targets.throughput_requests_per_second,
            actual_value=throughput,
            unit="requests/second",
            passed=passed,
            details={
                'total_requests': completed_requests,
                'test_duration_seconds': actual_duration,
                'concurrent_users': concurrent_users
            }
        )
    
    async def run_comprehensive_benchmark(self) -> PerformanceReport:
        """包括的ベンチマーク実行"""
        self.logger.info("Starting comprehensive performance benchmark...")
        
        start_time = time.time()
        
        # 最適化コンポーネント初期化
        await self.initialize_optimizers()
        
        # 各ベンチマーク実行
        results = []
        
        try:
            # 1. 推薦レスポンス時間
            recommendation_result = await self.benchmark_recommendation_latency()
            results.append(recommendation_result)
            
            # 2. データベースパフォーマンス
            database_result = await self.benchmark_database_performance()
            results.append(database_result)
            
            # 3. スループット
            throughput_result = await self.benchmark_throughput()
            results.append(throughput_result)
            
            # 4. MLトレーニング時間 (時間がかかるので最後に実行)
            training_result = await self.benchmark_training_time()
            results.append(training_result)
            
        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            raise
        
        # 結果集計
        overall_passed = all(result.passed for result in results)
        
        benchmark_duration = time.time() - start_time
        
        summary = {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r.passed),
            'failed_tests': sum(1 for r in results if not r.passed),
            'benchmark_duration_minutes': benchmark_duration / 60,
            'performance_targets_met': overall_passed
        }
        
        report = PerformanceReport(
            test_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            targets=self.targets,
            results=results,
            overall_passed=overall_passed,
            summary=summary
        )
        
        self.logger.info(f"Comprehensive benchmark completed in {benchmark_duration/60:.2f} minutes")
        return report
    
    def generate_performance_report(self, report: PerformanceReport) -> str:
        """パフォーマンスレポート生成"""
        report_lines = [
            "=" * 80,
            "PERFORMANCE OPTIMIZATION VERIFICATION REPORT",
            "=" * 80,
            f"Test Timestamp: {report.test_timestamp}",
            f"Overall Result: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}",
            "",
            "PERFORMANCE TARGETS:",
            f"  • Recommendation Latency: ≤ {report.targets.recommendation_latency_ms} ms",
            f"  • ML Training Time: ≤ {report.targets.training_time_hours} hours",
            f"  • Database Query Time: ≤ {report.targets.database_query_ms} ms",
            f"  • System Throughput: ≥ {report.targets.throughput_requests_per_second} req/s",
            "",
            "BENCHMARK RESULTS:",
            "-" * 50
        ]
        
        for result in report.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            improvement = ""
            if result.improvement_ratio:
                improvement = f" ({result.improvement_ratio:.1f}x improvement)"
            
            report_lines.extend([
                f"{result.test_name}: {status}",
                f"  Target: {result.target_value} {result.unit}",
                f"  Actual: {result.actual_value:.2f} {result.unit}{improvement}",
                ""
            ])
        
        report_lines.extend([
            "SUMMARY:",
            "-" * 50,
            f"Total Tests: {report.summary['total_tests']}",
            f"Passed: {report.summary['passed_tests']}",
            f"Failed: {report.summary['failed_tests']}",
            f"Test Duration: {report.summary['benchmark_duration_minutes']:.1f} minutes",
            "",
            "PERFORMANCE OPTIMIZATION STATUS:",
            f"{'✅ All performance targets met!' if report.overall_passed else '❌ Some performance targets not met.'}",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    async def close(self):
        """リソースクリーンアップ"""
        if self.database_optimizer:
            await self.database_optimizer.close()
        self.logger.info("Performance verifier closed")


async def main():
    """パフォーマンス検証実行"""
    logging.basicConfig(level=logging.INFO)
    
    verifier = PerformanceVerifier()
    
    try:
        # 包括的ベンチマーク実行
        report = await verifier.run_comprehensive_benchmark()
        
        # レポート出力
        report_text = verifier.generate_performance_report(report)
        print(report_text)
        
        # JSON形式でも出力
        json_report = {
            'timestamp': report.test_timestamp,
            'targets': asdict(report.targets),
            'results': [asdict(r) for r in report.results],
            'overall_passed': report.overall_passed,
            'summary': report.summary
        }
        
        with open('performance_verification_report.json', 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"\nDetailed report saved to: performance_verification_report.json")
        
    except Exception as e:
        print(f"Performance verification failed: {e}")
        raise
    finally:
        await verifier.close()


if __name__ == "__main__":
    asyncio.run(main())