"""
Recommendation System Performance Optimizer

推薦システムパフォーマンス最適化
- レスポンス時間最適化 (<500ms目標)
- キャッシュ戦略
- バッチ処理最適化
- クエリ最適化
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """最適化結果"""
    operation: str
    original_time_ms: float
    optimized_time_ms: float
    improvement_percent: float
    success: bool
    notes: str = ""

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    data: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

class RecommendationOptimizer:
    """推薦システム最適化"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        self.optimization_results: List[OptimizationResult] = []
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_threads'])
        
        # パフォーマンスメトリクス
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time_ms': 0.0,
            'optimization_count': 0
        }
        
        logger.info(f"Recommendation optimizer initialized with config: {self.config}")
    
    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'cache_enabled': True,
            'cache_ttl_seconds': 300,  # 5分
            'max_cache_size': 10000,
            'batch_size': 100,
            'max_threads': 4,
            'enable_query_optimization': True,
            'enable_embedding_cache': True,
            'prefetch_popular_items': True,
            'target_response_time_ms': 500,
            'optimization_thresholds': {
                'slow_query_ms': 100,
                'cache_hit_rate_min': 0.8,
                'memory_usage_max_mb': 512
            }
        }
    
    async def optimize_recommendation_request(self, user_id: str, num_recommendations: int = 10) -> Tuple[List[Dict], float]:
        """推薦リクエストの最適化処理"""
        start_time = time.time()
        
        try:
            # キャッシュチェック
            cached_result = await self._get_cached_recommendations(user_id, num_recommendations)
            if cached_result is not None:
                response_time = (time.time() - start_time) * 1000
                self._update_metrics('cache_hit', response_time)
                return cached_result, response_time
            
            # 最適化された推薦生成
            recommendations = await self._generate_optimized_recommendations(user_id, num_recommendations)
            
            # キャッシュに保存
            await self._cache_recommendations(user_id, num_recommendations, recommendations)
            
            response_time = (time.time() - start_time) * 1000
            self._update_metrics('cache_miss', response_time)
            
            return recommendations, response_time
            
        except Exception as e:
            logger.error(f"Recommendation optimization failed for user {user_id}: {e}")
            response_time = (time.time() - start_time) * 1000
            return [], response_time
    
    async def _get_cached_recommendations(self, user_id: str, num_recommendations: int) -> Optional[List[Dict]]:
        """キャッシュから推薦取得"""
        if not self.config['cache_enabled']:
            return None
        
        cache_key = f"rec_{user_id}_{num_recommendations}"
        
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if not entry.is_expired():
                    entry.access_count += 1
                    logger.debug(f"Cache hit for {cache_key}")
                    return entry.data
                else:
                    del self.cache[cache_key]
                    logger.debug(f"Cache expired for {cache_key}")
        
        return None
    
    async def _cache_recommendations(self, user_id: str, num_recommendations: int, recommendations: List[Dict]):
        """推薦結果をキャッシュ"""
        if not self.config['cache_enabled']:
            return
        
        cache_key = f"rec_{user_id}_{num_recommendations}"
        
        with self.cache_lock:
            # キャッシュサイズ制限
            if len(self.cache) >= self.config['max_cache_size']:
                self._evict_cache_entries()
            
            entry = CacheEntry(
                data=recommendations,
                timestamp=datetime.now(),
                ttl_seconds=self.config['cache_ttl_seconds']
            )
            self.cache[cache_key] = entry
            logger.debug(f"Cached recommendations for {cache_key}")
    
    def _evict_cache_entries(self):
        """キャッシュエビクション (LRU)"""
        # アクセス回数が少ない順にソート
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )
        
        # 最も使用頻度の低い25%を削除
        evict_count = len(self.cache) // 4
        for i in range(evict_count):
            cache_key, _ = sorted_entries[i]
            del self.cache[cache_key]
        
        logger.debug(f"Evicted {evict_count} cache entries")
    
    async def _generate_optimized_recommendations(self, user_id: str, num_recommendations: int) -> List[Dict]:
        """最適化された推薦生成"""
        try:
            # 並列処理での最適化
            tasks = [
                self._get_user_embedding_optimized(user_id),
                self._get_candidate_items_optimized(num_recommendations * 3),  # 3倍のキャンディデート
                self._get_user_context_optimized(user_id)
            ]
            
            user_embedding, candidate_items, user_context = await asyncio.gather(*tasks)
            
            # バッチ処理での類似度計算
            similarities = await self._calculate_similarities_batch(user_embedding, candidate_items)
            
            # ランキングと選択
            recommendations = await self._rank_and_select_optimized(
                candidate_items, similarities, user_context, num_recommendations
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Optimized recommendation generation failed: {e}")
            return []
    
    async def _get_user_embedding_optimized(self, user_id: str) -> Optional[np.ndarray]:
        """最適化されたユーザー埋め込み取得"""
        # キャッシュチェック
        cache_key = f"user_emb_{user_id}"
        if self.config['enable_embedding_cache']:
            cached_embedding = await self._get_cached_embedding(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            # 実際の実装では高速化されたクエリを実行
            # プレースホルダーとして768次元のランダムベクトル
            embedding = np.random.random(768).astype(np.float32)
            
            # キャッシュに保存
            if self.config['enable_embedding_cache']:
                await self._cache_embedding(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"User embedding optimization failed for {user_id}: {e}")
            return None
    
    async def _get_candidate_items_optimized(self, num_candidates: int) -> List[Dict]:
        """最適化されたキャンディデートアイテム取得"""
        try:
            # 人気アイテムの事前フェッチを活用
            if self.config['prefetch_popular_items']:
                popular_items = await self._get_popular_items_cached(num_candidates // 2)
            else:
                popular_items = []
            
            # 残りのキャンディデートをランダムサンプリング
            additional_needed = num_candidates - len(popular_items)
            if additional_needed > 0:
                random_items = await self._get_random_items_optimized(additional_needed)
                candidate_items = popular_items + random_items
            else:
                candidate_items = popular_items[:num_candidates]
            
            return candidate_items
            
        except Exception as e:
            logger.error(f"Candidate items optimization failed: {e}")
            return []
    
    async def _get_user_context_optimized(self, user_id: str) -> Dict[str, Any]:
        """最適化されたユーザーコンテキスト取得"""
        try:
            # 軽量なコンテキスト情報のみ取得
            context = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'preferences': await self._get_lightweight_preferences(user_id)
            }
            return context
            
        except Exception as e:
            logger.error(f"User context optimization failed for {user_id}: {e}")
            return {'user_id': user_id, 'timestamp': datetime.now().isoformat()}
    
    async def _calculate_similarities_batch(self, user_embedding: np.ndarray, candidate_items: List[Dict]) -> List[float]:
        """バッチでの類似度計算"""
        if user_embedding is None or not candidate_items:
            return []
        
        try:
            # アイテム埋め込みを一括取得
            item_embeddings = await self._get_item_embeddings_batch([item['id'] for item in candidate_items])
            
            if len(item_embeddings) == 0:
                return [0.0] * len(candidate_items)
            
            # ベクトル化された類似度計算
            similarities = np.dot(item_embeddings, user_embedding)
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Batch similarity calculation failed: {e}")
            return [0.0] * len(candidate_items)
    
    async def _get_item_embeddings_batch(self, item_ids: List[str]) -> np.ndarray:
        """アイテム埋め込みの一括取得"""
        try:
            # 実際の実装では最適化されたバッチクエリを実行
            # プレースホルダーとしてランダム埋め込み
            num_items = len(item_ids)
            embeddings = np.random.random((num_items, 768)).astype(np.float32)
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch item embeddings failed: {e}")
            return np.array([])
    
    async def _rank_and_select_optimized(
        self, 
        candidate_items: List[Dict], 
        similarities: List[float], 
        user_context: Dict, 
        num_recommendations: int
    ) -> List[Dict]:
        """最適化されたランキングと選択"""
        try:
            # 類似度スコアと組み合わせ
            scored_items = []
            for item, similarity in zip(candidate_items, similarities):
                score = similarity * self._calculate_boost_factor(item, user_context)
                scored_items.append({
                    **item,
                    'score': score,
                    'similarity': similarity
                })
            
            # スコア順にソート
            scored_items.sort(key=lambda x: x['score'], reverse=True)
            
            # 上位N件を選択
            recommendations = scored_items[:num_recommendations]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Ranking optimization failed: {e}")
            return candidate_items[:num_recommendations]
    
    def _calculate_boost_factor(self, item: Dict, user_context: Dict) -> float:
        """アイテムブーストファクター計算"""
        try:
            boost = 1.0
            
            # 新しいアイテムにブースト
            if item.get('is_new', False):
                boost *= 1.1
            
            # 高評価アイテムにブースト
            if item.get('rating', 0) > 4.0:
                boost *= 1.05
            
            return boost
            
        except Exception:
            return 1.0
    
    async def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """埋め込みキャッシュから取得"""
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if not entry.is_expired():
                    return entry.data
                else:
                    del self.cache[cache_key]
        return None
    
    async def _cache_embedding(self, cache_key: str, embedding: np.ndarray):
        """埋め込みをキャッシュ"""
        with self.cache_lock:
            if len(self.cache) >= self.config['max_cache_size']:
                self._evict_cache_entries()
            
            entry = CacheEntry(
                data=embedding,
                timestamp=datetime.now(),
                ttl_seconds=self.config['cache_ttl_seconds'] * 2  # 埋め込みは長時間キャッシュ
            )
            self.cache[cache_key] = entry
    
    async def _get_popular_items_cached(self, num_items: int) -> List[Dict]:
        """人気アイテムのキャッシュ取得"""
        cache_key = f"popular_items_{num_items}"
        cached = await self._get_cached_embedding(cache_key)
        
        if cached is not None:
            return cached
        
        # 実際の実装では人気アイテムクエリを実行
        popular_items = [
            {'id': f'popular_{i}', 'title': f'Popular Item {i}', 'rating': 4.5, 'is_new': False}
            for i in range(num_items)
        ]
        
        await self._cache_embedding(cache_key, popular_items)
        return popular_items
    
    async def _get_random_items_optimized(self, num_items: int) -> List[Dict]:
        """最適化されたランダムアイテム取得"""
        # 実際の実装では最適化されたランダムサンプリングクエリを実行
        return [
            {'id': f'random_{i}', 'title': f'Random Item {i}', 'rating': 3.5, 'is_new': True}
            for i in range(num_items)
        ]
    
    async def _get_lightweight_preferences(self, user_id: str) -> Dict[str, Any]:
        """軽量なユーザー嗜好情報取得"""
        # 実際の実装では必要最小限の嗜好情報を取得
        return {
            'genres': ['action', 'drama'],
            'rating_threshold': 3.0
        }
    
    def _update_metrics(self, operation: str, response_time_ms: float):
        """メトリクス更新"""
        self.metrics['total_requests'] += 1
        
        if operation == 'cache_hit':
            self.metrics['cache_hits'] += 1
        elif operation == 'cache_miss':
            self.metrics['cache_misses'] += 1
        
        # 指数移動平均でレスポンス時間を更新
        alpha = 0.1
        self.metrics['avg_response_time_ms'] = (
            alpha * response_time_ms + 
            (1 - alpha) * self.metrics['avg_response_time_ms']
        )
    
    async def run_performance_benchmark(self, num_users: int = 100) -> Dict[str, Any]:
        """パフォーマンスベンチマーク実行"""
        logger.info(f"Starting performance benchmark with {num_users} users")
        
        start_time = time.time()
        response_times = []
        successful_requests = 0
        
        # 並行リクエスト実行
        tasks = []
        for i in range(num_users):
            user_id = f"benchmark_user_{i}"
            task = self.optimize_recommendation_request(user_id, 10)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果分析
        for result in results:
            if isinstance(result, Exception):
                continue
            
            recommendations, response_time = result
            if len(recommendations) > 0:
                successful_requests += 1
                response_times.append(response_time)
        
        total_time = time.time() - start_time
        
        # 統計計算
        avg_response_time = np.mean(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        cache_hit_rate = self.metrics['cache_hits'] / max(self.metrics['total_requests'], 1)
        
        benchmark_result = {
            'total_users': num_users,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / num_users,
            'total_time_seconds': total_time,
            'avg_response_time_ms': avg_response_time,
            'p95_response_time_ms': p95_response_time,
            'cache_hit_rate': cache_hit_rate,
            'throughput_rps': successful_requests / total_time,
            'target_met': avg_response_time < self.config['target_response_time_ms']
        }
        
        logger.info(f"Benchmark completed: {benchmark_result}")
        return benchmark_result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化サマリー取得"""
        cache_hit_rate = self.metrics['cache_hits'] / max(self.metrics['total_requests'], 1)
        
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time_ms': self.metrics['avg_response_time_ms'],
            'cache_size': len(self.cache),
            'optimization_results': self.optimization_results,
            'target_response_time_ms': self.config['target_response_time_ms'],
            'performance_target_met': self.metrics['avg_response_time_ms'] < self.config['target_response_time_ms']
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("Cache cleared")

# ファクトリー関数
def create_recommendation_optimizer(config: Optional[Dict[str, Any]] = None) -> RecommendationOptimizer:
    """推薦システム最適化の作成"""
    return RecommendationOptimizer(config)