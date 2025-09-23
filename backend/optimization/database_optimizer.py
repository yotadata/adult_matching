"""
データベース最適化コンポーネント

データベースクエリ最適化、接続プール管理、
ベクター検索最適化などを実装
"""

import asyncio
import time
import psutil
import asyncpg
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import logging


@dataclass
class DatabaseMetrics:
    """データベースメトリクス"""
    connection_count: int
    active_queries: int
    avg_query_time: float
    cache_hit_ratio: float
    index_usage: Dict[str, float]
    vector_search_time: float
    
    
@dataclass
class ConnectionPoolConfig:
    """接続プール設定"""
    min_size: int = 5
    max_size: int = 20
    command_timeout: float = 60.0
    server_settings: Dict[str, str] = None
    
    def __post_init__(self):
        if self.server_settings is None:
            self.server_settings = {
                'jit': 'off',
                'shared_preload_libraries': 'vector',
                'max_connections': '200',
                'effective_cache_size': '2GB',
                'random_page_cost': '1.1'
            }


@dataclass
class QueryOptimization:
    """クエリ最適化結果"""
    original_query: str
    optimized_query: str
    performance_improvement: float
    execution_time_before: float
    execution_time_after: float
    optimization_techniques: List[str]


class DatabaseOptimizer:
    """データベース最適化"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.query_cache: Dict[str, Any] = {}
        self.metrics_history: List[DatabaseMetrics] = []
        
    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'database_url': 'postgresql://localhost:5432/postgres',
            'connection_pool': ConnectionPoolConfig(),
            'query_timeout': 30.0,
            'cache_size': 1000,
            'vector_index_maintenance_interval': 3600,  # 1 hour
            'monitoring_interval': 60.0,  # 1 minute
            'optimization_thresholds': {
                'slow_query_threshold': 1.0,  # seconds
                'cache_hit_ratio_min': 0.8,
                'connection_usage_max': 0.8
            }
        }
    
    async def initialize_connection_pool(self) -> asyncpg.Pool:
        """最適化された接続プール初期化"""
        pool_config = self.config['connection_pool']
        
        self.connection_pool = await asyncpg.create_pool(
            self.config['database_url'],
            min_size=pool_config.min_size,
            max_size=pool_config.max_size,
            command_timeout=pool_config.command_timeout,
            server_settings=pool_config.server_settings
        )
        
        self.logger.info(f"Database connection pool initialized: {pool_config.min_size}-{pool_config.max_size} connections")
        return self.connection_pool
    
    async def optimize_vector_queries(self, 
                                    embedding: List[float], 
                                    limit: int = 10,
                                    similarity_threshold: float = 0.7) -> Tuple[List[Dict], float]:
        """ベクター検索クエリ最適化"""
        start_time = time.time()
        
        # 最適化されたベクター検索クエリ
        optimized_query = """
        WITH vector_candidates AS (
            SELECT 
                v.id,
                v.title,
                v.embedding <=> $1::vector as distance,
                v.embedding <#> $1::vector as negative_inner_product
            FROM videos v
            WHERE v.embedding IS NOT NULL
            ORDER BY v.embedding <=> $1::vector
            LIMIT $2 * 2  -- 候補数を2倍に増やして精度向上
        ),
        filtered_results AS (
            SELECT *,
                   1.0 - distance as similarity
            FROM vector_candidates
            WHERE 1.0 - distance >= $3
        )
        SELECT 
            id,
            title,
            similarity,
            distance
        FROM filtered_results
        ORDER BY similarity DESC
        LIMIT $2;
        """
        
        if not self.connection_pool:
            await self.initialize_connection_pool()
        
        async with self.connection_pool.acquire() as conn:
            # ベクター演算子用のインデックスヒント使用
            await conn.execute("SET enable_seqscan = off;")
            await conn.execute("SET work_mem = '256MB';")
            
            results = await conn.fetch(
                optimized_query,
                embedding,
                limit,
                similarity_threshold
            )
            
            # 設定をリセット
            await conn.execute("RESET enable_seqscan;")
            await conn.execute("RESET work_mem;")
        
        execution_time = time.time() - start_time
        
        # 結果を辞書形式に変換
        formatted_results = [
            {
                'id': row['id'],
                'title': row['title'],
                'similarity': float(row['similarity']),
                'distance': float(row['distance'])
            }
            for row in results
        ]
        
        self.logger.info(f"Vector query optimized: {len(formatted_results)} results in {execution_time:.3f}s")
        return formatted_results, execution_time
    
    async def optimize_recommendation_queries(self, 
                                            user_id: str, 
                                            num_recommendations: int = 10) -> Tuple[List[Dict], float]:
        """推薦クエリ最適化"""
        start_time = time.time()
        
        # バッチ処理とインデックス最適化を組み合わせた推薦クエリ
        optimized_query = """
        WITH user_preferences AS (
            SELECT embedding, preferences
            FROM user_embeddings 
            WHERE user_id = $1
            LIMIT 1
        ),
        candidate_videos AS (
            SELECT 
                v.id,
                v.title,
                v.genre,
                v.maker,
                v.embedding,
                v.rating,
                v.view_count
            FROM videos v
            WHERE v.embedding IS NOT NULL
              AND v.id NOT IN (
                  SELECT video_id 
                  FROM likes 
                  WHERE user_id = $1
              )
        ),
        scored_recommendations AS (
            SELECT 
                cv.*,
                (1.0 - (cv.embedding <=> up.embedding)) as similarity_score,
                -- 複合スコア: 類似度 + 人気度 + 評価
                (1.0 - (cv.embedding <=> up.embedding)) * 0.6 +
                (cv.rating / 5.0) * 0.3 +
                (LEAST(cv.view_count, 10000) / 10000.0) * 0.1 as composite_score
            FROM candidate_videos cv
            CROSS JOIN user_preferences up
            WHERE (1.0 - (cv.embedding <=> up.embedding)) >= 0.6
        )
        SELECT 
            id,
            title,
            genre,
            maker,
            similarity_score,
            composite_score,
            rating,
            view_count
        FROM scored_recommendations
        ORDER BY composite_score DESC
        LIMIT $2;
        """
        
        if not self.connection_pool:
            await self.initialize_connection_pool()
        
        async with self.connection_pool.acquire() as conn:
            # クエリプランナー最適化
            await conn.execute("SET enable_nestloop = off;")
            await conn.execute("SET work_mem = '256MB';")
            await conn.execute("SET random_page_cost = 1.1;")
            
            results = await conn.fetch(
                optimized_query,
                user_id,
                num_recommendations
            )
            
            # 設定をリセット
            await conn.execute("RESET enable_nestloop;")
            await conn.execute("RESET work_mem;")
            await conn.execute("RESET random_page_cost;")
        
        execution_time = time.time() - start_time
        
        # 結果を辞書形式に変換
        formatted_results = [
            {
                'id': row['id'],
                'title': row['title'],
                'genre': row['genre'],
                'maker': row['maker'],
                'similarity_score': float(row['similarity_score']),
                'composite_score': float(row['composite_score']),
                'rating': float(row['rating']) if row['rating'] else 0.0,
                'view_count': row['view_count'] or 0
            }
            for row in results
        ]
        
        self.logger.info(f"Recommendation query optimized: {len(formatted_results)} results in {execution_time:.3f}s")
        return formatted_results, execution_time
    
    async def optimize_database_indexes(self) -> Dict[str, Any]:
        """データベースインデックス最適化"""
        if not self.connection_pool:
            await self.initialize_connection_pool()
        
        optimization_results = {
            'vector_indexes_created': [],
            'btree_indexes_created': [],
            'unused_indexes_dropped': [],
            'statistics_updated': []
        }
        
        async with self.connection_pool.acquire() as conn:
            # ベクターインデックス最適化
            vector_index_queries = [
                """
                CREATE INDEX IF NOT EXISTS idx_videos_embedding_cosine 
                ON videos USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_user_embeddings_embedding_cosine 
                ON user_embeddings USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 50);
                """
            ]
            
            for query in vector_index_queries:
                try:
                    await conn.execute(query)
                    optimization_results['vector_indexes_created'].append(query.split('\n')[1].strip())
                except Exception as e:
                    self.logger.warning(f"Vector index creation failed: {e}")
            
            # B-tree インデックス最適化
            btree_index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_videos_genre_rating ON videos(genre, rating DESC);",
                "CREATE INDEX IF NOT EXISTS idx_likes_user_video ON likes(user_id, video_id);",
                "CREATE INDEX IF NOT EXISTS idx_videos_view_count ON videos(view_count DESC);",
                "CREATE INDEX IF NOT EXISTS idx_user_embeddings_user_id ON user_embeddings(user_id);"
            ]
            
            for query in btree_index_queries:
                try:
                    await conn.execute(query)
                    optimization_results['btree_indexes_created'].append(query.split('ON')[0].split('CREATE INDEX IF NOT EXISTS')[1].strip())
                except Exception as e:
                    self.logger.warning(f"B-tree index creation failed: {e}")
            
            # 統計情報更新
            stats_queries = [
                "ANALYZE videos;",
                "ANALYZE user_embeddings;",
                "ANALYZE likes;"
            ]
            
            for query in stats_queries:
                try:
                    await conn.execute(query)
                    optimization_results['statistics_updated'].append(query.split()[1].rstrip(';'))
                except Exception as e:
                    self.logger.warning(f"Statistics update failed: {e}")
        
        self.logger.info(f"Database indexes optimized: {optimization_results}")
        return optimization_results
    
    async def monitor_database_performance(self) -> DatabaseMetrics:
        """データベースパフォーマンス監視"""
        if not self.connection_pool:
            await self.initialize_connection_pool()
        
        async with self.connection_pool.acquire() as conn:
            # 接続数監視
            connection_result = await conn.fetchrow("""
                SELECT count(*) as active_connections
                FROM pg_stat_activity 
                WHERE state = 'active';
            """)
            
            # アクティブクエリ数
            active_queries_result = await conn.fetchrow("""
                SELECT count(*) as active_queries
                FROM pg_stat_activity 
                WHERE state = 'active' AND query != '<IDLE>';
            """)
            
            # 平均クエリ時間
            avg_query_time_result = await conn.fetchrow("""
                SELECT COALESCE(AVG(mean_exec_time), 0) as avg_time
                FROM pg_stat_statements 
                WHERE calls > 10;
            """)
            
            # キャッシュヒット率
            cache_hit_result = await conn.fetchrow("""
                SELECT 
                    CASE 
                        WHEN blks_read + blks_hit = 0 THEN 1.0
                        ELSE blks_hit::float / (blks_read + blks_hit)
                    END as cache_hit_ratio
                FROM pg_stat_database 
                WHERE datname = current_database();
            """)
            
            # ベクター検索時間のサンプル測定
            vector_test_start = time.time()
            await conn.fetchrow("""
                SELECT id FROM videos 
                WHERE embedding IS NOT NULL 
                ORDER BY embedding <=> '[1,0,0]'::vector 
                LIMIT 1;
            """)
            vector_search_time = time.time() - vector_test_start
        
        metrics = DatabaseMetrics(
            connection_count=connection_result['active_connections'],
            active_queries=active_queries_result['active_queries'],
            avg_query_time=float(avg_query_time_result['avg_time'] or 0),
            cache_hit_ratio=float(cache_hit_result['cache_hit_ratio'] or 0),
            index_usage={},  # 簡略化
            vector_search_time=vector_search_time
        )
        
        self.metrics_history.append(metrics)
        
        # 履歴サイズ制限
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    async def optimize_connection_pooling(self) -> Dict[str, Any]:
        """接続プール最適化"""
        current_metrics = await self.monitor_database_performance()
        
        optimization_actions = []
        
        # 接続プールサイズ調整
        if self.connection_pool:
            pool_size = self.connection_pool.get_size()
            idle_connections = pool_size - current_metrics.active_queries
            
            if idle_connections < 2:
                # 接続数不足
                optimization_actions.append("increase_pool_size")
            elif idle_connections > pool_size * 0.5:
                # 接続数過多
                optimization_actions.append("consider_pool_reduction")
        
        # クエリタイムアウト調整
        if current_metrics.avg_query_time > self.config['optimization_thresholds']['slow_query_threshold']:
            optimization_actions.append("adjust_query_timeout")
        
        # キャッシュヒット率改善
        if current_metrics.cache_hit_ratio < self.config['optimization_thresholds']['cache_hit_ratio_min']:
            optimization_actions.append("improve_cache_hit_ratio")
        
        return {
            'current_metrics': current_metrics,
            'optimization_actions': optimization_actions,
            'pool_status': {
                'size': self.connection_pool.get_size() if self.connection_pool else 0,
                'idle': self.connection_pool.get_idle_size() if self.connection_pool else 0
            }
        }
    
    async def run_database_optimization(self) -> Dict[str, Any]:
        """包括的データベース最適化実行"""
        start_time = time.time()
        
        results = {
            'optimization_start_time': start_time,
            'connection_pool_initialized': False,
            'indexes_optimized': {},
            'performance_metrics': {},
            'pool_optimization': {},
            'total_optimization_time': 0.0
        }
        
        try:
            # 1. 接続プール初期化
            await self.initialize_connection_pool()
            results['connection_pool_initialized'] = True
            
            # 2. インデックス最適化
            results['indexes_optimized'] = await self.optimize_database_indexes()
            
            # 3. パフォーマンス監視
            results['performance_metrics'] = await self.monitor_database_performance()
            
            # 4. 接続プール最適化
            results['pool_optimization'] = await self.optimize_connection_pooling()
            
            results['total_optimization_time'] = time.time() - start_time
            results['success'] = True
            
            self.logger.info(f"Database optimization completed in {results['total_optimization_time']:.2f}s")
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Database optimization failed: {e}")
        
        return results
    
    async def close(self):
        """リソースクリーンアップ"""
        if self.connection_pool:
            await self.connection_pool.close()
            self.logger.info("Database connection pool closed")


def create_database_optimizer(config: Optional[Dict[str, Any]] = None) -> DatabaseOptimizer:
    """データベース最適化インスタンス作成"""
    return DatabaseOptimizer(config)


# 使用例とテスト
async def main():
    """使用例"""
    import json
    
    optimizer = create_database_optimizer()
    
    # 包括的最適化実行
    results = await optimizer.run_database_optimization()
    print("Database Optimization Results:")
    print(json.dumps(results, indent=2, default=str))
    
    # ベクター検索テスト
    test_embedding = [0.1] * 768  # 768次元テストベクター
    vector_results, vector_time = await optimizer.optimize_vector_queries(test_embedding, 5)
    print(f"\nVector Search Results ({vector_time:.3f}s):")
    for result in vector_results:
        print(f"  {result['title']}: {result['similarity']:.3f}")
    
    # 推薦クエリテスト
    rec_results, rec_time = await optimizer.optimize_recommendation_queries("test_user", 3)
    print(f"\nRecommendation Results ({rec_time:.3f}s):")
    for result in rec_results:
        print(f"  {result['title']}: {result['composite_score']:.3f}")
    
    await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())