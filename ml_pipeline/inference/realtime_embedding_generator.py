#!/usr/bin/env python3
"""
Realtime Embedding Generator

High-performance real-time embedding generation API for Two-Tower model.
Handles cold start problems and provides sub-500ms response times for
new users and videos.
"""

import os
import time
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import json
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for realtime embedding generation"""
    db_url: str
    redis_url: str = "redis://localhost:6379"
    model_path: str = "./models"
    cache_ttl_seconds: int = 3600
    cold_start_cache_ttl: int = 86400  # 24 hours for cold start embeddings
    batch_inference_size: int = 32
    max_concurrent_requests: int = 100

# Pydantic models for API
class EmbeddingRequest(BaseModel):
    entity_id: str = Field(..., description="User ID or Video ID")
    entity_type: str = Field(..., description="Type: 'user' or 'video'")
    force_refresh: bool = Field(default=False, description="Force regeneration ignoring cache")

class EmbeddingResponse(BaseModel):
    entity_id: str
    entity_type: str
    embedding: List[float]
    is_cached: bool
    is_cold_start: bool
    generation_time_ms: float
    timestamp: str

class BatchEmbeddingRequest(BaseModel):
    requests: List[EmbeddingRequest] = Field(..., max_items=50)

class BatchEmbeddingResponse(BaseModel):
    results: List[EmbeddingResponse]
    total_time_ms: float
    cache_hit_rate: float

class RealtimeEmbeddingGenerator:
    """High-performance realtime embedding generation service"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._initialized = False
        
        # Core components
        self.db_pool = None
        self.redis_client = None
        self.user_tower = None
        self.item_tower = None
        self.user_feature_extractor = None
        self.item_feature_processor = None
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._model_cache = {}
        self._cold_start_templates = {}
        
        # Metrics
        self.metrics = {
            'requests_total': 0,
            'cache_hits': 0,
            'cold_start_served': 0,
            'generation_time_total': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
    
    async def initialize(self):
        """Initialize all components asynchronously"""
        if self._initialized:
            return
            
        try:
            # Initialize database connection
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=self.config.db_url
            )
            
            # Initialize Redis cache
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0
            )
            
            # Test connections
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._test_connections
            )
            
            # Load models
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._load_models
            )
            
            # Prepare cold start templates
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._prepare_cold_start_templates
            )
            
            self._initialized = True
            logger.info("RealtimeEmbeddingGenerator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RealtimeEmbeddingGenerator: {e}")
            raise
    
    def _test_connections(self):
        """Test database and Redis connections"""
        # Test database
        conn = self.db_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        finally:
            self.db_pool.putconn(conn)
        
        # Test Redis
        self.redis_client.ping()
        logger.info("Database and Redis connections verified")
    
    def _load_models(self):
        """Load trained models for inference"""
        import tensorflow as tf
        from pathlib import Path
        
        model_path = Path(self.config.model_path)
        
        # Load towers
        user_tower_path = model_path / 'user_tower.keras'
        item_tower_path = model_path / 'item_tower.keras'
        
        if user_tower_path.exists():
            self.user_tower = tf.keras.models.load_model(user_tower_path)
            logger.info("Loaded user tower model")
        else:
            raise FileNotFoundError(f"User tower not found: {user_tower_path}")
        
        if item_tower_path.exists():
            self.item_tower = tf.keras.models.load_model(item_tower_path)
            logger.info("Loaded item tower model")
        else:
            raise FileNotFoundError(f"Item tower not found: {item_tower_path}")
        
        # Initialize feature processors
        conn = self.db_pool.getconn()
        try:
            from real_user_feature_extractor import RealUserFeatureExtractor
            from enhanced_item_feature_processor import EnhancedItemFeatureProcessor
            
            self.user_feature_extractor = RealUserFeatureExtractor(conn)
            self.item_feature_processor = EnhancedItemFeatureProcessor(conn)
            
            logger.info("Feature processors initialized")
        finally:
            self.db_pool.putconn(conn)
    
    def _prepare_cold_start_templates(self):
        """Prepare default embeddings for cold start scenarios"""
        try:
            # Generate average user embedding from active users
            conn = self.db_pool.getconn()
            try:
                cursor = conn.cursor()
                
                # Get sample of active users
                cursor.execute("""
                    SELECT user_id FROM (
                        SELECT user_id, COUNT(*) as decisions
                        FROM user_video_decisions
                        WHERE created_at >= %s
                        GROUP BY user_id
                        ORDER BY decisions DESC
                        LIMIT 100
                    ) active_users
                """, [datetime.now() - timedelta(days=30)])
                
                active_users = [row[0] for row in cursor.fetchall()]
                
                if active_users:
                    # Generate embeddings for sample users
                    sample_embeddings = []
                    for user_id in active_users[:20]:  # Limit to 20 for performance
                        try:
                            feature_vector = self.user_feature_extractor.extract_user_features(user_id)
                            embedding = self.user_tower.predict(
                                np.array([feature_vector.features]), 
                                verbose=0
                            )[0]
                            sample_embeddings.append(embedding)
                        except:
                            continue
                    
                    if sample_embeddings:
                        # Calculate average embedding
                        avg_user_embedding = np.mean(sample_embeddings, axis=0)
                        self._cold_start_templates['user'] = avg_user_embedding.tolist()
                
                # Generate average video embedding from popular videos
                cursor.execute("""
                    SELECT video_id FROM (
                        SELECT video_id, COUNT(*) as interactions
                        FROM user_video_decisions
                        WHERE created_at >= %s
                        GROUP BY video_id
                        ORDER BY interactions DESC
                        LIMIT 100
                    ) popular_videos
                """, [datetime.now() - timedelta(days=30)])
                
                popular_videos = [row[0] for row in cursor.fetchall()]
                
                if popular_videos:
                    # Generate embeddings for sample videos
                    sample_embeddings = []
                    for video_id in popular_videos[:20]:  # Limit to 20 for performance
                        try:
                            feature_vector = self.item_feature_processor.process_video_features(video_id)
                            embedding = self.item_tower.predict(
                                np.array([feature_vector.features]),
                                verbose=0
                            )[0]
                            sample_embeddings.append(embedding)
                        except:
                            continue
                    
                    if sample_embeddings:
                        # Calculate average embedding
                        avg_video_embedding = np.mean(sample_embeddings, axis=0)
                        self._cold_start_templates['video'] = avg_video_embedding.tolist()
                
                logger.info(f"Cold start templates prepared: {list(self._cold_start_templates.keys())}")
                
            finally:
                self.db_pool.putconn(conn)
                
        except Exception as e:
            logger.error(f"Failed to prepare cold start templates: {e}")
            # Use zero vectors as fallback
            self._cold_start_templates = {
                'user': [0.0] * 768,
                'video': [0.0] * 768
            }
    
    async def get_embedding(self, entity_id: str, entity_type: str, force_refresh: bool = False) -> EmbeddingResponse:
        """Get embedding for single entity with caching"""
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        with self._lock:
            self.metrics['requests_total'] += 1
        
        # Cache key
        cache_key = f"embedding:{entity_type}:{entity_id}"
        
        # Check cache first (unless force refresh)
        cached_embedding = None
        if not force_refresh:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    cached_embedding = json.loads(cached_data)
                    with self._lock:
                        self.metrics['cache_hits'] += 1
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
        
        if cached_embedding:
            response_time = (time.time() - start_time) * 1000
            return EmbeddingResponse(
                entity_id=entity_id,
                entity_type=entity_type,
                embedding=cached_embedding['embedding'],
                is_cached=True,
                is_cold_start=cached_embedding.get('is_cold_start', False),
                generation_time_ms=response_time,
                timestamp=datetime.now().isoformat()
            )
        
        # Generate new embedding
        embedding_result = await asyncio.get_event_loop().run_in_executor(
            self.executor, self._generate_embedding, entity_id, entity_type
        )
        
        embedding, is_cold_start = embedding_result
        
        # Cache the result
        try:
            ttl = self.config.cold_start_cache_ttl if is_cold_start else self.config.cache_ttl_seconds
            cache_data = {
                'embedding': embedding,
                'is_cold_start': is_cold_start,
                'generated_at': datetime.now().isoformat()
            }
            self.redis_client.setex(
                cache_key, 
                ttl,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
        
        response_time = (time.time() - start_time) * 1000
        
        with self._lock:
            self.metrics['generation_time_total'] += response_time
            if is_cold_start:
                self.metrics['cold_start_served'] += 1
        
        return EmbeddingResponse(
            entity_id=entity_id,
            entity_type=entity_type,
            embedding=embedding,
            is_cached=False,
            is_cold_start=is_cold_start,
            generation_time_ms=response_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_embedding(self, entity_id: str, entity_type: str) -> Tuple[List[float], bool]:
        """Generate embedding for entity (runs in thread pool)"""
        try:
            conn = self.db_pool.getconn()
            try:
                if entity_type == 'user':
                    return self._generate_user_embedding(conn, entity_id)
                elif entity_type == 'video':
                    return self._generate_video_embedding(conn, entity_id)
                else:
                    raise ValueError(f"Unknown entity type: {entity_type}")
            finally:
                self.db_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Embedding generation failed for {entity_type} {entity_id}: {e}")
            # Return cold start template
            template = self._cold_start_templates.get(entity_type, [0.0] * 768)
            return template, True
    
    def _generate_user_embedding(self, conn, user_id: str) -> Tuple[List[float], bool]:
        """Generate user embedding"""
        try:
            # Check if user has any interactions
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM user_video_decisions WHERE user_id = %s",
                [user_id]
            )
            interaction_count = cursor.fetchone()[0]
            
            if interaction_count == 0:
                # Cold start user - return template
                template = self._cold_start_templates.get('user', [0.0] * 768)
                return template, True
            
            # Extract features and generate embedding
            feature_vector = self.user_feature_extractor.extract_user_features(user_id)
            embedding = self.user_tower.predict(
                np.array([feature_vector.features]),
                verbose=0
            )[0]
            
            return embedding.tolist(), False
            
        except Exception as e:
            logger.error(f"User embedding generation failed: {e}")
            template = self._cold_start_templates.get('user', [0.0] * 768)
            return template, True
    
    def _generate_video_embedding(self, conn, video_id: str) -> Tuple[List[float], bool]:
        """Generate video embedding"""
        try:
            # Check if video exists
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM videos WHERE id = %s", [video_id])
            video_exists = cursor.fetchone() is not None
            
            if not video_exists:
                # Cold start video - return template
                template = self._cold_start_templates.get('video', [0.0] * 768)
                return template, True
            
            # Extract features and generate embedding
            feature_vector = self.item_feature_processor.process_video_features(video_id)
            embedding = self.item_tower.predict(
                np.array([feature_vector.features]),
                verbose=0
            )[0]
            
            return embedding.tolist(), False
            
        except Exception as e:
            logger.error(f"Video embedding generation failed: {e}")
            template = self._cold_start_templates.get('video', [0.0] * 768)
            return template, True
    
    async def get_batch_embeddings(self, requests: List[EmbeddingRequest]) -> BatchEmbeddingResponse:
        """Get embeddings for multiple entities in parallel"""
        start_time = time.time()
        
        # Process all requests in parallel
        tasks = []
        for req in requests:
            task = self.get_embedding(req.entity_id, req.entity_type, req.force_refresh)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                # Create error response with cold start template
                req = requests[i]
                template = self._cold_start_templates.get(req.entity_type, [0.0] * 768)
                error_response = EmbeddingResponse(
                    entity_id=req.entity_id,
                    entity_type=req.entity_type,
                    embedding=template,
                    is_cached=False,
                    is_cold_start=True,
                    generation_time_ms=0.0,
                    timestamp=datetime.now().isoformat()
                )
                valid_results.append(error_response)
            else:
                valid_results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        cache_hits = sum(1 for r in valid_results if r.is_cached)
        cache_hit_rate = cache_hits / len(valid_results) if valid_results else 0.0
        
        return BatchEmbeddingResponse(
            results=valid_results,
            total_time_ms=total_time,
            cache_hit_rate=cache_hit_rate
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics['requests_total'] > 0:
            metrics['avg_generation_time_ms'] = metrics['generation_time_total'] / metrics['requests_total']
            metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['requests_total']
            metrics['cold_start_rate'] = metrics['cold_start_served'] / metrics['requests_total']
        else:
            metrics['avg_generation_time_ms'] = 0.0
            metrics['cache_hit_rate'] = 0.0
            metrics['cold_start_rate'] = 0.0
        
        metrics['timestamp'] = datetime.now().isoformat()
        return metrics
    
    async def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.db_pool:
            self.db_pool.closeall()
        if self.redis_client:
            self.redis_client.close()
        logger.info("RealtimeEmbeddingGenerator cleaned up")

# Global generator instance
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    global generator
    
    # Startup
    config = EmbeddingConfig(
        db_url=os.getenv("DATABASE_URL", "postgresql://localhost:5432/adult_matching"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        model_path=os.getenv("MODEL_PATH", "./models")
    )
    
    generator = RealtimeEmbeddingGenerator(config)
    await generator.initialize()
    
    yield
    
    # Shutdown
    if generator:
        await generator.cleanup()

# FastAPI app
app = FastAPI(
    title="Realtime Embedding Generator",
    description="High-performance real-time embedding generation for Two-Tower model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/embedding", response_model=EmbeddingResponse)
async def get_single_embedding(request: EmbeddingRequest):
    """Get embedding for single entity"""
    if not generator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return await generator.get_embedding(
            request.entity_id,
            request.entity_type,
            request.force_refresh
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/batch", response_model=BatchEmbeddingResponse)
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    """Get embeddings for multiple entities"""
    if not generator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return await generator.get_batch_embeddings(request.requests)
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    if not generator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return generator.get_metrics()

@app.post("/cache/clear")
async def clear_cache(background_tasks: BackgroundTasks):
    """Clear embedding cache"""
    if not generator or not generator.redis_client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    def clear_redis_cache():
        try:
            # Clear all embedding cache keys
            keys = generator.redis_client.keys("embedding:*")
            if keys:
                generator.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
    
    background_tasks.add_task(clear_redis_cache)
    return {"message": "Cache clear initiated"}

def main():
    """Run the service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Realtime Embedding Generator API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--db-url', help='PostgreSQL connection URL')
    parser.add_argument('--redis-url', help='Redis connection URL')
    parser.add_argument('--model-path', help='Path to trained models')
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.db_url:
        os.environ['DATABASE_URL'] = args.db_url
    if args.redis_url:
        os.environ['REDIS_URL'] = args.redis_url
    if args.model_path:
        os.environ['MODEL_PATH'] = args.model_path
    
    # Run the server
    uvicorn.run(
        "realtime_embedding_generator:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == '__main__':
    main()