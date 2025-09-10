#!/usr/bin/env python3
"""
Production Embedding Updater

Large-scale batch embedding update system for Two-Tower model.
Optimized for 200k+ videos and 10k+ users with 768-dimensional vectors.
Provides efficient PostgreSQL integration with transaction safety.
"""

import os
import sys
import json
import time
import logging
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Generator
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import ThreadedConnectionPool
import threading
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class UpdateConfig:
    """Configuration for production embedding updates"""
    db_url: str
    model_path: str
    batch_size: int = 2000
    max_workers: int = 4
    max_memory_gb: float = 6.0
    timeout_minutes: int = 8
    user_hours_threshold: int = 24
    video_hours_threshold: int = 168  # 1 week
    connection_pool_size: int = 10
    upsert_batch_size: int = 1000

@dataclass 
class UpdateMetrics:
    """Metrics tracking for embedding updates"""
    start_time: datetime
    users_processed: int = 0
    videos_processed: int = 0
    users_updated: int = 0
    videos_updated: int = 0
    errors: int = 0
    peak_memory_gb: float = 0.0
    processing_time_seconds: float = 0.0

class MemoryMonitor:
    """Memory usage monitoring and protection"""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.process = psutil.Process()
        
    def check_memory(self) -> float:
        """Check current memory usage in GB"""
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        return memory_gb
    
    def is_memory_safe(self) -> bool:
        """Check if memory usage is within limits"""
        return self.check_memory() < self.max_memory_gb

class ProductionEmbeddingUpdater:
    """Production-grade embedding update system with advanced optimizations"""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        self.metrics = UpdateMetrics(start_time=datetime.now())
        
        # Database connection pool
        self.pool = None
        
        # Model components
        self.user_tower = None
        self.item_tower = None
        self.user_feature_extractor = None
        self.item_feature_processor = None
        
        # Threading
        self._shutdown_event = threading.Event()
        
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=self.config.connection_pool_size,
                dsn=self.config.db_url
            )
            logger.info(f"Initialized connection pool with {self.config.connection_pool_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def load_models(self):
        """Load trained models and feature processors"""
        try:
            import tensorflow as tf
            from enhanced_two_tower_trainer import EnhancedTwoTowerTrainer
            from real_user_feature_extractor import RealUserFeatureExtractor
            from enhanced_item_feature_processor import EnhancedItemFeatureProcessor
            
            model_path = Path(self.config.model_path)
            
            # Load trained models
            user_tower_path = model_path / 'user_tower.keras'
            item_tower_path = model_path / 'item_tower.keras'
            
            if user_tower_path.exists():
                self.user_tower = tf.keras.models.load_model(user_tower_path)
                logger.info("Loaded user tower model")
            else:
                raise FileNotFoundError(f"User tower model not found: {user_tower_path}")
            
            if item_tower_path.exists():
                self.item_tower = tf.keras.models.load_model(item_tower_path)
                logger.info("Loaded item tower model")
            else:
                raise FileNotFoundError(f"Item tower model not found: {item_tower_path}")
            
            # Initialize feature processors
            with self.get_connection() as conn:
                self.user_feature_extractor = RealUserFeatureExtractor(conn)
                self.item_feature_processor = EnhancedItemFeatureProcessor(conn)
            
            logger.info("Initialized feature processors")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def get_users_needing_updates(self) -> List[str]:
        """Get users whose embeddings need updating based on recent activity"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Use user_video_decisions table instead of likes
            query = """
            SELECT DISTINCT uvd.user_id
            FROM user_video_decisions uvd
            LEFT JOIN user_embeddings ue ON uvd.user_id = ue.user_id
            WHERE uvd.created_at >= %s
            OR ue.updated_at IS NULL
            OR ue.updated_at < %s
            ORDER BY uvd.user_id
            """
            
            threshold_time = datetime.now() - timedelta(hours=self.config.user_hours_threshold)
            cursor.execute(query, [threshold_time, threshold_time])
            
            users = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(users)} users needing embedding updates")
            
            return users
    
    def get_videos_needing_updates(self) -> List[str]:
        """Get videos whose embeddings need updating"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT v.id
            FROM videos v
            LEFT JOIN video_embeddings ve ON v.id = ve.video_id
            WHERE v.created_at >= %s
            OR ve.updated_at IS NULL
            OR ve.updated_at < %s
            ORDER BY v.created_at DESC
            LIMIT 200000
            """
            
            threshold_time = datetime.now() - timedelta(hours=self.config.video_hours_threshold)
            cursor.execute(query, [threshold_time, threshold_time])
            
            videos = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(videos)} videos needing embedding updates")
            
            return videos
    
    def batch_generator(self, items: List[str], batch_size: int) -> Generator[List[str], None, None]:
        """Generate batches from item list"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    def process_user_batch(self, user_ids: List[str]) -> int:
        """Process a batch of users to generate embeddings"""
        if not user_ids or self._shutdown_event.is_set():
            return 0
        
        # Memory check
        if not self.memory_monitor.is_memory_safe():
            logger.warning(f"Memory usage {self.memory_monitor.check_memory():.2f}GB exceeds limit")
            return 0
        
        try:
            with self.get_connection() as conn:
                # Extract user features using updated extractor
                user_feature_vectors = []
                valid_user_ids = []
                
                for user_id in user_ids:
                    try:
                        feature_vector = self.user_feature_extractor.extract_user_features(user_id)
                        user_feature_vectors.append(feature_vector.features)
                        valid_user_ids.append(user_id)
                    except Exception as e:
                        logger.warning(f"Failed to extract features for user {user_id}: {e}")
                        self.metrics.errors += 1
                        continue
                
                if not valid_user_ids:
                    return 0
                
                # Generate embeddings using user tower
                features_array = np.array(user_feature_vectors)
                embeddings = self.user_tower.predict(features_array, batch_size=len(features_array))
                
                # Batch upsert to database
                updated_count = self._upsert_user_embeddings(conn, valid_user_ids, embeddings)
                
                self.metrics.users_processed += len(user_ids)
                self.metrics.users_updated += updated_count
                
                logger.info(f"Processed user batch: {len(user_ids)} users, {updated_count} updated")
                return updated_count
                
        except Exception as e:
            logger.error(f"Failed to process user batch: {e}")
            self.metrics.errors += 1
            return 0
    
    def process_video_batch(self, video_ids: List[str]) -> int:
        """Process a batch of videos to generate embeddings"""
        if not video_ids or self._shutdown_event.is_set():
            return 0
        
        # Memory check
        if not self.memory_monitor.is_memory_safe():
            logger.warning(f"Memory usage {self.memory_monitor.check_memory():.2f}GB exceeds limit")
            return 0
        
        try:
            with self.get_connection() as conn:
                # Extract video features using updated processor
                video_features_list = []
                valid_video_ids = []
                
                for video_id in video_ids:
                    try:
                        feature_vector = self.item_feature_processor.process_video_features(video_id)
                        video_features_list.append(feature_vector.features)
                        valid_video_ids.append(video_id)
                    except Exception as e:
                        logger.warning(f"Failed to process features for video {video_id}: {e}")
                        self.metrics.errors += 1
                        continue
                
                if not valid_video_ids:
                    return 0
                
                # Generate embeddings using item tower
                features_array = np.array(video_features_list)
                embeddings = self.item_tower.predict(features_array, batch_size=len(features_array))
                
                # Batch upsert to database
                updated_count = self._upsert_video_embeddings(conn, valid_video_ids, embeddings)
                
                self.metrics.videos_processed += len(video_ids)
                self.metrics.videos_updated += updated_count
                
                logger.info(f"Processed video batch: {len(video_ids)} videos, {updated_count} updated")
                return updated_count
                
        except Exception as e:
            logger.error(f"Failed to process video batch: {e}")
            self.metrics.errors += 1
            return 0
    
    def _upsert_user_embeddings(self, conn, user_ids: List[str], embeddings: np.ndarray) -> int:
        """Efficient batch upsert for user embeddings"""
        cursor = conn.cursor()
        
        # Prepare data for batch upsert
        update_data = []
        current_time = datetime.now()
        
        for i, user_id in enumerate(user_ids):
            embedding = embeddings[i].tolist()
            update_data.append((user_id, embedding, current_time))
        
        # Batch upsert using execute_batch for performance
        upsert_query = """
        INSERT INTO user_embeddings (user_id, embedding, updated_at)
        VALUES %s
        ON CONFLICT (user_id)
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            updated_at = EXCLUDED.updated_at
        """
        
        try:
            execute_batch(
                cursor, 
                upsert_query, 
                update_data,
                template=None,
                page_size=self.config.upsert_batch_size
            )
            conn.commit()
            return len(user_ids)
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert user embeddings: {e}")
            raise
    
    def _upsert_video_embeddings(self, conn, video_ids: List[str], embeddings: np.ndarray) -> int:
        """Efficient batch upsert for video embeddings"""
        cursor = conn.cursor()
        
        # Prepare data for batch upsert
        update_data = []
        current_time = datetime.now()
        
        for i, video_id in enumerate(video_ids):
            embedding = embeddings[i].tolist()
            update_data.append((video_id, embedding, current_time))
        
        # Batch upsert using execute_batch for performance
        upsert_query = """
        INSERT INTO video_embeddings (video_id, embedding, updated_at)
        VALUES %s
        ON CONFLICT (video_id)
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            updated_at = EXCLUDED.updated_at
        """
        
        try:
            execute_batch(
                cursor,
                upsert_query,
                update_data,
                template=None,
                page_size=self.config.upsert_batch_size
            )
            conn.commit()
            return len(video_ids)
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert video embeddings: {e}")
            raise
    
    def run_parallel_updates(self, update_users: bool = True, update_videos: bool = True) -> UpdateMetrics:
        """Run embedding updates in parallel with advanced monitoring"""
        start_time = time.time()
        
        try:
            self.initialize_connection_pool()
            self.load_models()
            
            futures = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                
                if update_users:
                    # Get users needing updates
                    users_to_update = self.get_users_needing_updates()
                    logger.info(f"Processing {len(users_to_update)} users in parallel")
                    
                    # Submit user batches
                    for batch in self.batch_generator(users_to_update, self.config.batch_size):
                        if self._shutdown_event.is_set():
                            break
                        future = executor.submit(self.process_user_batch, batch)
                        futures.append(('users', future))
                
                if update_videos:
                    # Get videos needing updates  
                    videos_to_update = self.get_videos_needing_updates()
                    logger.info(f"Processing {len(videos_to_update)} videos in parallel")
                    
                    # Submit video batches
                    for batch in self.batch_generator(videos_to_update, self.config.batch_size):
                        if self._shutdown_event.is_set():
                            break
                        future = executor.submit(self.process_video_batch, batch)
                        futures.append(('videos', future))
                
                # Process completed futures with timeout protection
                timeout_seconds = self.config.timeout_minutes * 60
                
                for batch_type, future in as_completed(futures, timeout=timeout_seconds):
                    try:
                        result = future.result(timeout=30)  # Individual batch timeout
                        
                        # Update peak memory tracking
                        current_memory = self.memory_monitor.check_memory()
                        self.metrics.peak_memory_gb = max(self.metrics.peak_memory_gb, current_memory)
                        
                        if not self.memory_monitor.is_memory_safe():
                            logger.warning("Memory limit exceeded, initiating graceful shutdown")
                            self._shutdown_event.set()
                            break
                            
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Batch processing timeout for {batch_type}")
                        self.metrics.errors += 1
                        future.cancel()
                    except Exception as e:
                        logger.error(f"Batch processing failed for {batch_type}: {e}")
                        self.metrics.errors += 1
            
            # Calculate final metrics
            self.metrics.processing_time_seconds = time.time() - start_time
            
            logger.info(f"Parallel update completed in {self.metrics.processing_time_seconds:.1f}s")
            logger.info(f"Users: {self.metrics.users_updated}/{self.metrics.users_processed}")
            logger.info(f"Videos: {self.metrics.videos_updated}/{self.metrics.videos_processed}")
            logger.info(f"Peak memory: {self.metrics.peak_memory_gb:.2f}GB")
            logger.info(f"Errors: {self.metrics.errors}")
            
        except Exception as e:
            logger.error(f"Parallel update failed: {e}")
            raise
        finally:
            if self.pool:
                self.pool.closeall()
        
        return self.metrics
    
    def run_status_check(self) -> Dict[str, Any]:
        """Check system status and pending updates"""
        try:
            self.initialize_connection_pool()
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check embedding table status
                status = {}
                
                # User embedding statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_users,
                        COUNT(ue.user_id) as users_with_embeddings,
                        MAX(ue.updated_at) as last_user_update
                    FROM (SELECT DISTINCT user_id FROM user_video_decisions) u
                    LEFT JOIN user_embeddings ue ON u.user_id = ue.user_id
                """)
                user_stats = cursor.fetchone()
                
                # Video embedding statistics  
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_videos,
                        COUNT(ve.video_id) as videos_with_embeddings,
                        MAX(ve.updated_at) as last_video_update
                    FROM videos v
                    LEFT JOIN video_embeddings ve ON v.id = ve.video_id
                """)
                video_stats = cursor.fetchone()
                
                # Pending updates
                users_needing_updates = len(self.get_users_needing_updates())
                videos_needing_updates = len(self.get_videos_needing_updates())
                
                status = {
                    'users': {
                        'total': user_stats[0],
                        'with_embeddings': user_stats[1],
                        'last_update': user_stats[2].isoformat() if user_stats[2] else None,
                        'pending_updates': users_needing_updates
                    },
                    'videos': {
                        'total': video_stats[0], 
                        'with_embeddings': video_stats[1],
                        'last_update': video_stats[2].isoformat() if video_stats[2] else None,
                        'pending_updates': videos_needing_updates
                    },
                    'system': {
                        'memory_gb': self.memory_monitor.check_memory(),
                        'memory_limit_gb': self.config.max_memory_gb,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                return status
                
        finally:
            if self.pool:
                self.pool.closeall()

def main():
    parser = argparse.ArgumentParser(description='Production Embedding Updater')
    parser.add_argument('--db-url', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--model-path', required=True, help='Path to trained models')
    parser.add_argument('--batch-size', type=int, default=2000, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=4, help='Max parallel workers')
    parser.add_argument('--max-memory-gb', type=float, default=6.0, help='Memory limit in GB')
    parser.add_argument('--timeout-minutes', type=int, default=8, help='Processing timeout in minutes')
    parser.add_argument('--users-only', action='store_true', help='Update only user embeddings')
    parser.add_argument('--videos-only', action='store_true', help='Update only video embeddings')
    parser.add_argument('--status', action='store_true', help='Show embedding status')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated')
    
    args = parser.parse_args()
    
    # Build configuration
    config = UpdateConfig(
        db_url=args.db_url,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_memory_gb=args.max_memory_gb,
        timeout_minutes=args.timeout_minutes
    )
    
    # Initialize updater
    updater = ProductionEmbeddingUpdater(config)
    
    try:
        if args.status:
            # Status check mode
            status = updater.run_status_check()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return
        
        if args.dry_run:
            # Dry run mode
            updater.initialize_connection_pool()
            users = updater.get_users_needing_updates()
            videos = updater.get_videos_needing_updates()
            
            print(f"Would update {len(users)} users and {len(videos)} videos")
            print(f"Estimated processing time: {(len(users) + len(videos)) / config.batch_size * 2:.1f} minutes")
            return
        
        # Run actual update
        update_users = not args.videos_only
        update_videos = not args.users_only
        
        metrics = updater.run_parallel_updates(
            update_users=update_users,
            update_videos=update_videos
        )
        
        # Output final results
        result = asdict(metrics)
        result['start_time'] = result['start_time'].isoformat()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"Production update failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()