#!/usr/bin/env python3
"""
Batch Embedding Update Script

This script updates user and video embeddings in batches for the Two-Tower model.
It can be run as a cron job for periodic model updates.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchEmbeddingUpdater:
    def __init__(self, 
                 db_url: str,
                 model_path: str,
                 batch_size: int = 1000):
        self.db_url = db_url
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.conn = None
        
        # Model components
        self.user_tower = None
        self.item_tower = None
        self.preprocessing = None
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                self.db_url,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_models(self):
        """Load trained Two-Tower models"""
        try:
            import tensorflow as tf
            import pickle
            
            # Load models
            user_tower_path = self.model_path / 'user_tower'
            item_tower_path = self.model_path / 'item_tower'
            
            if user_tower_path.exists():
                self.user_tower = tf.keras.models.load_model(user_tower_path)
                logger.info("Loaded user tower model")
            
            if item_tower_path.exists():
                self.item_tower = tf.keras.models.load_model(item_tower_path)
                logger.info("Loaded item tower model")
            
            # Load preprocessing artifacts
            preprocessing_path = self.model_path / 'preprocessing.pkl'
            if preprocessing_path.exists():
                with open(preprocessing_path, 'rb') as f:
                    self.preprocessing = pickle.load(f)
                logger.info("Loaded preprocessing artifacts")
            
            if not all([self.user_tower, self.item_tower, self.preprocessing]):
                raise ValueError("Failed to load required model components")
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def get_users_to_update(self, hours_threshold: int = 24) -> List[str]:
        """Get list of users whose embeddings need updating"""
        cursor = self.conn.cursor()
        
        query = """
        SELECT DISTINCT l.user_id
        FROM likes l
        LEFT JOIN user_embeddings ue ON l.user_id = ue.user_id
        WHERE l.created_at >= %s
        OR ue.updated_at IS NULL
        OR ue.updated_at < %s
        """
        
        threshold_time = datetime.now() - timedelta(hours=hours_threshold)
        cursor.execute(query, [threshold_time, threshold_time])
        
        users = [row['user_id'] for row in cursor.fetchall()]
        logger.info(f"Found {len(users)} users to update")
        
        return users
    
    def get_videos_to_update(self, hours_threshold: int = 168) -> List[str]:  # 1 week
        """Get list of videos whose embeddings need updating"""
        cursor = self.conn.cursor()
        
        query = """
        SELECT v.id
        FROM videos v
        LEFT JOIN video_embeddings ve ON v.id = ve.video_id
        WHERE v.created_at >= %s
        OR ve.updated_at IS NULL
        OR ve.updated_at < %s
        ORDER BY v.created_at DESC
        """
        
        threshold_time = datetime.now() - timedelta(hours=hours_threshold)
        cursor.execute(query, [threshold_time, threshold_time])
        
        videos = [row['id'] for row in cursor.fetchall()]
        logger.info(f"Found {len(videos)} videos to update")
        
        return videos
    
    def prepare_user_features(self, user_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Prepare user features for embedding generation"""
        cursor = self.conn.cursor()
        
        # Get user interaction data
        query = """
        SELECT 
            l.user_id,
            COUNT(*) as total_likes,
            AVG(v.price) as avg_price,
            STRING_AGG(DISTINCT v.genre, ', ') as liked_genres,
            MIN(l.created_at) as first_like,
            MAX(l.created_at) as last_like
        FROM likes l
        JOIN videos v ON l.video_id = v.id
        WHERE l.user_id = ANY(%s)
        AND l.created_at >= %s
        GROUP BY l.user_id
        """
        
        six_months_ago = datetime.now() - timedelta(days=180)
        cursor.execute(query, [user_ids, six_months_ago])
        
        user_data = {row['user_id']: row for row in cursor.fetchall()}
        
        # Prepare feature matrix
        features_list = []
        valid_user_ids = []
        
        for user_id in user_ids:
            if user_id not in user_data:
                # Cold start user - use defaults
                user_features = np.array([0.0, 0.0, 0.0])  # Basic features
                genre_prefs = np.zeros(len(self.preprocessing['genre_encoder'].classes_))
            else:
                data = user_data[user_id]
                
                # Calculate basic features
                total_likes = data['total_likes']
                avg_price = data['avg_price'] or 0
                
                first_like = data['first_like']
                last_like = data['last_like'] 
                days_active = max(1, (last_like - first_like).days)
                likes_per_day = total_likes / days_active
                
                user_features = np.array([
                    np.log(total_likes + 1) / 10,  # Log-normalized likes
                    avg_price / 10000,             # Price normalized
                    min(likes_per_day, 10) / 10    # Activity capped
                ])
                
                # Genre preferences
                genre_prefs = np.zeros(len(self.preprocessing['genre_encoder'].classes_))
                liked_genres = (data['liked_genres'] or '').split(', ')
                
                for genre in liked_genres:
                    if genre in self.preprocessing['genre_encoder'].classes_:
                        idx = self.preprocessing['genre_encoder'].transform([genre])[0]
                        genre_prefs[idx] = 1
                
                genre_prefs = genre_prefs / max(1, genre_prefs.sum())  # Normalize
            
            # Combine features
            combined_features = np.concatenate([user_features, genre_prefs])
            features_list.append(combined_features)
            valid_user_ids.append(user_id)
        
        return np.array(features_list), valid_user_ids
    
    def prepare_video_features(self, video_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Prepare video features for embedding generation"""
        cursor = self.conn.cursor()
        
        # Get video metadata
        query = """
        SELECT 
            v.id,
            v.title,
            v.description,
            v.maker,
            v.genre,
            v.price,
            v.duration_seconds,
            ARRAY_AGG(DISTINCT t.name) as tags
        FROM videos v
        LEFT JOIN video_tags vt ON v.id = vt.video_id
        LEFT JOIN tags t ON vt.tag_id = t.id
        WHERE v.id = ANY(%s)
        GROUP BY v.id, v.title, v.description, v.maker, v.genre, v.price, v.duration_seconds
        """
        
        cursor.execute(query, [video_ids])
        video_data = {row['id']: row for row in cursor.fetchall()}
        
        # Prepare feature matrix
        features_list = []
        valid_video_ids = []
        
        for video_id in video_ids:
            if video_id not in video_data:
                continue
                
            data = video_data[video_id]
            
            # Text features
            text = f"{data['title'] or ''} {data['description'] or ''}"
            try:
                text_features = self.preprocessing['text_vectorizer'].transform([text]).toarray()[0]
            except:
                text_features = np.zeros(self.preprocessing['text_vectorizer'].max_features)
            
            # Categorical features
            genre = data['genre'] or 'Unknown'
            maker = data['maker'] or 'Unknown'
            
            try:
                genre_encoded = self.preprocessing['genre_encoder'].transform([genre])[0]
            except:
                genre_encoded = 0
                
            try:
                maker_encoded = self.preprocessing['maker_encoder'].transform([maker])[0] 
            except:
                maker_encoded = 0
            
            # Numerical features
            price = (data['price'] or 0) / 10000  # Normalize
            duration = (data['duration_seconds'] or 0) / 7200  # 2 hours max
            
            numerical_features = np.array([price, duration])
            categorical_features = np.array([genre_encoded, maker_encoded])
            
            # Combine all features
            combined_features = np.concatenate([
                text_features,
                categorical_features,
                numerical_features
            ])
            
            features_list.append(combined_features)
            valid_video_ids.append(video_id)
        
        return np.array(features_list), valid_video_ids
    
    def update_user_embeddings(self, user_ids: List[str]) -> int:
        """Update user embeddings in database"""
        if not user_ids:
            return 0
            
        logger.info(f"Updating embeddings for {len(user_ids)} users")
        
        # Prepare features
        user_features, valid_user_ids = self.prepare_user_features(user_ids)
        
        if len(valid_user_ids) == 0:
            return 0
        
        # Generate embeddings
        embeddings = self.user_tower.predict(user_features, batch_size=self.batch_size)
        
        # Update database
        cursor = self.conn.cursor()
        updated_count = 0
        
        for i, user_id in enumerate(valid_user_ids):
            embedding = embeddings[i].tolist()
            
            try:
                cursor.execute("""
                    INSERT INTO user_embeddings (user_id, embedding, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding, 
                        updated_at = EXCLUDED.updated_at
                """, (user_id, embedding, datetime.now().isoformat()))
                
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update user {user_id}: {e}")
        
        self.conn.commit()
        logger.info(f"Successfully updated {updated_count} user embeddings")
        
        return updated_count
    
    def update_video_embeddings(self, video_ids: List[str]) -> int:
        """Update video embeddings in database"""
        if not video_ids:
            return 0
            
        logger.info(f"Updating embeddings for {len(video_ids)} videos")
        
        # Prepare features
        video_features, valid_video_ids = self.prepare_video_features(video_ids)
        
        if len(valid_video_ids) == 0:
            return 0
        
        # Generate embeddings
        embeddings = self.item_tower.predict(video_features, batch_size=self.batch_size)
        
        # Update database
        cursor = self.conn.cursor()
        updated_count = 0
        
        for i, video_id in enumerate(valid_video_ids):
            embedding = embeddings[i].tolist()
            
            try:
                cursor.execute("""
                    INSERT INTO video_embeddings (video_id, embedding, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (video_id)
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        updated_at = EXCLUDED.updated_at
                """, (video_id, embedding, datetime.now().isoformat()))
                
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update video {video_id}: {e}")
        
        self.conn.commit()
        logger.info(f"Successfully updated {updated_count} video embeddings")
        
        return updated_count
    
    def run_batch_update(self, update_users: bool = True, update_videos: bool = True) -> Dict[str, int]:
        """Run complete batch update process"""
        results = {'users_updated': 0, 'videos_updated': 0}
        
        try:
            self.connect_db()
            self.load_models()
            
            if update_users:
                # Update user embeddings
                users_to_update = self.get_users_to_update()
                
                # Process in batches
                for i in range(0, len(users_to_update), self.batch_size):
                    batch = users_to_update[i:i + self.batch_size]
                    updated = self.update_user_embeddings(batch)
                    results['users_updated'] += updated
            
            if update_videos:
                # Update video embeddings
                videos_to_update = self.get_videos_to_update()
                
                # Process in batches
                for i in range(0, len(videos_to_update), self.batch_size):
                    batch = videos_to_update[i:i + self.batch_size]
                    updated = self.update_video_embeddings(batch)
                    results['videos_updated'] += updated
            
            logger.info(f"Batch update completed: {results}")
            
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Batch Embedding Update')
    parser.add_argument('--db-url', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--model-path', required=True, help='Path to trained models')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size')
    parser.add_argument('--users-only', action='store_true', help='Update only user embeddings')
    parser.add_argument('--videos-only', action='store_true', help='Update only video embeddings')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = BatchEmbeddingUpdater(
        db_url=args.db_url,
        model_path=args.model_path,
        batch_size=args.batch_size
    )
    
    try:
        if args.dry_run:
            # Dry run mode
            updater.connect_db()
            users = updater.get_users_to_update()
            videos = updater.get_videos_to_update()
            
            print(f"Would update {len(users)} users and {len(videos)} videos")
            return
        
        # Run actual update
        update_users = not args.videos_only
        update_videos = not args.users_only
        
        results = updater.run_batch_update(
            update_users=update_users,
            update_videos=update_videos
        )
        
        print(f"Update completed: {results}")
        
    except Exception as e:
        logger.error(f"Batch update failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()