#!/usr/bin/env python3
"""
Two-Tower Recommendation Model Training Script

This script trains a Two-Tower model for the adult video recommendation system.
It can run locally and export models for Supabase Edge Functions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwoTowerTrainer:
    def __init__(self, 
                 db_url: str,
                 embedding_dim: int = 64,  # Reduced from 768 for efficiency
                 learning_rate: float = 0.001,
                 batch_size: int = 512,    # Increased for better training
                 epochs: int = 20):
        self.db_url = db_url
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Feature processors
        self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.genre_encoder = LabelEncoder()
        self.maker_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Models
        self.user_tower = None
        self.item_tower = None
        self.full_model = None

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

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training data from database"""
        logger.info("Loading training data...")
        
        # Load videos with metadata
        video_query = """
        SELECT 
            v.id,
            v.title,
            v.description,
            v.maker,
            v.genre,
            v.price,
            v.duration_seconds,
            ARRAY_AGG(DISTINCT p.name) as performers,
            ARRAY_AGG(DISTINCT t.name) as tags
        FROM videos v
        LEFT JOIN video_performers vp ON v.id = vp.video_id
        LEFT JOIN performers p ON vp.performer_id = p.id
        LEFT JOIN video_tags vt ON v.id = vt.video_id
        LEFT JOIN tags t ON vt.tag_id = t.id
        GROUP BY v.id, v.title, v.description, v.maker, v.genre, v.price, v.duration_seconds
        """
        
        videos_df = pd.read_sql(video_query, self.conn)
        
        # Load user interactions (likes)
        interactions_query = """
        SELECT 
            user_id,
            video_id,
            created_at,
            1 as rating
        FROM likes
        WHERE created_at >= %s
        """
        
        # Get interactions from last 6 months for training
        six_months_ago = (datetime.now() - timedelta(days=180)).isoformat()
        interactions_df = pd.read_sql(interactions_query, self.conn, params=[six_months_ago])
        
        # Aggregate user features
        user_features_query = """
        SELECT 
            l.user_id,
            COUNT(*) as total_likes,
            AVG(v.price) as avg_price,
            STRING_AGG(DISTINCT v.genre, ', ') as liked_genres,
            STRING_AGG(DISTINCT v.maker, ', ') as liked_makers,
            MIN(l.created_at) as first_like,
            MAX(l.created_at) as last_like
        FROM likes l
        JOIN videos v ON l.video_id = v.id
        WHERE l.created_at >= %s
        GROUP BY l.user_id
        HAVING COUNT(*) >= 3
        """
        
        users_df = pd.read_sql(user_features_query, self.conn, params=[six_months_ago])
        
        logger.info(f"Loaded {len(videos_df)} videos, {len(users_df)} users, {len(interactions_df)} interactions")
        return videos_df, users_df, interactions_df

    def preprocess_features(self, videos_df: pd.DataFrame, users_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features for training"""
        logger.info("Preprocessing features...")
        
        # Video features
        video_features = []
        
        # Text features (title + description)
        videos_df['text'] = videos_df['title'].fillna('') + ' ' + videos_df['description'].fillna('')
        text_features = self.text_vectorizer.fit_transform(videos_df['text']).toarray()
        
        # Categorical features
        videos_df['genre'] = videos_df['genre'].fillna('Unknown')
        videos_df['maker'] = videos_df['maker'].fillna('Unknown')
        genre_encoded = self.genre_encoder.fit_transform(videos_df['genre']).reshape(-1, 1)
        maker_encoded = self.maker_encoder.fit_transform(videos_df['maker']).reshape(-1, 1)
        
        # Numerical features
        numerical_features = videos_df[['price', 'duration_seconds']].fillna(0).values
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine video features
        video_features = np.hstack([
            text_features,
            genre_encoded,
            maker_encoded,
            numerical_features
        ])
        
        # User features
        user_features = []
        
        # User activity features
        users_df['days_active'] = (pd.to_datetime(users_df['last_like']) - pd.to_datetime(users_df['first_like'])).dt.days
        users_df['days_active'] = users_df['days_active'].fillna(1)
        users_df['likes_per_day'] = users_df['total_likes'] / (users_df['days_active'] + 1)
        
        user_numerical = users_df[['total_likes', 'avg_price', 'likes_per_day']].fillna(0).values
        user_numerical = StandardScaler().fit_transform(user_numerical)
        
        # Genre preferences (simplified)
        genre_preferences = []
        for genres in users_df['liked_genres'].fillna(''):
            pref_vector = np.zeros(len(self.genre_encoder.classes_))
            for genre in genres.split(', '):
                if genre in self.genre_encoder.classes_:
                    idx = self.genre_encoder.transform([genre])[0]
                    pref_vector[idx] = 1
            genre_preferences.append(pref_vector)
        
        genre_preferences = np.array(genre_preferences)
        
        # Combine user features
        user_features = np.hstack([
            user_numerical,
            genre_preferences
        ])
        
        logger.info(f"Video features shape: {video_features.shape}")
        logger.info(f"User features shape: {user_features.shape}")
        
        return video_features, user_features

    def build_towers(self, video_feature_dim: int, user_feature_dim: int):
        """Build user and item towers"""
        logger.info("Building Two-Tower model...")
        
        # User Tower
        user_input = tf.keras.Input(shape=(user_feature_dim,), name='user_input')
        user_dense1 = tf.keras.layers.Dense(512, activation='relu')(user_input)
        user_dropout1 = tf.keras.layers.Dropout(0.3)(user_dense1)
        user_dense2 = tf.keras.layers.Dense(256, activation='relu')(user_dropout1)
        user_dropout2 = tf.keras.layers.Dropout(0.3)(user_dense2)
        user_embedding = tf.keras.layers.Dense(self.embedding_dim, activation='tanh', name='user_embedding')(user_dropout2)
        user_normalized = tf.keras.utils.normalize(user_embedding, axis=1)
        
        self.user_tower = tf.keras.Model(inputs=user_input, outputs=user_normalized, name='user_tower')
        
        # Item Tower
        item_input = tf.keras.Input(shape=(video_feature_dim,), name='item_input')
        item_dense1 = tf.keras.layers.Dense(512, activation='relu')(item_input)
        item_dropout1 = tf.keras.layers.Dropout(0.3)(item_dense1)
        item_dense2 = tf.keras.layers.Dense(256, activation='relu')(item_dropout1)
        item_dropout2 = tf.keras.layers.Dropout(0.3)(item_dense2)
        item_embedding = tf.keras.layers.Dense(self.embedding_dim, activation='tanh', name='item_embedding')(item_dropout2)
        item_normalized = tf.keras.utils.normalize(item_embedding, axis=1)
        
        self.item_tower = tf.keras.Model(inputs=item_input, outputs=item_normalized, name='item_tower')
        
        # Full model (for training)
        user_emb = self.user_tower(user_input)
        item_emb = self.item_tower(item_input)
        
        # Dot product for similarity (Two-Tower standard)
        similarity = tf.keras.layers.Dot(axes=1, normalize=False)([user_emb, item_emb])
        # Apply sigmoid to convert to probability
        output = tf.keras.activations.sigmoid(similarity)
        
        self.full_model = tf.keras.Model(inputs=[user_input, item_input], 
                                        outputs=tf.expand_dims(output, -1), 
                                        name='two_tower_model')
        
        # Compile model with custom metrics
        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info("Model architecture:")
        self.full_model.summary()

    def create_training_pairs(self, videos_df: pd.DataFrame, users_df: pd.DataFrame, 
                            interactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create positive and negative training pairs"""
        logger.info("Creating training pairs...")
        
        # Create user and video mappings
        user_to_idx = {user_id: idx for idx, user_id in enumerate(users_df['user_id'])}
        video_to_idx = {video_id: idx for idx, video_id in enumerate(videos_df['id'])}
        
        positive_pairs = []
        for _, interaction in interactions_df.iterrows():
            user_id = interaction['user_id']
            video_id = interaction['video_id']
            
            if user_id in user_to_idx and video_id in video_to_idx:
                positive_pairs.append((user_to_idx[user_id], video_to_idx[video_id], 1))
        
        logger.info(f"Created {len(positive_pairs)} positive pairs")
        
        # Generate negative samples
        negative_pairs = []
        for user_id, user_idx in user_to_idx.items():
            # Get videos liked by this user
            liked_videos = set(interactions_df[interactions_df['user_id'] == user_id]['video_id'])
            
            # Sample random videos not liked by this user
            available_videos = set(videos_df['id']) - liked_videos
            if len(available_videos) > 0:
                # Sample same number of negatives as positives for this user
                user_positives = len(liked_videos)
                n_negatives = min(user_positives * 2, len(available_videos))  # 2x negatives
                
                sampled_videos = np.random.choice(list(available_videos), size=n_negatives, replace=False)
                for video_id in sampled_videos:
                    if video_id in video_to_idx:
                        negative_pairs.append((user_idx, video_to_idx[video_id], 0))
        
        logger.info(f"Created {len(negative_pairs)} negative pairs")
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        np.random.shuffle(all_pairs)
        
        user_indices = np.array([pair[0] for pair in all_pairs])
        video_indices = np.array([pair[1] for pair in all_pairs])
        labels = np.array([pair[2] for pair in all_pairs])
        
        return user_indices, video_indices, labels

    def train_model(self, video_features: np.ndarray, user_features: np.ndarray,
                   user_indices: np.ndarray, video_indices: np.ndarray, labels: np.ndarray):
        """Train the Two-Tower model"""
        logger.info("Training Two-Tower model...")
        
        # Prepare training data
        X_user = user_features[user_indices]
        X_video = video_features[video_indices]
        y = labels.astype(np.float32)
        
        # Train/validation split
        train_user, val_user, train_video, val_video, train_y, val_y = train_test_split(
            X_user, X_video, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # Train
        history = self.full_model.fit(
            [train_user, train_video], train_y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([val_user, val_video], val_y),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc, val_prec, val_recall = self.full_model.evaluate(
            [val_user, val_video], val_y, verbose=0
        )
        
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, "
                   f"Precision: {val_prec:.4f}, Recall: {val_recall:.4f}")
        
        return history

    def export_models(self, output_dir: str = 'models'):
        """Export models for production use"""
        logger.info("Exporting models...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save TensorFlow models
        self.user_tower.save(f'{output_dir}/user_tower')
        self.item_tower.save(f'{output_dir}/item_tower')
        self.full_model.save(f'{output_dir}/two_tower_model')
        
        # Convert to TensorFlow.js format
        try:
            import tensorflowjs as tfjs
            tfjs.converters.save_keras_model(self.user_tower, f'{output_dir}/user_tower_js')
            tfjs.converters.save_keras_model(self.item_tower, f'{output_dir}/item_tower_js')
            logger.info("Exported TensorFlow.js models")
        except ImportError:
            logger.warning("TensorFlow.js not available, skipping JS export")
        
        # Convert to ONNX format
        try:
            import tf2onnx
            import onnx
            
            # Convert user tower to ONNX
            user_onnx_model, _ = tf2onnx.convert.from_keras(
                self.user_tower,
                input_signature=None,
                opset=13
            )
            onnx.save(user_onnx_model, f'{output_dir}/user_tower.onnx')
            
            # Convert item tower to ONNX
            item_onnx_model, _ = tf2onnx.convert.from_keras(
                self.item_tower,
                input_signature=None,
                opset=13
            )
            onnx.save(item_onnx_model, f'{output_dir}/item_tower.onnx')
            
            logger.info("Exported ONNX models")
        except ImportError:
            logger.warning("tf2onnx not available, skipping ONNX export")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
        
        # Save preprocessing artifacts
        preprocessing = {
            'text_vectorizer': self.text_vectorizer,
            'genre_encoder': self.genre_encoder,
            'maker_encoder': self.maker_encoder,
            'scaler': self.scaler,
            'embedding_dim': self.embedding_dim,
        }
        
        import pickle
        with open(f'{output_dir}/preprocessing.pkl', 'wb') as f:
            pickle.dump(preprocessing, f)
        
        # Save model metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'model_version': datetime.now().isoformat(),
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
            }
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models exported to {output_dir}/")

    def generate_embeddings(self, video_features: np.ndarray, user_features: np.ndarray,
                          videos_df: pd.DataFrame, users_df: pd.DataFrame):
        """Generate and save embeddings to database"""
        logger.info("Generating embeddings...")
        
        # Generate video embeddings
        video_embeddings = self.item_tower.predict(video_features, batch_size=self.batch_size)
        
        # Update video_embeddings table
        cursor = self.conn.cursor()
        for i, video_id in enumerate(videos_df['id']):
            embedding = video_embeddings[i].tolist()
            cursor.execute("""
                INSERT INTO video_embeddings (video_id, embedding, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (video_id)
                DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at
            """, (video_id, embedding, datetime.now().isoformat()))
        
        # Generate user embeddings
        user_embeddings = self.user_tower.predict(user_features, batch_size=self.batch_size)
        
        # Update user_embeddings table
        for i, user_id in enumerate(users_df['user_id']):
            embedding = user_embeddings[i].tolist()
            cursor.execute("""
                INSERT INTO user_embeddings (user_id, embedding, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at
            """, (user_id, embedding, datetime.now().isoformat()))
        
        self.conn.commit()
        logger.info(f"Updated embeddings for {len(videos_df)} videos and {len(users_df)} users")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Two-Tower Recommendation Model')
    parser.add_argument('--db-url', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--embedding-dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    parser.add_argument('--update-db', action='store_true', help='Update database with embeddings')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TwoTowerTrainer(
        db_url=args.db_url,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    try:
        # Connect to database
        trainer.connect_db()
        
        # Load and preprocess data
        videos_df, users_df, interactions_df = trainer.load_data()
        video_features, user_features = trainer.preprocess_features(videos_df, users_df)
        
        # Build model
        trainer.build_towers(video_features.shape[1], user_features.shape[1])
        
        # Create training pairs
        user_indices, video_indices, labels = trainer.create_training_pairs(
            videos_df, users_df, interactions_df
        )
        
        # Train model
        trainer.train_model(video_features, user_features, user_indices, video_indices, labels)
        
        # Export models
        trainer.export_models(args.output_dir)
        
        # Update database embeddings
        if args.update_db:
            trainer.generate_embeddings(video_features, user_features, videos_df, users_df)
        
        logger.info("Training completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Review model metrics and validation results")  
        logger.info("2. Test inference with Edge Functions integration")
        logger.info("3. Deploy to production with A/B testing")
        logger.info(f"4. Models saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if hasattr(trainer, 'conn'):
            trainer.conn.close()

if __name__ == '__main__':
    main()