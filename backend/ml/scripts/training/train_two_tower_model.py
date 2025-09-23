#!/usr/bin/env python3
"""
Two-Tower Recommendation Model Training
Pattern 1: Rating-based conversion (4.0+ = Like, 3.0- = Skip)
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoTowerModelTrainer:
    """Two-Tower recommendation model trainer for adult video matching."""
    
    def __init__(self, 
                 db_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
                 pseudo_users_file: str = "data/pseudo_users_from_reviews.json"):
        self.db_url = db_url
        self.pseudo_users_file = pseudo_users_file
        
        # Model parameters
        self.user_embedding_dim = 64
        self.item_embedding_dim = 64
        self.hidden_units = [128, 64, 32]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 256
        self.epochs = 50
        
        # Data containers
        self.user_data = None
        self.item_data = None
        self.interactions = None
        self.user_encoders = {}
        self.item_encoders = {}
        self.scalers = {}
        
        # Model components
        self.user_tower = None
        self.item_tower = None
        self.model = None
        
    def load_pseudo_users(self) -> Dict:
        """Load pseudo-user data from JSON file."""
        logger.info(f"Loading pseudo-users from {self.pseudo_users_file}")
        
        with open(self.pseudo_users_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data['users'])} pseudo-users with {data['stats']['totalLikes'] + data['stats']['totalSkips']} total actions")
        return data
    
    def load_video_data(self) -> pd.DataFrame:
        """Load video data from PostgreSQL database."""
        logger.info("Loading video data from database...")
        
        conn = psycopg2.connect(self.db_url)
        
        query = """
        SELECT 
            v.id,
            v.external_id,
            v.title,
            v.price::float as price,
            v.duration_seconds::float as duration_seconds,
            v.product_released_at,
            v.genre,
            v.maker,
            v.director,
            v.series,
            COALESCE(tag_count.tag_count, 0) as tag_count,
            COALESCE(performer_count.performer_count, 0) as performer_count
        FROM videos v
        LEFT JOIN (
            SELECT video_id, COUNT(*) as tag_count
            FROM video_tags
            GROUP BY video_id
        ) tag_count ON v.id = tag_count.video_id
        LEFT JOIN (
            SELECT video_id, COUNT(*) as performer_count
            FROM video_performers
            GROUP BY video_id
        ) performer_count ON v.id = performer_count.video_id
        WHERE v.source = 'dmm'
        """
        
        video_df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(video_df)} videos from database")
        return video_df
    
    def create_interaction_data(self, pseudo_users_data: Dict, video_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction dataset from pseudo-users and video data."""
        logger.info("Creating interaction dataset...")
        
        interactions = []
        
        for user_idx, user in enumerate(pseudo_users_data['users']):
            user_name = user['name']
            
            # Process likes (rating 4.0+)
            for content_id in user['likes']:
                video_match = video_df[video_df['external_id'] == content_id]
                if not video_match.empty:
                    interactions.append({
                        'user_id': user_idx,
                        'user_name': user_name,
                        'video_id': video_match.iloc[0]['id'],
                        'content_id': content_id,
                        'label': 1,  # Like
                        'interaction_type': 'like'
                    })
            
            # Process skips (rating 3.0-)
            for content_id in user['skips']:
                video_match = video_df[video_df['external_id'] == content_id]
                if not video_match.empty:
                    interactions.append({
                        'user_id': user_idx,
                        'user_name': user_name,
                        'video_id': video_match.iloc[0]['id'],
                        'content_id': content_id,
                        'label': 0,  # Skip
                        'interaction_type': 'skip'
                    })
        
        interaction_df = pd.DataFrame(interactions)
        logger.info(f"Created {len(interaction_df)} interactions ({(interaction_df['label']==1).sum()} likes, {(interaction_df['label']==0).sum()} skips)")
        
        return interaction_df
    
    def prepare_user_features(self, pseudo_users_data: Dict) -> pd.DataFrame:
        """Prepare user feature matrix."""
        logger.info("Preparing user features...")
        
        user_features = []
        
        for user_idx, user in enumerate(pseudo_users_data['users']):
            total_actions = len(user['likes']) + len(user['skips'])
            like_ratio = len(user['likes']) / total_actions if total_actions > 0 else 0
            
            user_features.append({
                'user_id': user_idx,
                'total_interactions': total_actions,
                'like_ratio': like_ratio,
                'avg_rating': user.get('avgRating', 4.0),
                'like_count': len(user['likes']),
                'skip_count': len(user['skips']),
                'review_count': user.get('totalReviews', total_actions)
            })
        
        user_df = pd.DataFrame(user_features)
        logger.info(f"Prepared features for {len(user_df)} users")
        
        return user_df
    
    def prepare_item_features(self, video_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare item (video) feature matrix."""
        logger.info("Preparing item features...")
        
        # Handle missing values and normalize numerical features
        item_df = video_df.copy()
        
        # Fill missing numerical values
        item_df['price'] = item_df['price'].fillna(item_df['price'].median())
        item_df['duration_seconds'] = item_df['duration_seconds'].fillna(item_df['duration_seconds'].median())
        
        # Calculate release recency (days from release to now)
        item_df['product_released_at'] = pd.to_datetime(item_df['product_released_at'], utc=True)
        current_date = pd.Timestamp.now(tz='UTC')
        item_df['release_recency'] = (current_date - item_df['product_released_at']).dt.days
        item_df['release_recency'] = item_df['release_recency'].fillna(item_df['release_recency'].median())
        
        # Fill missing categorical values
        for col in ['genre', 'maker', 'director', 'series']:
            item_df[col] = item_df[col].fillna('unknown')
        
        logger.info(f"Prepared features for {len(item_df)} items")
        return item_df
    
    def encode_features(self, user_df: pd.DataFrame, item_df: pd.DataFrame, interaction_df: pd.DataFrame):
        """Encode categorical features and normalize numerical features."""
        logger.info("Encoding features...")
        
        # Encode user categorical features
        user_encoded = user_df.copy()
        
        # Encode item categorical features
        item_encoded = item_df.copy()
        
        categorical_item_features = ['genre', 'maker', 'director', 'series']
        for feature in categorical_item_features:
            encoder = LabelEncoder()
            item_encoded[f'{feature}_encoded'] = encoder.fit_transform(item_encoded[feature].astype(str))
            self.item_encoders[feature] = encoder
        
        # Normalize numerical features
        numerical_user_features = ['total_interactions', 'like_ratio', 'avg_rating', 'like_count', 'skip_count', 'review_count']
        numerical_item_features = ['price', 'duration_seconds', 'release_recency', 'tag_count', 'performer_count']
        
        for feature in numerical_user_features:
            scaler = StandardScaler()
            user_encoded[f'{feature}_scaled'] = scaler.fit_transform(user_encoded[[feature]])
            self.scalers[f'user_{feature}'] = scaler
        
        for feature in numerical_item_features:
            scaler = StandardScaler()
            item_encoded[f'{feature}_scaled'] = scaler.fit_transform(item_encoded[[feature]])
            self.scalers[f'item_{feature}'] = scaler
        
        self.user_data = user_encoded
        self.item_data = item_encoded
        self.interactions = interaction_df
        
        logger.info("Feature encoding completed")
    
    def build_user_tower(self, user_input_dim: int) -> keras.Model:
        """Build user tower architecture."""
        logger.info("Building user tower...")
        
        user_input = keras.Input(shape=(user_input_dim,), name='user_features')
        
        x = user_input
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        user_embedding = layers.Dense(self.user_embedding_dim, activation='relu', name='user_embedding')(x)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(user_embedding)
        
        user_tower = keras.Model(inputs=user_input, outputs=user_embedding, name='user_tower')
        return user_tower
    
    def build_item_tower(self, item_input_dim: int) -> keras.Model:
        """Build item tower architecture."""
        logger.info("Building item tower...")
        
        item_input = keras.Input(shape=(item_input_dim,), name='item_features')
        
        x = item_input
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        item_embedding = layers.Dense(self.item_embedding_dim, activation='relu', name='item_embedding')(x)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item_embedding)
        
        item_tower = keras.Model(inputs=item_input, outputs=item_embedding, name='item_tower')
        return item_tower
    
    def build_two_tower_model(self) -> keras.Model:
        """Build complete Two-Tower model."""
        logger.info("Building Two-Tower model...")
        
        # Get feature dimensions
        user_feature_cols = [col for col in self.user_data.columns if col.endswith('_scaled')]
        item_feature_cols = [col for col in self.item_data.columns if col.endswith('_scaled') or col.endswith('_encoded')]
        
        user_input_dim = len(user_feature_cols)
        item_input_dim = len(item_feature_cols)
        
        logger.info(f"User input dimension: {user_input_dim}")
        logger.info(f"Item input dimension: {item_input_dim}")
        
        # Build towers
        self.user_tower = self.build_user_tower(user_input_dim)
        self.item_tower = self.build_item_tower(item_input_dim)
        
        # Combined model inputs
        user_input = keras.Input(shape=(user_input_dim,), name='user_features')
        item_input = keras.Input(shape=(item_input_dim,), name='item_features')
        
        # Get embeddings
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        
        # Compute similarity (dot product)
        similarity = layers.Dot(axes=1)([user_embedding, item_embedding])
        
        # Output layer for binary classification (like/skip)
        output = layers.Dense(1, activation='sigmoid', name='prediction')(similarity)
        
        # Combined model
        model = keras.Model(inputs=[user_input, item_input], outputs=output, name='two_tower_model')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Two-Tower model built successfully")
        return model
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data arrays."""
        logger.info("Preparing training data...")
        
        # Get feature columns
        user_feature_cols = [col for col in self.user_data.columns if col.endswith('_scaled')]
        item_feature_cols = [col for col in self.item_data.columns if col.endswith('_scaled') or col.endswith('_encoded')]
        
        # Prepare arrays
        user_features = []
        item_features = []
        labels = []
        
        for _, interaction in self.interactions.iterrows():
            user_id = interaction['user_id']
            video_id = interaction['video_id']
            label = interaction['label']
            
            # Get user features
            user_row = self.user_data[self.user_data['user_id'] == user_id]
            if not user_row.empty:
                user_feat = user_row[user_feature_cols].values[0]
                
                # Get item features
                item_row = self.item_data[self.item_data['id'] == video_id]
                if not item_row.empty:
                    item_feat = item_row[item_feature_cols].values[0]
                    
                    user_features.append(user_feat)
                    item_features.append(item_feat)
                    labels.append(label)
        
        user_features = np.array(user_features)
        item_features = np.array(item_features)
        labels = np.array(labels)
        
        logger.info(f"Training data prepared: {len(labels)} samples")
        logger.info(f"User features shape: {user_features.shape}")
        logger.info(f"Item features shape: {item_features.shape}")
        logger.info(f"Labels distribution - Likes: {np.sum(labels)}, Skips: {len(labels) - np.sum(labels)}")
        
        return user_features, item_features, labels
    
    def train_model(self):
        """Train the Two-Tower model."""
        logger.info("Starting model training...")
        
        # Prepare data
        user_features, item_features, labels = self.prepare_training_data()
        
        # Split data
        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
        
        X_user_train, X_user_test = user_features[train_indices], user_features[test_indices]
        X_item_train, X_item_test = item_features[train_indices], item_features[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        
        logger.info(f"Training set: {len(y_train)} samples")
        logger.info(f"Test set: {len(y_test)} samples")
        
        # Build model
        self.model = self.build_two_tower_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            [X_user_train, X_item_train], y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([X_user_test, X_item_test], y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc, test_prec, test_recall = self.model.evaluate([X_user_test, X_item_test], y_test, verbose=0)
        
        logger.info(f"Test Results:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_acc:.4f}")
        logger.info(f"  Precision: {test_prec:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        
        # Detailed classification report
        y_pred = (self.model.predict([X_user_test, X_item_test]) > 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Skip', 'Like']))
        
        return history
    
    def save_model(self, model_dir: str = "models/two_tower_pattern1"):
        """Save the trained model and encoders."""
        logger.info(f"Saving model to {model_dir}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the full model
        self.model.save(os.path.join(model_dir, "two_tower_model.keras"))
        
        # Save individual towers
        self.user_tower.save(os.path.join(model_dir, "user_tower.keras"))
        self.item_tower.save(os.path.join(model_dir, "item_tower.keras"))
        
        # Save encoders and scalers
        import pickle
        with open(os.path.join(model_dir, "encoders_scalers.pkl"), 'wb') as f:
            pickle.dump({
                'item_encoders': self.item_encoders,
                'scalers': self.scalers
            }, f)
        
        # Save metadata
        metadata = {
            'model_type': 'two_tower',
            'pattern': 'pattern1_rating_based',
            'like_threshold': 4.0,
            'skip_threshold': 3.0,
            'user_embedding_dim': self.user_embedding_dim,
            'item_embedding_dim': self.item_embedding_dim,
            'hidden_units': self.hidden_units,
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(self.interactions)
        }
        
        with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model saved successfully")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("🚀 Starting Two-Tower Model Training Pipeline - Pattern 1")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load data
            pseudo_users_data = self.load_pseudo_users()
            video_df = self.load_video_data()
            
            # Step 2: Create interaction data
            interaction_df = self.create_interaction_data(pseudo_users_data, video_df)
            
            # Step 3: Prepare features
            user_df = self.prepare_user_features(pseudo_users_data)
            item_df = self.prepare_item_features(video_df)
            
            # Step 4: Encode features
            self.encode_features(user_df, item_df, interaction_df)
            
            # Step 5: Train model
            history = self.train_model()
            
            # Step 6: Save model
            self.save_model()
            
            logger.info("✅ Two-Tower Model Training Completed Successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise

def main():
    """Main training function."""
    trainer = TwoTowerModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()