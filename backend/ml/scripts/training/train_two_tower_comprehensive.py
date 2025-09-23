#!/usr/bin/env python3
"""
Two-Tower Recommendation Model Training - Comprehensive Version
Pattern 1: Rating-based conversion (4.0+ = Like, 3.0- = Skip)
Using comprehensive pseudo-users data (7,435 user actions)
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

class ComprehensiveTwoTowerTrainer:
    """Two-Tower recommendation model trainer using comprehensive pseudo-users data."""
    
    def __init__(self, 
                 db_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres",
                 pseudo_users_file: str = "data/comprehensive_pseudo_users.json"):
        self.db_url = db_url
        self.pseudo_users_file = pseudo_users_file
        
        # Model parameters - simplified for better stability
        self.user_embedding_dim = 32
        self.item_embedding_dim = 32
        self.hidden_units = [64, 32]
        self.dropout_rate = 0.3
        self.learning_rate = 0.01
        self.batch_size = 128
        self.epochs = 100
        
        # Data containers
        self.user_data = None
        self.item_data = None
        self.interactions = None
        self.item_encoders = {}
        self.scalers = {}
        
        # Model components
        self.model = None
        
    def load_comprehensive_pseudo_users(self) -> Dict:
        """Load comprehensive pseudo-user data from JSON file."""
        logger.info(f"Loading comprehensive pseudo-users from {self.pseudo_users_file}")
        
        with open(self.pseudo_users_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_actions = data['stats']['totalLikes'] + data['stats']['totalSkips']
        logger.info(f"Loaded {data['stats']['validUsers']} pseudo-users with {total_actions:,} total actions")
        logger.info(f"Like ratio: {data['stats']['likeRate']:.1f}%")
        
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
            COALESCE(v.price::float, 0) as price,
            COALESCE(v.duration_seconds::float, 0) as duration_seconds,
            v.product_released_at,
            COALESCE(v.genre, 'unknown') as genre,
            COALESCE(v.maker, 'unknown') as maker,
            COALESCE(v.director, 'unknown') as director,
            COALESCE(v.series, 'unknown') as series,
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
        
        logger.info(f"Loaded {len(video_df):,} videos from database")
        return video_df
    
    def create_interaction_data(self, pseudo_users_data: Dict, video_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction dataset from comprehensive pseudo-users and video data."""
        logger.info("Creating comprehensive interaction dataset...")
        
        interactions = []
        video_id_map = dict(zip(video_df['external_id'], video_df['id']))
        
        for user_idx, user in enumerate(pseudo_users_data['users']):
            user_name = user['name']
            
            # Process likes (rating 4.0+)
            for content_id in user['likes']:
                if content_id in video_id_map:
                    interactions.append({
                        'user_id': user_idx,
                        'user_name': user_name,
                        'video_id': video_id_map[content_id],
                        'content_id': content_id,
                        'label': 1,  # Like
                        'interaction_type': 'like'
                    })
            
            # Process skips (rating 3.0-)
            for content_id in user['skips']:
                if content_id in video_id_map:
                    interactions.append({
                        'user_id': user_idx,
                        'user_name': user_name,
                        'video_id': video_id_map[content_id],
                        'content_id': content_id,
                        'label': 0,  # Skip
                        'interaction_type': 'skip'
                    })
        
        interaction_df = pd.DataFrame(interactions)
        
        likes_count = (interaction_df['label'] == 1).sum()
        skips_count = (interaction_df['label'] == 0).sum()
        
        logger.info(f"Created {len(interaction_df):,} interactions ({likes_count:,} likes, {skips_count:,} skips)")
        logger.info(f"Like ratio: {likes_count/len(interaction_df)*100:.1f}%")
        
        return interaction_df
    
    def prepare_user_features(self, pseudo_users_data: Dict) -> pd.DataFrame:
        """Prepare user feature matrix from comprehensive data."""
        logger.info("Preparing comprehensive user features...")
        
        user_features = []
        
        for user_idx, user in enumerate(pseudo_users_data['users']):
            total_actions = len(user['likes']) + len(user['skips'])
            like_ratio = len(user['likes']) / total_actions if total_actions > 0 else 0
            
            # Use actual rating data from the comprehensive dataset
            ratings = user.get('ratings', [])
            avg_rating = np.mean(ratings) if ratings else 4.0
            rating_variance = np.var(ratings) if ratings else 0.0
            
            user_features.append({
                'user_id': user_idx,
                'total_interactions': total_actions,
                'like_ratio': like_ratio,
                'avg_rating': avg_rating,
                'rating_variance': rating_variance,
                'like_count': len(user['likes']),
                'skip_count': len(user['skips']),
                'total_reviews': user.get('totalReviews', total_actions)
            })
        
        user_df = pd.DataFrame(user_features)
        logger.info(f"Prepared features for {len(user_df)} users")
        
        return user_df
    
    def prepare_item_features(self, video_df: pd.DataFrame, interaction_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare item (video) feature matrix with interaction-based filtering."""
        logger.info("Preparing item features...")
        
        # Only keep videos that appear in interactions
        interacted_video_ids = set(interaction_df['video_id'])
        item_df = video_df[video_df['id'].isin(interacted_video_ids)].copy()
        
        logger.info(f"Filtered to {len(item_df):,} videos with interactions")
        
        # Handle missing values
        item_df['price'] = item_df['price'].fillna(item_df['price'].median())
        item_df['duration_seconds'] = item_df['duration_seconds'].fillna(item_df['duration_seconds'].median())
        
        # Calculate release recency safely
        try:
            item_df['product_released_at'] = pd.to_datetime(item_df['product_released_at'], utc=True, errors='coerce')
            current_date = pd.Timestamp.now(tz='UTC')
            item_df['release_recency'] = (current_date - item_df['product_released_at']).dt.days
            item_df['release_recency'] = item_df['release_recency'].fillna(item_df['release_recency'].median())
        except Exception as e:
            logger.warning(f"Date handling issue: {e}, using default recency")
            item_df['release_recency'] = 365  # Default to 1 year
        
        logger.info(f"Prepared features for {len(item_df):,} items")
        return item_df
    
    def encode_features(self, user_df: pd.DataFrame, item_df: pd.DataFrame, interaction_df: pd.DataFrame):
        """Encode categorical features and normalize numerical features."""
        logger.info("Encoding features...")
        
        # Process user features - only numerical normalization needed
        user_encoded = user_df.copy()
        
        # Process item features
        item_encoded = item_df.copy()
        
        # Encode categorical item features
        categorical_item_features = ['genre', 'maker', 'director', 'series']
        for feature in categorical_item_features:
            encoder = LabelEncoder()
            # Handle unknown values gracefully
            item_encoded[f'{feature}_encoded'] = encoder.fit_transform(item_encoded[feature].astype(str))
            self.item_encoders[feature] = encoder
        
        # Normalize numerical features with better handling
        numerical_user_features = ['total_interactions', 'like_ratio', 'avg_rating', 'rating_variance', 
                                  'like_count', 'skip_count', 'total_reviews']
        numerical_item_features = ['price', 'duration_seconds', 'release_recency', 'tag_count', 'performer_count']
        
        # User feature normalization
        for feature in numerical_user_features:
            if user_encoded[feature].std() > 0:  # Only scale if there's variation
                scaler = StandardScaler()
                user_encoded[f'{feature}_scaled'] = scaler.fit_transform(user_encoded[[feature]])
                self.scalers[f'user_{feature}'] = scaler
            else:
                user_encoded[f'{feature}_scaled'] = user_encoded[feature]
        
        # Item feature normalization
        for feature in numerical_item_features:
            if item_encoded[feature].std() > 0:  # Only scale if there's variation
                scaler = StandardScaler()
                item_encoded[f'{feature}_scaled'] = scaler.fit_transform(item_encoded[[feature]])
                self.scalers[f'item_{feature}'] = scaler
            else:
                user_encoded[f'{feature}_scaled'] = item_encoded[feature]
        
        self.user_data = user_encoded
        self.item_data = item_encoded
        self.interactions = interaction_df
        
        logger.info("Feature encoding completed successfully")
    
    def build_simplified_model(self) -> keras.Model:
        """Build a simplified but more stable Two-Tower model."""
        logger.info("Building simplified Two-Tower model...")
        
        # Get feature dimensions
        user_feature_cols = [col for col in self.user_data.columns if col.endswith('_scaled')]
        item_feature_cols = [col for col in self.item_data.columns 
                           if col.endswith('_scaled') or col.endswith('_encoded')]
        
        user_input_dim = len(user_feature_cols)
        item_input_dim = len(item_feature_cols)
        
        logger.info(f"User input dimension: {user_input_dim}")
        logger.info(f"Item input dimension: {item_input_dim}")
        
        # User tower
        user_input = keras.Input(shape=(user_input_dim,), name='user_features')
        user_x = user_input
        for units in self.hidden_units:
            user_x = layers.Dense(units, activation='relu', 
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros')(user_x)
            user_x = layers.BatchNormalization()(user_x)
            user_x = layers.Dropout(self.dropout_rate)(user_x)
        
        user_embedding = layers.Dense(self.user_embedding_dim, activation='relu', 
                                    name='user_embedding',
                                    kernel_initializer='glorot_uniform')(user_x)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(user_embedding)
        
        # Item tower
        item_input = keras.Input(shape=(item_input_dim,), name='item_features')
        item_x = item_input
        for units in self.hidden_units:
            item_x = layers.Dense(units, activation='relu',
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros')(item_x)
            item_x = layers.BatchNormalization()(item_x)
            item_x = layers.Dropout(self.dropout_rate)(item_x)
        
        item_embedding = layers.Dense(self.item_embedding_dim, activation='relu', 
                                    name='item_embedding',
                                    kernel_initializer='glorot_uniform')(item_x)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item_embedding)
        
        # Compute similarity (dot product)
        similarity = layers.Dot(axes=1)([user_embedding, item_embedding])
        
        # Add bias term and final activation
        output = layers.Dense(1, activation='sigmoid', 
                            name='prediction',
                            kernel_initializer='glorot_uniform')(layers.Reshape((1,))(similarity))
        
        # Build model
        model = keras.Model(inputs=[user_input, item_input], outputs=output, name='two_tower_model')
        
        # Use more stable optimizer and loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Simplified Two-Tower model built successfully")
        return model
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data arrays."""
        logger.info("Preparing comprehensive training data...")
        
        # Get feature columns
        user_feature_cols = [col for col in self.user_data.columns if col.endswith('_scaled')]
        item_feature_cols = [col for col in self.item_data.columns 
                           if col.endswith('_scaled') or col.endswith('_encoded')]
        
        # Create video ID mapping for faster lookup
        item_map = {row['id']: idx for idx, row in self.item_data.iterrows()}
        
        # Prepare arrays more efficiently
        user_features = []
        item_features = []
        labels = []
        
        # Get user features once (since we likely have only 1 user)
        user_data_dict = {}
        for _, user_row in self.user_data.iterrows():
            user_id = user_row['user_id']
            user_data_dict[user_id] = user_row[user_feature_cols].values
        
        for _, interaction in self.interactions.iterrows():
            user_id = interaction['user_id']
            video_id = interaction['video_id']
            label = interaction['label']
            
            if user_id in user_data_dict:
                # Get item features
                item_row = self.item_data[self.item_data['id'] == video_id]
                if not item_row.empty:
                    user_feat = user_data_dict[user_id]
                    item_feat = item_row[item_feature_cols].values[0]
                    
                    user_features.append(user_feat)
                    item_features.append(item_feat)
                    labels.append(label)
        
        user_features = np.array(user_features, dtype=np.float32)
        item_features = np.array(item_features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        # Check for any NaN values
        if np.isnan(user_features).any():
            logger.warning("Found NaN values in user features, replacing with 0")
            user_features = np.nan_to_num(user_features, 0)
        
        if np.isnan(item_features).any():
            logger.warning("Found NaN values in item features, replacing with 0") 
            item_features = np.nan_to_num(item_features, 0)
        
        logger.info(f"Training data prepared: {len(labels):,} samples")
        logger.info(f"User features shape: {user_features.shape}")
        logger.info(f"Item features shape: {item_features.shape}")
        logger.info(f"Labels distribution - Likes: {np.sum(labels):,}, Skips: {len(labels) - np.sum(labels):,}")
        
        return user_features, item_features, labels
    
    def train_model(self):
        """Train the comprehensive Two-Tower model."""
        logger.info("Starting comprehensive model training...")
        
        # Prepare data
        user_features, item_features, labels = self.prepare_training_data()
        
        # Split data with stratification
        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )
        val_indices, test_indices = train_test_split(
            test_indices, test_size=0.5, random_state=42, 
            stratify=labels[test_indices]
        )
        
        X_user_train = user_features[train_indices]
        X_user_val = user_features[val_indices]  
        X_user_test = user_features[test_indices]
        
        X_item_train = item_features[train_indices]
        X_item_val = item_features[val_indices]
        X_item_test = item_features[test_indices]
        
        y_train = labels[train_indices]
        y_val = labels[val_indices]
        y_test = labels[test_indices]
        
        logger.info(f"Training set: {len(y_train):,} samples")
        logger.info(f"Validation set: {len(y_val):,} samples")
        logger.info(f"Test set: {len(y_test):,} samples")
        
        # Build model
        self.model = self.build_simplified_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1
        )
        
        # Train model with class weighting for imbalanced data
        class_weight = {
            0: len(labels) / (2 * np.sum(labels == 0)),  # Skip class weight
            1: len(labels) / (2 * np.sum(labels == 1))   # Like class weight
        }
        
        logger.info(f"Using class weights: {class_weight}")
        
        history = self.model.fit(
            [X_user_train, X_item_train], y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([X_user_val, X_item_val], y_val),
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        test_results = self.model.evaluate([X_user_test, X_item_test], y_test, verbose=0)
        
        logger.info(f"Test Results:")
        logger.info(f"  Loss: {test_results[0]:.4f}")
        logger.info(f"  Accuracy: {test_results[1]:.4f}")
        logger.info(f"  Precision: {test_results[2]:.4f}")
        logger.info(f"  Recall: {test_results[3]:.4f}")
        
        # Detailed classification report
        y_pred = (self.model.predict([X_user_test, X_item_test]) > 0.5).astype(int).flatten()
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Skip', 'Like']))
        
        return history
    
    def save_model(self, model_dir: str = "models/comprehensive_two_tower_pattern1"):
        """Save the trained model and encoders."""
        logger.info(f"Saving comprehensive model to {model_dir}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the full model
        self.model.save(os.path.join(model_dir, "comprehensive_two_tower_model.keras"))
        
        # Save encoders and scalers
        import pickle
        with open(os.path.join(model_dir, "encoders_scalers.pkl"), 'wb') as f:
            pickle.dump({
                'item_encoders': self.item_encoders,
                'scalers': self.scalers
            }, f)
        
        # Save metadata
        metadata = {
            'model_type': 'comprehensive_two_tower',
            'pattern': 'pattern1_rating_based_comprehensive',
            'like_threshold': 4.0,
            'skip_threshold': 3.0,
            'user_embedding_dim': self.user_embedding_dim,
            'item_embedding_dim': self.item_embedding_dim,
            'hidden_units': self.hidden_units,
            'training_samples': len(self.interactions),
            'trained_at': datetime.now().isoformat(),
            'source_data': 'comprehensive_pseudo_users.json'
        }
        
        with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Comprehensive model saved successfully")
    
    def run_training_pipeline(self):
        """Run the complete comprehensive training pipeline."""
        logger.info("🚀 Starting Comprehensive Two-Tower Model Training - Pattern 1")
        logger.info("=" * 70)
        
        try:
            # Step 1: Load comprehensive data
            pseudo_users_data = self.load_comprehensive_pseudo_users()
            video_df = self.load_video_data()
            
            # Step 2: Create comprehensive interaction data
            interaction_df = self.create_interaction_data(pseudo_users_data, video_df)
            
            # Step 3: Prepare features
            user_df = self.prepare_user_features(pseudo_users_data)
            item_df = self.prepare_item_features(video_df, interaction_df)
            
            # Step 4: Encode features
            self.encode_features(user_df, item_df, interaction_df)
            
            # Step 5: Train model
            history = self.train_model()
            
            # Step 6: Save model
            self.save_model()
            
            logger.info("✅ Comprehensive Two-Tower Model Training Completed Successfully!")
            logger.info(f"📊 Final Training Stats:")
            logger.info(f"   - Total interactions processed: {len(self.interactions):,}")
            logger.info(f"   - Likes: {(self.interactions['label']==1).sum():,}")
            logger.info(f"   - Skips: {(self.interactions['label']==0).sum():,}")
            logger.info(f"   - Videos in training: {len(self.item_data):,}")
            logger.info(f"   - Users in training: {len(self.user_data):,}")
            
        except Exception as e:
            logger.error(f"❌ Comprehensive training failed: {str(e)}")
            raise

def main():
    """Main comprehensive training function."""
    trainer = ComprehensiveTwoTowerTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()