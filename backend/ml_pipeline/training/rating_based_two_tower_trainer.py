"""
評価ベースTwo-Tower推薦モデル訓練システム（パターン1）

レビュー評価値から変換された疑似ユーザーデータでTwo-Towerモデルを訓練
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging
from datetime import datetime
from pathlib import Path
import pickle
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RatingBasedTwoTowerTrainer:
    """評価ベース疑似ユーザーデータでTwo-Towerモデルを訓練"""
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 50,
                 validation_split: float = 0.2):
        
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        
        # 特徴処理器
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000, 
            ngram_range=(1, 2),
            min_df=2,
            stop_words=None  # 日本語対応のため無効化
        )
        self.genre_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # モデル
        self.user_tower = None
        self.item_tower = None
        self.full_model = None
        
        # 訓練統計
        self.training_stats = {
            'start_time': datetime.now().isoformat(),
            'data_stats': {},
            'training_history': {},
            'evaluation_metrics': {},
            'model_version': 'rating_based_v1.0'
        }
    
    def load_pseudo_user_data(self, 
                             pseudo_users_file: str = "../../data_processing/processed_data/rating_based_pseudo_users.json",
                             integrated_reviews_file: str = "../../data_processing/processed_data/integrated_reviews.json") -> Tuple[List[Dict], List[Dict]]:
        """疑似ユーザーデータと統合レビューデータを読み込み"""
        
        logger.info("疑似ユーザーデータ読み込み開始...")
        
        # 疑似ユーザーデータ読み込み
        try:
            with open(pseudo_users_file, 'r', encoding='utf-8') as f:
                pseudo_users = json.load(f)
            logger.info(f"疑似ユーザー読み込み完了: {len(pseudo_users)}人")
        except Exception as e:
            logger.error(f"疑似ユーザーデータ読み込みエラー: {e}")
            return [], []
        
        # 統合レビューデータ読み込み
        try:
            with open(integrated_reviews_file, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
            logger.info(f"統合レビューデータ読み込み完了: {len(reviews)}件")
        except Exception as e:
            logger.error(f"統合レビューデータ読み込みエラー: {e}")
            return pseudo_users, []
        
        return pseudo_users, reviews
    
    def prepare_training_data(self, pseudo_users: List[Dict], reviews: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """訓練データを準備"""
        
        logger.info("訓練データ準備開始...")
        
        # Step 1: コンテンツ特徴量作成
        content_features = self._create_content_features(reviews)
        logger.info(f"コンテンツ特徴量: {len(content_features)} アイテム")
        
        # Step 2: ユーザー特徴量とインタラクションデータ作成
        user_features, interactions = self._create_user_interactions(pseudo_users, content_features)
        logger.info(f"ユーザー特徴量: {len(user_features)} ユーザー")
        logger.info(f"インタラクションデータ: {len(interactions)} 行動")
        
        # データフレーム変換
        content_df = pd.DataFrame(content_features)
        interactions_df = pd.DataFrame(interactions)
        
        # 統計情報保存
        self.training_stats['data_stats'] = {
            'total_users': len(user_features),
            'total_items': len(content_features),
            'total_interactions': len(interactions),
            'like_ratio': sum(1 for i in interactions if i['label'] == 1) / len(interactions),
            'avg_interactions_per_user': len(interactions) / len(user_features)
        }
        
        return content_df, interactions_df
    
    def _create_content_features(self, reviews: List[Dict]) -> List[Dict]:
        """レビューからコンテンツ特徴量を生成"""
        
        content_dict = {}
        
        for review in reviews:
            content_id = review.get('content_id', '')
            if not content_id or content_id in content_dict:
                continue
            
            # 基本情報
            content_features = {
                'content_id': content_id,
                'title': review.get('title', ''),
                'category': review.get('category', 'ビデオ(動画)'),
                'content_title': review.get('content_title', ''),
                'content_url': review.get('content_url', ''),
            }
            
            # テキスト特徴量（タイトル+コンテンツタイトル）
            combined_text = f"{content_features['title']} {content_features['content_title']}"
            content_features['combined_text'] = combined_text
            
            # ジャンル推論（タイトルからキーワード抽出）
            content_features['inferred_genres'] = self._infer_genres_from_text(combined_text)
            
            # 人気度（レビュー数の代理指標として、後で計算）
            content_features['review_count'] = 0
            content_features['avg_rating'] = 0.0
            
            content_dict[content_id] = content_features
        
        # レビュー統計を計算
        for review in reviews:
            content_id = review.get('content_id', '')
            if content_id in content_dict:
                content_dict[content_id]['review_count'] += 1
                rating = review.get('rating')
                if rating is not None:
                    # 累積平均の更新
                    current_avg = content_dict[content_id]['avg_rating']
                    count = content_dict[content_id]['review_count']
                    content_dict[content_id]['avg_rating'] = (current_avg * (count - 1) + rating) / count
        
        return list(content_dict.values())
    
    def _infer_genres_from_text(self, text: str) -> List[str]:
        """テキストからジャンルを推論"""
        text_lower = text.lower()
        
        genre_keywords = {
            'vr': ['vr', 'バーチャル', '仮想現実'],
            'amateur': ['素人', 'アマチュア', '一般人', '初撮り'],
            'mature': ['熟女', '人妻', 'ミセス', '年上'],
            'young': ['学生', '制服', 'jk', '若い', '10代', '新人'],
            'big_breasts': ['爆乳', '巨乳', 'lカップ', 'パイズリ', 'おっぱい'],
            'fetish': ['フェチ', 'm女', 'sm', '調教', '変態'],
            'anal': ['アナル', 'お尻', '肛門'],
            'group': ['乱交', '3p', '4p', '複数', 'ハーレム'],
            'cosplay': ['コスプレ', 'コス', 'メイド']
        }
        
        detected_genres = []
        for genre, keywords in genre_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_genres.append(genre)
        
        return detected_genres or ['general']
    
    def _create_user_interactions(self, pseudo_users: List[Dict], content_features: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """ユーザー特徴量とインタラクションデータを作成"""
        
        user_features = []
        interactions = []
        content_id_set = set(c['content_id'] for c in content_features)
        
        for user in pseudo_users:
            user_id = user['user_id']
            profile = user['profile']
            actions = user['actions']
            
            # ユーザー特徴量
            user_feature = {
                'user_id': user_id,
                'review_count': profile.get('review_count', 0),
                'avg_rating': profile.get('rating_stats', {}).get('avg_rating', 3.0),
                'rating_std': profile.get('rating_stats', {}).get('std_rating', 1.0),
                'engagement_level': profile.get('engagement_level', 0.5),
                'genre_preferences': profile.get('genre_preferences', {}),
                'conversion_method': user.get('conversion_method', 'rating_based'),
                'confidence_score': user.get('confidence_score', 0.5)
            }
            user_features.append(user_feature)
            
            # インタラクションデータ
            for action in actions:
                content_id = action.get('content_id', '')
                if content_id not in content_id_set:
                    continue  # 存在しないコンテンツはスキップ
                
                interaction = {
                    'user_id': user_id,
                    'content_id': content_id,
                    'label': 1 if action.get('action') == 'like' else 0,
                    'confidence': action.get('confidence', 0.5),
                    'source': action.get('source', 'unknown'),
                    'timestamp': action.get('timestamp', datetime.now().isoformat())
                }
                interactions.append(interaction)
        
        return user_features, interactions
    
    def build_two_tower_model(self, user_vocab_size: int, item_vocab_size: int, 
                             user_feature_dim: int, item_feature_dim: int) -> tf.keras.Model:
        """Two-Towerモデルを構築"""
        
        logger.info("Two-Towerモデル構築開始...")
        
        # User Tower
        user_id_input = tf.keras.layers.Input(shape=(), name='user_id')
        user_features_input = tf.keras.layers.Input(shape=(user_feature_dim,), name='user_features')
        
        user_embedding = tf.keras.layers.Embedding(user_vocab_size, self.embedding_dim)(user_id_input)
        user_embedding = tf.keras.layers.Flatten()(user_embedding)
        
        user_dense = tf.keras.layers.Dense(128, activation='relu')(user_features_input)
        user_dense = tf.keras.layers.Dropout(0.3)(user_dense)
        user_dense = tf.keras.layers.Dense(64, activation='relu')(user_dense)
        
        user_concat = tf.keras.layers.Concatenate()([user_embedding, user_dense])
        user_tower_output = tf.keras.layers.Dense(self.embedding_dim, activation='tanh', name='user_tower')(user_concat)
        
        # Item Tower
        item_id_input = tf.keras.layers.Input(shape=(), name='item_id')
        item_features_input = tf.keras.layers.Input(shape=(item_feature_dim,), name='item_features')
        
        item_embedding = tf.keras.layers.Embedding(item_vocab_size, self.embedding_dim)(item_id_input)
        item_embedding = tf.keras.layers.Flatten()(item_embedding)
        
        item_dense = tf.keras.layers.Dense(128, activation='relu')(item_features_input)
        item_dense = tf.keras.layers.Dropout(0.3)(item_dense)
        item_dense = tf.keras.layers.Dense(64, activation='relu')(item_dense)
        
        item_concat = tf.keras.layers.Concatenate()([item_embedding, item_dense])
        item_tower_output = tf.keras.layers.Dense(self.embedding_dim, activation='tanh', name='item_tower')(item_concat)
        
        # Interaction (Dot Product + Output)
        interaction = tf.keras.layers.Dot(axes=1)([user_tower_output, item_tower_output])
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='interaction')(interaction)
        
        # Full Model
        model = tf.keras.Model(
            inputs=[user_id_input, user_features_input, item_id_input, item_features_input],
            outputs=output
        )
        
        # Separate Tower Models for inference
        self.user_tower = tf.keras.Model(
            inputs=[user_id_input, user_features_input],
            outputs=user_tower_output
        )
        
        self.item_tower = tf.keras.Model(
            inputs=[item_id_input, item_features_input],
            outputs=item_tower_output
        )
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Two-Towerモデル構築完了")
        logger.info(f"User Tower output shape: {user_tower_output.shape}")
        logger.info(f"Item Tower output shape: {item_tower_output.shape}")
        
        return model
    
    def prepare_features(self, content_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """特徴量エンジニアリング"""
        
        logger.info("特徴量エンジニアリング開始...")
        
        # User IDs and Item IDs
        unique_users = interactions_df['user_id'].unique()
        unique_items = content_df['content_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        # User Features (simple numerical features)
        user_stats = interactions_df.groupby('user_id').agg({
            'label': ['count', 'mean'],
            'confidence': 'mean'
        }).round(3)
        
        user_features = []
        for user_id in unique_users:
            if user_id in user_stats.index:
                stats = user_stats.loc[user_id]
                user_features.append([
                    stats[('label', 'count')],      # interaction count
                    stats[('label', 'mean')],       # like ratio
                    stats[('confidence', 'mean')]   # avg confidence
                ])
            else:
                user_features.append([0, 0.5, 0.5])  # default values
        
        user_features_array = np.array(user_features, dtype=np.float32)
        
        # Item Features (text + numerical)
        item_texts = content_df['combined_text'].fillna('').tolist()
        
        if len(item_texts) > 0:
            text_features = self.text_vectorizer.fit_transform(item_texts).toarray()
        else:
            text_features = np.zeros((len(unique_items), 1000))
        
        # Category encoding
        categories = content_df['category'].fillna('unknown').tolist()
        if len(set(categories)) > 1:
            category_encoded = self.category_encoder.fit_transform(categories).reshape(-1, 1)
        else:
            category_encoded = np.zeros((len(categories), 1))
        
        # Numerical features
        numerical_features = content_df[['review_count', 'avg_rating']].fillna(0).values
        
        # Combine item features
        item_features_array = np.concatenate([
            text_features,
            category_encoded,
            numerical_features
        ], axis=1).astype(np.float32)
        
        # Scale features
        user_features_array = self.scaler.fit_transform(user_features_array)
        
        feature_info = {
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'user_vocab_size': len(unique_users),
            'item_vocab_size': len(unique_items),
            'user_feature_dim': user_features_array.shape[1],
            'item_feature_dim': item_features_array.shape[1]
        }
        
        logger.info(f"特徴量準備完了:")
        logger.info(f"  User特徴量: {user_features_array.shape}")
        logger.info(f"  Item特徴量: {item_features_array.shape}")
        logger.info(f"  User vocabulary: {feature_info['user_vocab_size']}")
        logger.info(f"  Item vocabulary: {feature_info['item_vocab_size']}")
        
        return feature_info, user_features_array, item_features_array, interactions_df
    
    def create_training_dataset(self, interactions_df: pd.DataFrame, 
                              user_features_array: np.ndarray, item_features_array: np.ndarray,
                              feature_info: Dict) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """訓練用データセットを作成"""
        
        logger.info("訓練データセット作成開始...")
        
        # Prepare inputs
        user_ids = np.array([feature_info['user_to_idx'].get(uid, 0) for uid in interactions_df['user_id']])
        item_ids = np.array([feature_info['item_to_idx'].get(iid, 0) for iid in interactions_df['content_id']])
        labels = interactions_df['label'].values.astype(np.float32)
        
        # User and item features for each interaction
        user_feats_for_interactions = user_features_array[user_ids]
        item_feats_for_interactions = item_features_array[item_ids]
        
        # Train/validation split
        indices = np.arange(len(interactions_df))
        train_indices, val_indices = train_test_split(
            indices, test_size=self.validation_split, random_state=42, 
            stratify=labels
        )
        
        def create_dataset(indices):
            # Convert to tensors first to avoid None values
            user_id_tensor = tf.convert_to_tensor(user_ids[indices], dtype=tf.int32)
            user_features_tensor = tf.convert_to_tensor(user_feats_for_interactions[indices], dtype=tf.float32)
            item_id_tensor = tf.convert_to_tensor(item_ids[indices], dtype=tf.int32)
            item_features_tensor = tf.convert_to_tensor(item_feats_for_interactions[indices], dtype=tf.float32)
            labels_tensor = tf.convert_to_tensor(labels[indices], dtype=tf.float32)
            
            dataset = tf.data.Dataset.from_tensor_slices((
                (user_id_tensor, user_features_tensor, item_id_tensor, item_features_tensor),
                labels_tensor
            ))
            
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return dataset
        
        train_dataset = create_dataset(train_indices)
        val_dataset = create_dataset(val_indices)
        
        logger.info(f"データセット作成完了:")
        logger.info(f"  訓練データ: {len(train_indices)} サンプル")
        logger.info(f"  検証データ: {len(val_indices)} サンプル")
        
        return train_dataset, val_dataset
    
    def train_model(self, model: tf.keras.Model, train_dataset: tf.data.Dataset, 
                   val_dataset: tf.data.Dataset) -> tf.keras.callbacks.History:
        """モデル訓練"""
        
        logger.info("Two-Towerモデル訓練開始...")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        
        # Training
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_stats['training_history'] = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        
        logger.info("モデル訓練完了")
        return history
    
    def evaluate_model(self, model: tf.keras.Model, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """モデル評価"""
        
        logger.info("モデル評価開始...")
        
        # Predictions
        y_pred = model.predict(val_dataset)
        
        # Ground truth - correct format for the new dataset structure
        y_true = np.concatenate([batch[1].numpy() for batch in val_dataset])
        y_pred_flat = y_pred.flatten()
        
        # Metrics
        auc_score = roc_auc_score(y_true, y_pred_flat)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_flat)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'auc_roc': float(auc_score),
            'auc_pr': float(pr_auc),
            'accuracy': float(np.mean((y_pred_flat > 0.5) == y_true))
        }
        
        self.training_stats['evaluation_metrics'] = metrics
        
        logger.info(f"評価完了:")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def save_model(self, model: tf.keras.Model, feature_info: Dict, 
                  output_dir: str = "../models/rating_based_two_tower"):
        """モデル保存"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full model
        model.save(output_path / "full_model.keras")
        
        # Save tower models
        if self.user_tower:
            self.user_tower.save(output_path / "user_tower.keras")
        if self.item_tower:
            self.item_tower.save(output_path / "item_tower.keras")
        
        # Save preprocessing objects
        with open(output_path / "preprocessors.pkl", 'wb') as f:
            pickle.dump({
                'text_vectorizer': self.text_vectorizer,
                'category_encoder': self.category_encoder,
                'scaler': self.scaler
            }, f)
        
        # Save feature info and stats
        with open(output_path / "feature_info.json", 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "training_stats.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"モデル保存完了: {output_path}")
    
    def run_training(self) -> Dict[str, Any]:
        """フル訓練パイプライン実行"""
        
        logger.info("=== 評価ベースTwo-Tower訓練開始 ===")
        
        # Step 1: データ読み込み
        pseudo_users, reviews = self.load_pseudo_user_data()
        if not pseudo_users or not reviews:
            logger.error("データ読み込み失敗")
            return {}
        
        # Step 2: 訓練データ準備  
        content_df, interactions_df = self.prepare_training_data(pseudo_users, reviews)
        
        # Step 3: 特徴量エンジニアリング
        feature_info, user_features, item_features, interactions_df = self.prepare_features(content_df, interactions_df)
        
        # Step 4: モデル構築
        model = self.build_two_tower_model(
            feature_info['user_vocab_size'],
            feature_info['item_vocab_size'], 
            feature_info['user_feature_dim'],
            feature_info['item_feature_dim']
        )
        
        # Step 5: データセット作成
        train_dataset, val_dataset = self.create_training_dataset(
            interactions_df, user_features, item_features, feature_info
        )
        
        # Step 6: 訓練実行
        history = self.train_model(model, train_dataset, val_dataset)
        
        # Step 7: 評価
        metrics = self.evaluate_model(model, val_dataset)
        
        # Step 8: 保存
        self.save_model(model, feature_info)
        
        # 完了統計
        self.training_stats['end_time'] = datetime.now().isoformat()
        self.training_stats['success'] = True
        
        logger.info("=== 評価ベースTwo-Tower訓練完了 ===")
        logger.info(f"最終AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return self.training_stats

def main():
    """メイン実行"""
    trainer = RatingBasedTwoTowerTrainer(
        embedding_dim=64,
        learning_rate=0.001,
        batch_size=256,
        epochs=50
    )
    
    results = trainer.run_training()
    
    if results.get('success'):
        print("\n🎉 評価ベースTwo-Tower訓練成功!")
        metrics = results.get('evaluation_metrics', {})
        print(f"AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
        print(f"AUC-PR: {metrics.get('auc_pr', 0):.4f}")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    else:
        print("❌ 訓練失敗")

if __name__ == "__main__":
    main()