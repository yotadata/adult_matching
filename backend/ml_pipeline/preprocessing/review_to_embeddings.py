"""
レビューデータを埋め込みベクトルに変換

クリーニングされたレビューデータを
Two-Tower モデル用の埋め込みベクトルに変換する
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class ReviewEmbeddingProcessor:
    """レビューデータを埋め込みベクトルに変換するクラス"""
    
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese"):
        """
        初期化
        
        Args:
            model_name: 使用するBERTモデル名（日本語BERT）
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_model(self):
        """BERTモデルとトークナイザーを読み込み"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print(f"モデル読み込み完了: {self.model_name}")
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            # フォールバック: シンプルなベクトル化
            print("シンプルなベクトル化を使用します")
    
    def text_to_embedding(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            texts: テキストのリスト
            max_length: 最大トークン長
            
        Returns:
            埋め込みベクトルの配列
        """
        embeddings = []
        
        if self.model is None or self.tokenizer is None:
            # シンプルなベクトル化（TF-IDFの簡易版）
            return self._simple_vectorization(texts)
        
        print(f"BERT埋め込み変換開始: {len(texts)} 件")
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"処理中: {i+1}/{len(texts)}")
                
            try:
                # トークン化
                inputs = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # BERTで埋め込みベクトル取得
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # [CLS] トークンの埋め込みベクトルを使用
                    embedding = outputs.last_hidden_state[0, 0, :].numpy()
                    embeddings.append(embedding)
                    
            except Exception as e:
                print(f"テキスト {i} の処理エラー: {e}")
                # ゼロベクトルで埋める
                embeddings.append(np.zeros(768))  # BERT-baseの次元数
        
        return np.array(embeddings)
    
    def _simple_vectorization(self, texts: List[str]) -> np.ndarray:
        """シンプルなベクトル化（BERTが使えない場合のフォールバック）"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        import MeCab
        
        print("シンプルなTF-IDFベクトル化を実行")
        
        def tokenize_japanese(text):
            """日本語テキストの形態素解析"""
            try:
                tagger = MeCab.Tagger("-Owakati")
                return tagger.parse(text).strip().split()
            except:
                # MeCabが使えない場合は文字レベル
                return list(text.replace(' ', ''))
        
        # 前処理されたテキスト
        processed_texts = []
        for text in texts:
            tokens = tokenize_japanese(str(text)[:1000])  # 1000文字まで
            processed_texts.append(' '.join(tokens))
        
        # TF-IDFベクトル化
        vectorizer = TfidfVectorizer(
            max_features=1000,  # 上位1000特徴量
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        vectors = vectorizer.fit_transform(processed_texts)
        return vectors.toarray()
    
    def create_user_item_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        ユーザーとアイテムの特徴量を作成
        
        Args:
            df: レビューデータのDataFrame
            
        Returns:
            user_features, item_features のタプル
        """
        # ユーザー特徴量（レビュアーベース）
        user_features = []
        if 'reviewer_id' in df.columns:
            # レビュアーごとの統計
            reviewer_stats = df.groupby('reviewer_id').agg({
                'rating': ['count', 'mean', 'std'],
                'text_length': 'mean',
                'positive_words': 'mean',
                'negative_words': 'mean'
            }).fillna(0)
            
            # カラム名を平坦化
            reviewer_stats.columns = ['_'.join(col).strip() for col in reviewer_stats.columns]
            user_features = reviewer_stats.values
        else:
            # レビュアーIDがない場合はダミー特徴量
            user_features = np.ones((len(df), 5))
        
        # アイテム特徴量（コンテンツベース）
        item_features = []
        if 'content_id' in df.columns:
            # コンテンツごとの統計
            content_stats = df.groupby('content_id').agg({
                'rating': ['count', 'mean', 'std'],
                'text_length': 'mean',
                'positive_words': 'mean',
                'negative_words': 'mean'
            }).fillna(0)
            
            content_stats.columns = ['_'.join(col).strip() for col in content_stats.columns]
            item_features = content_stats.values
        else:
            # コンテンツIDがない場合はテキストベースの特徴量
            text_features = df[['text_length', 'positive_words', 'negative_words']].fillna(0)
            if 'rating' in df.columns:
                rating_features = df[['rating']].fillna(df['rating'].mean())
                item_features = np.concatenate([text_features.values, rating_features.values], axis=1)
            else:
                item_features = text_features.values
        
        return user_features, item_features
    
    def prepare_training_data(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """
        訓練用データを準備
        
        Args:
            input_file: 前処理済みレビューデータファイル
            output_dir: 出力ディレクトリ
            
        Returns:
            データセット情報の辞書
        """
        print(f"訓練データ準備開始: {input_file}")
        
        # データ読み込み
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(input_file)
        
        if df.empty:
            raise ValueError("データが空です")
        
        print(f"データ読み込み完了: {len(df)} 件")
        
        # BERTモデル読み込み
        self.load_model()
        
        # レビューテキストの埋め込みベクトル化
        if 'review_text' in df.columns:
            review_texts = df['review_text'].fillna('').tolist()
            text_embeddings = self.text_to_embedding(review_texts)
            print(f"テキスト埋め込み完了: {text_embeddings.shape}")
        else:
            text_embeddings = np.random.randn(len(df), 768)  # ダミーデータ
        
        # ユーザー・アイテム特徴量
        user_features, item_features = self.create_user_item_features(df)
        print(f"ユーザー特徴量: {user_features.shape}")
        print(f"アイテム特徴量: {item_features.shape}")
        
        # 評価値の処理
        if 'rating' in df.columns:
            ratings = df['rating'].fillna(df['rating'].mean()).values
        else:
            # 評価値がない場合は、テキストの感情極性から推定
            pos_ratio = df['positive_words'] / (df['positive_words'] + df['negative_words'] + 1)
            ratings = pos_ratio * 5  # 0-5スケール
        
        # 正規化
        text_embeddings = self.scaler.fit_transform(text_embeddings)
        
        # 訓練・テストデータ分割
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # データセットを保存
        dataset = {
            'text_embeddings_train': text_embeddings[train_idx],
            'text_embeddings_test': text_embeddings[test_idx],
            'user_features_train': user_features[train_idx] if len(user_features) == len(df) else user_features,
            'user_features_test': user_features[test_idx] if len(user_features) == len(df) else user_features,
            'item_features_train': item_features[train_idx] if len(item_features) == len(df) else item_features,
            'item_features_test': item_features[test_idx] if len(item_features) == len(df) else item_features,
            'ratings_train': ratings[train_idx],
            'ratings_test': ratings[test_idx],
            'metadata': {
                'total_samples': len(df),
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'embedding_dim': text_embeddings.shape[1],
                'user_feature_dim': user_features.shape[1] if user_features.ndim > 1 else 1,
                'item_feature_dim': item_features.shape[1] if item_features.ndim > 1 else 1,
                'created_at': datetime.now().isoformat()
            }
        }
        
        # NPZ形式で保存
        np.savez_compressed(
            f"{output_dir}/training_data.npz",
            **{k: v for k, v in dataset.items() if k != 'metadata'}
        )
        
        # メタデータをJSONで保存
        with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
            json.dump(dataset['metadata'], f, indent=2)
        
        print(f"訓練データ保存完了: {output_dir}")
        return dataset

def main():
    """メイン処理"""
    processor = ReviewEmbeddingProcessor()
    
    # 前処理済みデータから訓練データを作成
    input_file = "../processed_data/cleaned_reviews.json"
    output_dir = "../../ml_pipeline/training/data"
    
    # 出力ディレクトリを作成
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        dataset = processor.prepare_training_data(input_file, output_dir)
        
        print("\n=== データセット情報 ===")
        metadata = dataset['metadata']
        for key, value in metadata.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"処理エラー: {e}")

if __name__ == "__main__":
    main()