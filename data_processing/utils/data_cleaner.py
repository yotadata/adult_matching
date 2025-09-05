"""
データクリーニングユーティリティ

スクレイピングで取得した生データを
機械学習用に前処理・クリーニングする機能
"""

import pandas as pd
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

class ReviewDataCleaner:
    """レビューデータのクリーニングクラス"""
    
    def __init__(self):
        self.stop_words = {
            'japanese': [
                'の', 'に', 'は', 'を', 'が', 'で', 'て', 'と', 'から', 'まで',
                'だ', 'である', 'です', 'ます', 'した', 'する', 'される',
                '！', '！！', '？', '…', '・', '～', '〜'
            ]
        }
        
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """生データファイルを読み込む"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            else:
                raise ValueError("サポートされていないファイル形式です")
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return pd.DataFrame()
    
    def clean_review_text(self, text: str) -> str:
        """レビューテキストをクリーニング"""
        if not isinstance(text, str):
            return ""
        
        # HTML タグの除去
        text = re.sub(r'<[^>]+>', '', text)
        
        # 特殊文字・記号の正規化
        text = re.sub(r'[！]{2,}', '！', text)  # 連続する感嘆符
        text = re.sub(r'[？]{2,}', '？', text)  # 連続する疑問符
        text = re.sub(r'[…。]{2,}', '…', text)  # 連続する省略記号
        
        # 数字・英字・カタカナ・ひらがな・漢字・基本記号以外を除去
        text = re.sub(r'[^\w\s！？。、…〜～\-\(\)（）【】「」『』]', '', text)
        
        # 連続する空白の除去
        text = re.sub(r'\s+', ' ', text)
        
        # 前後の空白を除去
        text = text.strip()
        
        return text
    
    def extract_rating(self, rating_value: Any) -> Optional[float]:
        """評価値を抽出・正規化"""
        if pd.isna(rating_value):
            return None
            
        if isinstance(rating_value, str):
            # 数値を含む文字列から抽出
            match = re.search(r'(\d+(?:\.\d+)?)', rating_value)
            if match:
                rating = float(match.group(1))
            else:
                return None
        else:
            try:
                rating = float(rating_value)
            except:
                return None
        
        # 0-5の範囲に正規化
        if rating > 10:
            rating = rating / 2  # 10点満点を5点満点に変換
        elif rating > 5:
            rating = 5  # 上限を5に設定
        
        return max(0, min(5, rating))
    
    def filter_valid_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """有効なレビューのみを抽出"""
        # 基本的な条件設定
        min_text_length = 20
        max_text_length = 10000
        
        filtered_df = df.copy()
        
        # レビューテキストの長さでフィルタ
        if 'review_text' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['review_text'].str.len() >= min_text_length) &
                (filtered_df['review_text'].str.len() <= max_text_length)
            ]
        
        # 重複除去（テキストの最初の100文字で判定）
        if 'review_text' in filtered_df.columns and len(filtered_df) > 0:
            filtered_df['text_snippet'] = filtered_df['review_text'].str[:100]
            filtered_df = filtered_df.drop_duplicates(subset=['text_snippet'])
            filtered_df = filtered_df.drop('text_snippet', axis=1)
        
        return filtered_df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """レビューから特徴量を抽出"""
        if df.empty:
            return df
            
        features_df = df.copy()
        
        # テキスト関連の特徴量
        if 'review_text' in features_df.columns:
            features_df['text_length'] = features_df['review_text'].str.len()
            features_df['sentence_count'] = features_df['review_text'].str.count('[。！？]')
            features_df['exclamation_count'] = features_df['review_text'].str.count('[！!]')
            features_df['question_count'] = features_df['review_text'].str.count('[？?]')
            
            # ポジティブ・ネガティブキーワード数（簡易版）
            positive_keywords = ['良い', 'すごい', '最高', '素晴らしい', '面白い', '楽しい']
            negative_keywords = ['悪い', 'つまらない', '最悪', 'ダメ', '嫌い', 'がっかり']
            
            features_df['positive_words'] = features_df['review_text'].apply(
                lambda x: sum(keyword in str(x) for keyword in positive_keywords)
            )
            features_df['negative_words'] = features_df['review_text'].apply(
                lambda x: sum(keyword in str(x) for keyword in negative_keywords)
            )
        
        return features_df
    
    def process_dataset(self, input_file: str, output_file: str) -> pd.DataFrame:
        """データセット全体の処理パイプライン"""
        print(f"データ処理開始: {input_file}")
        
        # データ読み込み
        df = self.load_raw_data(input_file)
        print(f"読み込み完了: {len(df)} 件")
        
        if df.empty:
            print("データが空です")
            return df
        
        # テキストクリーニング
        if 'review_text' in df.columns:
            df['review_text'] = df['review_text'].apply(self.clean_review_text)
            print("テキストクリーニング完了")
        
        # 評価値の正規化
        if 'rating' in df.columns:
            df['rating'] = df['rating'].apply(self.extract_rating)
            print("評価値正規化完了")
        
        # 有効なレビューのフィルタリング
        df = self.filter_valid_reviews(df)
        print(f"フィルタリング後: {len(df)} 件")
        
        # 特徴量抽出
        df = self.extract_features(df)
        print("特徴量抽出完了")
        
        # 結果保存
        if output_file.endswith('.json'):
            df.to_json(output_file, orient='records', force_ascii=False, indent=2)
        else:
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"処理完了: {output_file}")
        return df

def main():
    """メイン処理"""
    cleaner = ReviewDataCleaner()
    
    # 生データの処理
    input_file = "../raw_data/dmm_reviews_cookie_20250903_221347.json"
    output_file = "../processed_data/cleaned_reviews.json"
    
    processed_df = cleaner.process_dataset(input_file, output_file)
    
    # 統計情報の出力
    if not processed_df.empty:
        print("\n=== データ統計 ===")
        print(f"総レビュー数: {len(processed_df)}")
        
        if 'rating' in processed_df.columns:
            valid_ratings = processed_df['rating'].dropna()
            if len(valid_ratings) > 0:
                print(f"評価あり: {len(valid_ratings)} 件")
                print(f"平均評価: {valid_ratings.mean():.2f}")
        
        if 'text_length' in processed_df.columns:
            print(f"平均文字数: {processed_df['text_length'].mean():.0f}")
            print(f"文字数範囲: {processed_df['text_length'].min()}-{processed_df['text_length'].max()}")

if __name__ == "__main__":
    main()