"""
バッチレビューデータ統合・クリーニングシステム

大規模収集したレビューデータを統合し、ML学習用に前処理する
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import re

class BatchDataIntegrator:
    """バッチレビューデータ統合・クリーニングクラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 統計情報
        self.stats = {
            'total_files': 0,
            'total_reviews': 0,
            'valid_reviews': 0,
            'duplicate_reviews': 0,
            'invalid_reviews': 0,
            'reviewers_count': 0,
            'avg_reviews_per_reviewer': 0,
            'rating_coverage': 0,
            'processing_start': datetime.now().isoformat()
        }
        
        # データ品質チェック設定
        self.quality_settings = {
            'min_text_length': 10,      # 最低テキスト長
            'max_text_length': 10000,   # 最大テキスト長
            'min_rating': 1.0,          # 最低評価値
            'max_rating': 5.0,          # 最高評価値
            'required_fields': ['reviewer_id', 'content_id']  # 必須フィールド
        }
    
    def load_batch_reviews(self, batch_dir: str = "../raw_data/batch_reviews") -> List[Dict[str, Any]]:
        """バッチディレクトリから全レビューファイルを読み込み"""
        batch_path = Path(batch_dir)
        
        if not batch_path.exists():
            self.logger.error(f"バッチディレクトリが見つかりません: {batch_dir}")
            return []
        
        review_files = list(batch_path.glob("reviewer_*.json"))
        self.stats['total_files'] = len(review_files)
        self.logger.info(f"発見されたレビューファイル数: {len(review_files)}")
        
        all_reviews = []
        successful_files = 0
        
        for file_path in review_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
                
                if isinstance(reviews, list):
                    all_reviews.extend(reviews)
                    successful_files += 1
                    self.logger.debug(f"{file_path.name}: {len(reviews)}件読み込み")
                else:
                    self.logger.warning(f"不正なデータ形式: {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"ファイル読み込みエラー {file_path.name}: {e}")
                continue
        
        self.stats['total_reviews'] = len(all_reviews)
        self.logger.info(f"統合読み込み完了: {successful_files}/{len(review_files)}ファイル, {len(all_reviews)}件レビュー")
        
        return all_reviews
    
    def clean_and_validate_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """レビューデータのクリーニングと検証"""
        self.logger.info(f"データクリーニング開始: {len(reviews)}件")
        
        valid_reviews = []
        duplicate_tracker = set()
        
        for review in reviews:
            # 基本検証
            if not self._is_valid_review(review):
                self.stats['invalid_reviews'] += 1
                continue
            
            # 重複チェック
            review_key = self._generate_review_key(review)
            if review_key in duplicate_tracker:
                self.stats['duplicate_reviews'] += 1
                self.logger.debug(f"重複レビュー検出: {review_key}")
                continue
            
            duplicate_tracker.add(review_key)
            
            # データクリーニング実行
            cleaned_review = self._clean_review(review)
            if cleaned_review:
                valid_reviews.append(cleaned_review)
        
        self.stats['valid_reviews'] = len(valid_reviews)
        self.logger.info(f"クリーニング完了: {len(valid_reviews)}件が有効")
        
        return valid_reviews
    
    def _is_valid_review(self, review: Dict[str, Any]) -> bool:
        """レビューの基本検証"""
        # 必須フィールドチェック
        for field in self.quality_settings['required_fields']:
            if not review.get(field):
                return False
        
        # テキスト長チェック
        review_text = review.get('review_text', '')
        if not isinstance(review_text, str):
            return False
        
        text_length = len(review_text.strip())
        if text_length < self.quality_settings['min_text_length']:
            return False
        
        if text_length > self.quality_settings['max_text_length']:
            return False
        
        # 評価値チェック（ある場合のみ）
        rating = review.get('rating')
        if rating is not None:
            try:
                rating_float = float(rating)
                if rating_float < self.quality_settings['min_rating'] or rating_float > self.quality_settings['max_rating']:
                    return False
            except (ValueError, TypeError):
                # 評価値が不正な場合はNoneに設定
                review['rating'] = None
        
        return True
    
    def _generate_review_key(self, review: Dict[str, Any]) -> str:
        """重複チェック用のキー生成"""
        reviewer_id = review.get('reviewer_id', '')
        content_id = review.get('content_id', '')
        review_text = review.get('review_text', '')
        
        # テキストの最初の100文字をハッシュに含める
        text_snippet = review_text[:100] if review_text else ''
        
        return f"{reviewer_id}_{content_id}_{hash(text_snippet)}"
    
    def _clean_review(self, review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """個別レビューのクリーニング"""
        try:
            cleaned = review.copy()
            
            # テキストクリーニング
            if 'review_text' in cleaned:
                cleaned_text = self._clean_text(cleaned['review_text'])
                cleaned['review_text'] = cleaned_text
                
                # クリーニング後の長さチェック
                if len(cleaned_text.strip()) < self.quality_settings['min_text_length']:
                    return None
            
            # 評価値の正規化
            if 'rating' in cleaned and cleaned['rating'] is not None:
                try:
                    cleaned['rating'] = float(cleaned['rating'])
                except (ValueError, TypeError):
                    cleaned['rating'] = None
            
            # 日付の正規化
            if 'write_date' in cleaned:
                cleaned['write_date'] = self._normalize_date(cleaned['write_date'])
            
            # helpful_countの正規化
            if 'helpful_count' in cleaned:
                try:
                    cleaned['helpful_count'] = int(cleaned['helpful_count'])
                except (ValueError, TypeError):
                    cleaned['helpful_count'] = 0
            
            # 追加メタデータ
            cleaned['processed_at'] = datetime.now().isoformat()
            cleaned['text_length'] = len(cleaned.get('review_text', ''))
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"レビュークリーニングエラー: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """テキストクリーニング"""
        if not isinstance(text, str):
            return ""
        
        # 基本的なクリーニング
        cleaned = text.strip()
        
        # 連続する空白・改行を正規化
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        # 特殊文字の正規化
        cleaned = cleaned.replace('\u3000', ' ')  # 全角スペース
        cleaned = cleaned.replace('\xa0', ' ')    # ノーブレークスペース
        
        return cleaned.strip()
    
    def _normalize_date(self, date_str: Any) -> Optional[str]:
        """日付文字列の正規化"""
        if not date_str:
            return None
        
        try:
            date_str = str(date_str)
            # 基本的な日付形式をISO形式に変換
            if len(date_str) >= 19:
                return date_str[:19]  # YYYY-MM-DD HH:MM:SS
            return date_str
        except:
            return None
    
    def generate_statistics(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """統計情報生成"""
        if not reviews:
            return {}
        
        # 基本統計
        reviewers = set(r.get('reviewer_id') for r in reviews)
        ratings = [r.get('rating') for r in reviews if r.get('rating') is not None]
        text_lengths = [r.get('text_length', 0) for r in reviews]
        
        self.stats.update({
            'reviewers_count': len(reviewers),
            'avg_reviews_per_reviewer': len(reviews) / len(reviewers) if reviewers else 0,
            'rating_coverage': len(ratings) / len(reviews) if reviews else 0
        })
        
        statistics = {
            'basic_stats': {
                'total_reviews': len(reviews),
                'unique_reviewers': len(reviewers),
                'unique_content': len(set(r.get('content_id') for r in reviews)),
                'avg_reviews_per_reviewer': self.stats['avg_reviews_per_reviewer'],
                'processing_time': datetime.now().isoformat()
            },
            'rating_stats': {
                'rating_coverage': self.stats['rating_coverage'],
                'total_rated': len(ratings),
                'avg_rating': np.mean(ratings) if ratings else None,
                'std_rating': np.std(ratings) if ratings else None,
                'min_rating': min(ratings) if ratings else None,
                'max_rating': max(ratings) if ratings else None,
                'rating_distribution': self._calculate_rating_distribution(ratings)
            },
            'text_stats': {
                'avg_length': np.mean(text_lengths) if text_lengths else 0,
                'std_length': np.std(text_lengths) if text_lengths else 0,
                'min_length': min(text_lengths) if text_lengths else 0,
                'max_length': max(text_lengths) if text_lengths else 0,
                'total_characters': sum(text_lengths)
            },
            'quality_stats': self.stats
        }
        
        return statistics
    
    def _calculate_rating_distribution(self, ratings: List[float]) -> Dict[str, int]:
        """評価分布の計算"""
        if not ratings:
            return {}
        
        distribution = {}
        for rating in ratings:
            rating_int = int(rating)
            distribution[str(rating_int)] = distribution.get(str(rating_int), 0) + 1
        
        return distribution
    
    def save_integrated_data(self, reviews: List[Dict[str, Any]], 
                           output_file: str = "../processed_data/integrated_reviews.json") -> bool:
        """統合データの保存"""
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(reviews, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"統合データ保存完了: {output_file}")
            self.logger.info(f"保存件数: {len(reviews)} 件")
            
            return True
            
        except Exception as e:
            self.logger.error(f"統合データ保存エラー: {e}")
            return False
    
    def save_statistics(self, statistics: Dict[str, Any], 
                       output_file: str = "../processed_data/data_statistics.json") -> bool:
        """統計情報の保存"""
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"統計情報保存完了: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"統計情報保存エラー: {e}")
            return False
    
    def print_statistics(self, statistics: Dict[str, Any]):
        """統計情報の表示"""
        basic = statistics.get('basic_stats', {})
        rating = statistics.get('rating_stats', {})
        text = statistics.get('text_stats', {})
        quality = statistics.get('quality_stats', {})
        
        print("\n=== バッチデータ統合統計 ===")
        print(f"総レビュー数: {basic.get('total_reviews', 0):,}")
        print(f"ユニークレビュワー: {basic.get('unique_reviewers', 0):,}")
        print(f"ユニーク動画: {basic.get('unique_content', 0):,}")
        print(f"平均レビュー/人: {basic.get('avg_reviews_per_reviewer', 0):.1f}")
        
        print(f"\n評価データ:")
        print(f"  評価カバー率: {rating.get('rating_coverage', 0):.1%}")
        print(f"  平均評価: {rating.get('avg_rating', 0):.2f}")
        print(f"  評価分布: {rating.get('rating_distribution', {})}")
        
        print(f"\nテキスト統計:")
        print(f"  平均文字数: {text.get('avg_length', 0):.0f}")
        print(f"  総文字数: {text.get('total_characters', 0):,}")
        
        print(f"\n品質統計:")
        print(f"  処理ファイル数: {quality.get('total_files', 0)}")
        print(f"  有効レビュー: {quality.get('valid_reviews', 0)}")
        print(f"  重複除外: {quality.get('duplicate_reviews', 0)}")
        print(f"  無効除外: {quality.get('invalid_reviews', 0)}")
    
    def run(self, batch_dir: str = "../raw_data/batch_reviews") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """メイン実行関数"""
        self.logger.info("=== バッチデータ統合・クリーニング開始 ===")
        
        # Step 1: バッチデータ読み込み
        reviews = self.load_batch_reviews(batch_dir)
        if not reviews:
            self.logger.error("レビューデータの読み込みに失敗")
            return [], {}
        
        # Step 2: クリーニング・検証
        valid_reviews = self.clean_and_validate_reviews(reviews)
        if not valid_reviews:
            self.logger.error("有効なレビューデータがありません")
            return [], {}
        
        # Step 3: 統計情報生成
        statistics = self.generate_statistics(valid_reviews)
        
        # Step 4: 結果保存
        self.save_integrated_data(valid_reviews)
        self.save_statistics(statistics)
        
        # Step 5: 統計表示
        self.print_statistics(statistics)
        
        self.logger.info("=== バッチデータ統合完了 ===")
        return valid_reviews, statistics

def main():
    """メイン実行"""
    import logging
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # バッチデータ統合実行
    integrator = BatchDataIntegrator()
    reviews, statistics = integrator.run()
    
    if reviews:
        print(f"\n成功: {len(reviews)} 件のレビューデータを統合・クリーニングしました")

if __name__ == "__main__":
    main()