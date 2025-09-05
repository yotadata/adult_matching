"""
評価ベース疑似ユーザー生成システム（パターン1）

レビュー評価値ベースでユーザー行動を変換する学習モデル
- 評価4以上 → Like
- 評価3以下 → Skip
"""

import pandas as pd
import numpy as np
import json
import hashlib
import re
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
import logging

@dataclass
class RatingBasedPseudoUser:
    """評価ベース疑似ユーザーデータクラス"""
    user_id: str
    original_reviewer_id: str
    profile: Dict[str, Any]
    actions: List[Dict[str, Any]]
    confidence_score: float
    conversion_method: str = "rating_based"
    like_threshold: float = 4.0
    total_reviews: int = 0
    valid_reviews: int = 0

class RatingBasedUserGenerator:
    """評価値ベースの疑似ユーザー生成クラス（パターン1）"""
    
    def __init__(self, like_threshold: float = 4.0, salt: str = None):
        self.like_threshold = like_threshold  # 4.0以上でLike
        self.salt = salt or "rating_based_salt_2025"
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # ジャンル・キーワード辞書
        self.genre_keywords = self._load_genre_keywords()
        
        # 統計情報
        self.conversion_stats = {
            'total_reviewers': 0,
            'total_reviews': 0,
            'valid_ratings': 0,
            'like_actions': 0,
            'skip_actions': 0,
            'no_rating_actions': 0,
            'conversion_method': 'rating_based_v1.0'
        }
    
    def _load_genre_keywords(self) -> Dict[str, List[str]]:
        """ジャンルキーワード辞書をロード"""
        return {
            'amateur': ['素人', 'アマチュア', '一般人', '初撮り'],
            'mature': ['熟女', '人妻', 'ミセス', '年上'],
            'young': ['学生', '制服', 'JK', '若い', '10代', '新人'],
            'fetish': ['フェチ', 'M女', 'SM', '調教', '変態', 'マゾ'],
            'group': ['乱交', '3P', '4P', '複数', 'ハーレム'],
            'outdoor': ['野外', '露出', '屋外', 'アウトドア'],
            'cosplay': ['コスプレ', 'コス', '制服', 'メイド'],
            'vr': ['VR', 'バーチャル', '仮想現実'],
            'big_breasts': ['爆乳', '巨乳', 'Lカップ', 'パイズリ'],
            'anal': ['アナル', 'お尻', '肛門', 'アナルFUCK'],
            'creampie': ['中出し', '生中', '膣内射精']
        }
    
    def load_batch_reviews(self, batch_dir: str = "../raw_data/batch_reviews") -> List[Dict[str, Any]]:
        """バッチ収集済みレビューデータを読み込み"""
        import os
        from pathlib import Path
        
        batch_path = Path(batch_dir)
        all_reviews = []
        
        if not batch_path.exists():
            self.logger.warning(f"バッチディレクトリが見つかりません: {batch_dir}")
            return []
        
        review_files = list(batch_path.glob("reviewer_*.json"))
        self.logger.info(f"レビューファイル数: {len(review_files)}")
        
        for file_path in review_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
                    
                if isinstance(reviews, list):
                    all_reviews.extend(reviews)
                    self.logger.debug(f"{file_path.name}: {len(reviews)}件読み込み")
                else:
                    self.logger.warning(f"不正なデータ形式: {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"ファイル読み込みエラー {file_path.name}: {e}")
                continue
        
        self.logger.info(f"総レビュー数: {len(all_reviews)}")
        return all_reviews
    
    def generate_pseudo_users(self, reviews: List[Dict[str, Any]]) -> List[RatingBasedPseudoUser]:
        """
        評価値ベースで疑似ユーザーを生成
        
        Args:
            reviews: レビューデータのリスト
            
        Returns:
            評価ベース疑似ユーザーのリスト
        """
        self.logger.info(f"評価ベース疑似ユーザー生成開始: {len(reviews)} 件のレビューから")
        self.logger.info(f"変換ルール: {self.like_threshold}以上→Like, {self.like_threshold}未満→Skip")
        
        # Step 1: レビュアーごとにグループ化
        reviewer_groups = self._group_by_reviewer(reviews)
        self.logger.info(f"レビュアー数: {len(reviewer_groups)}")
        
        # Step 2: 各レビュアーから疑似ユーザー生成
        pseudo_users = []
        for reviewer_id, reviewer_reviews in reviewer_groups.items():
            pseudo_user = self._create_rating_based_user(reviewer_id, reviewer_reviews)
            if pseudo_user:
                pseudo_users.append(pseudo_user)
        
        # 統計情報更新
        self.conversion_stats['total_reviewers'] = len(reviewer_groups)
        self.conversion_stats['total_reviews'] = len(reviews)
        
        self.logger.info(f"評価ベース疑似ユーザー生成完了: {len(pseudo_users)} ユーザー")
        return pseudo_users
    
    def _group_by_reviewer(self, reviews: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """レビューをレビュアーIDごとにグループ化"""
        reviewer_groups = defaultdict(list)
        
        for review in reviews:
            reviewer_id = review.get('reviewer_id', 'unknown')
            reviewer_groups[reviewer_id].append(review)
        
        # 最低レビュー数でフィルタ（評価ベースは1件から可能）
        filtered_groups = {
            reviewer_id: reviews 
            for reviewer_id, reviews in reviewer_groups.items() 
            if len(reviews) >= 1  # 最低1件
        }
        
        return filtered_groups
    
    def _create_rating_based_user(self, reviewer_id: str, reviews: List[Dict[str, Any]]) -> Optional[RatingBasedPseudoUser]:
        """レビュアーから評価ベース疑似ユーザーを生成"""
        try:
            # 基本統計
            total_reviews = len(reviews)
            valid_ratings = [r for r in reviews if r.get('rating') is not None]
            valid_rating_count = len(valid_ratings)
            
            if valid_rating_count == 0:
                self.logger.warning(f"レビュアー {reviewer_id}: 有効な評価データなし")
                return None
            
            self.logger.debug(f"レビュアー {reviewer_id}: {total_reviews}件中{valid_rating_count}件に評価あり")
            
            # ユーザーID生成
            user_id = self._generate_user_id(reviewer_id, "rating_based")
            
            # プロフィール生成
            profile = self._generate_user_profile(reviews)
            
            # 評価ベース行動データ生成
            actions = self._generate_rating_based_actions(user_id, reviews)
            
            # 信頼度計算
            confidence_score = self._calculate_confidence(reviews, actions)
            
            # 統計更新
            like_count = sum(1 for a in actions if a['action'] == 'like')
            skip_count = sum(1 for a in actions if a['action'] == 'skip')
            no_rating_count = sum(1 for a in actions if a.get('source') == 'no_rating_inference')
            
            self.conversion_stats['valid_ratings'] += valid_rating_count
            self.conversion_stats['like_actions'] += like_count
            self.conversion_stats['skip_actions'] += skip_count
            self.conversion_stats['no_rating_actions'] += no_rating_count
            
            return RatingBasedPseudoUser(
                user_id=user_id,
                original_reviewer_id=reviewer_id,
                profile=profile,
                actions=actions,
                confidence_score=confidence_score,
                conversion_method="rating_based",
                like_threshold=self.like_threshold,
                total_reviews=total_reviews,
                valid_reviews=valid_rating_count
            )
            
        except Exception as e:
            self.logger.error(f"疑似ユーザー生成エラー {reviewer_id}: {e}")
            return None
    
    def _generate_user_id(self, reviewer_id: str, method: str) -> str:
        """ユーザーIDを生成"""
        combined = f"{reviewer_id}_{method}_{self.salt}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _generate_user_profile(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ユーザープロフィール生成"""
        # 基本統計
        ratings = [r.get('rating') for r in reviews if r.get('rating') is not None]
        review_lengths = [len(r.get('review_text', '')) for r in reviews]
        
        # ジャンル嗜好分析
        genre_preferences = self._analyze_genre_preferences(reviews)
        
        # 時間パターン分析
        temporal_pattern = self._analyze_temporal_patterns(reviews)
        
        profile = {
            'review_count': len(reviews),
            'valid_rating_count': len(ratings),
            'rating_stats': {
                'avg_rating': np.mean(ratings) if ratings else None,
                'std_rating': np.std(ratings) if ratings else None,
                'min_rating': min(ratings) if ratings else None,
                'max_rating': max(ratings) if ratings else None,
                'rating_coverage': len(ratings) / len(reviews) if reviews else 0
            },
            'review_stats': {
                'avg_length': np.mean(review_lengths),
                'std_length': np.std(review_lengths),
                'total_characters': sum(review_lengths)
            },
            'genre_preferences': genre_preferences,
            'temporal_pattern': temporal_pattern,
            'engagement_level': self._calculate_engagement_level(reviews)
        }
        
        return profile
    
    def _generate_rating_based_actions(self, user_id: str, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """評価値ベースで行動データを生成"""
        actions = []
        
        for review in reviews:
            rating = review.get('rating')
            content_id = review.get('content_id', f"unknown_{random.randint(10000, 99999)}")
            
            # 基本行動: 評価値ベース変換
            if rating is not None:
                if rating >= self.like_threshold:
                    action_type = 'like'
                    confidence = 0.9  # 高信頼度
                else:
                    action_type = 'skip'
                    confidence = 0.8  # 中高信頼度
                
                source = 'rating_conversion'
                metadata = {
                    'original_rating': rating,
                    'threshold': self.like_threshold,
                    'review_length': len(review.get('review_text', '')),
                    'has_text': bool(review.get('review_text', '').strip())
                }
            else:
                # 評価値がない場合：テキスト感情分析でフォールバック
                action_type, confidence = self._infer_action_from_text(review)
                source = 'no_rating_inference'
                metadata = {
                    'original_rating': None,
                    'inference_method': 'text_sentiment',
                    'review_length': len(review.get('review_text', ''))
                }
            
            # 行動データ作成
            action = {
                'user_id': user_id,
                'content_id': content_id,
                'action': action_type,
                'confidence': confidence,
                'timestamp': self._generate_timestamp(review),
                'session_id': f"session_{random.randint(1000, 9999)}",
                'source': source,
                'metadata': metadata
            }
            
            actions.append(action)
        
        return actions
    
    def _infer_action_from_text(self, review: Dict[str, Any]) -> Tuple[str, float]:
        """テキストから行動を推論（評価値がない場合のフォールバック）"""
        review_text = review.get('review_text', '').lower()
        
        # ポジティブキーワード
        positive_keywords = [
            '最高', '素晴らしい', '良い', 'おすすめ', '満足', '感動', 
            '完璧', 'エロい', '興奮', '良作', 'グッド'
        ]
        
        # ネガティブキーワード
        negative_keywords = [
            'だめ', '悪い', 'つまらない', '期待外れ', '残念', 
            '微妙', 'いまいち', '退屈', 'がっかり'
        ]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in review_text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in review_text)
        
        if positive_count > negative_count:
            return 'like', 0.6  # 中信頼度
        elif negative_count > positive_count:
            return 'skip', 0.6  # 中信頼度
        else:
            # 中性的またはキーワードなし → ランダム（やや Like 寄り）
            return random.choice(['like', 'like', 'skip']), 0.4  # 低信頼度
    
    def _analyze_genre_preferences(self, reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """ジャンル嗜好を分析"""
        genre_scores = defaultdict(int)
        total_reviews = len(reviews)
        
        for review in reviews:
            review_text = review.get('review_text', '').lower()
            title = review.get('title', '').lower()
            combined_text = review_text + ' ' + title
            
            # 各ジャンルのキーワード出現をカウント
            for genre, keywords in self.genre_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in combined_text:
                        genre_scores[genre] += 1
        
        # 正規化（0-1スケール）
        if total_reviews > 0:
            normalized_scores = {
                genre: score / total_reviews 
                for genre, score in genre_scores.items()
            }
        else:
            normalized_scores = {}
        
        return normalized_scores
    
    def _analyze_temporal_patterns(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """時間パターンを分析"""
        # シンプルな時間パターン（ランダム生成）
        return {
            'activity_span_days': random.randint(30, 365),
            'avg_reviews_per_month': len(reviews) / 12,
            'seasonal_preference': random.choice(['spring', 'summer', 'autumn', 'winter']),
            'time_preference': random.choice(['morning', 'afternoon', 'evening', 'night'])
        }
    
    def _calculate_engagement_level(self, reviews: List[Dict[str, Any]]) -> float:
        """エンゲージメントレベルを計算"""
        review_lengths = [len(r.get('review_text', '')) for r in reviews]
        avg_length = np.mean(review_lengths) if review_lengths else 0
        
        # 長いレビュー = 高エンゲージメント
        engagement = min(avg_length / 1000.0, 1.0)  # 1000文字で最大エンゲージメント
        return engagement
    
    def _generate_timestamp(self, review: Dict[str, Any]) -> str:
        """タイムスタンプを生成"""
        # レビューの日付があればベース、なければランダム
        write_date = review.get('write_date')
        
        if write_date:
            try:
                # 日付文字列をパース
                base_date = datetime.strptime(str(write_date)[:19], "%Y-%m-%d %H:%M:%S")
            except:
                base_date = datetime.now() - timedelta(days=random.randint(1, 365))
        else:
            base_date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        # ランダムオフセット追加
        offset_hours = random.randint(-48, 48)
        final_date = base_date + timedelta(hours=offset_hours)
        
        return final_date.isoformat()
    
    def _calculate_confidence(self, reviews: List[Dict[str, Any]], actions: List[Dict[str, Any]]) -> float:
        """疑似ユーザーの信頼度を計算"""
        valid_ratings = sum(1 for r in reviews if r.get('rating') is not None)
        total_reviews = len(reviews)
        
        # 評価データの割合ベース
        rating_coverage = valid_ratings / total_reviews if total_reviews > 0 else 0
        
        # アクション信頼度の平均
        action_confidences = [a['confidence'] for a in actions]
        avg_action_confidence = np.mean(action_confidences) if action_confidences else 0
        
        # 総合信頼度
        overall_confidence = (rating_coverage * 0.7) + (avg_action_confidence * 0.3)
        
        return round(overall_confidence, 3)
    
    def save_pseudo_users(self, pseudo_users: List[RatingBasedPseudoUser], 
                         output_file: str = "../processed_data/rating_based_pseudo_users.json") -> bool:
        """評価ベース疑似ユーザーを保存"""
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # データクラスをJSONシリアライズ用に変換
            serializable_users = []
            for user in pseudo_users:
                user_dict = {
                    'user_id': user.user_id,
                    'original_reviewer_id': user.original_reviewer_id,
                    'profile': user.profile,
                    'actions': user.actions,
                    'confidence_score': user.confidence_score,
                    'conversion_method': user.conversion_method,
                    'like_threshold': user.like_threshold,
                    'total_reviews': user.total_reviews,
                    'valid_reviews': user.valid_reviews
                }
                serializable_users.append(user_dict)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_users, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"評価ベース疑似ユーザー保存完了: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存エラー: {e}")
            return False
    
    def print_conversion_stats(self):
        """変換統計を表示"""
        stats = self.conversion_stats
        print("\n=== 評価ベース変換統計 ===")
        print(f"変換方式: {stats['conversion_method']}")
        print(f"閾値設定: {self.like_threshold}以上→Like")
        print(f"総レビュワー数: {stats['total_reviewers']}")
        print(f"総レビュー数: {stats['total_reviews']}")
        print(f"有効評価数: {stats['valid_ratings']}")
        print(f"Like変換: {stats['like_actions']}")
        print(f"Skip変換: {stats['skip_actions']}")
        print(f"推論補完: {stats['no_rating_actions']}")
        
        if stats['total_reviews'] > 0:
            like_rate = stats['like_actions'] / (stats['like_actions'] + stats['skip_actions']) * 100
            rating_coverage = stats['valid_ratings'] / stats['total_reviews'] * 100
            print(f"Like率: {like_rate:.1f}%")
            print(f"評価カバー率: {rating_coverage:.1f}%")
    
    def run(self, batch_dir: str = "../raw_data/batch_reviews") -> List[RatingBasedPseudoUser]:
        """メイン実行関数"""
        self.logger.info("=== 評価ベース疑似ユーザー生成開始 ===")
        self.logger.info(f"変換ルール: {self.like_threshold}以上→Like, 未満→Skip")
        
        # バッチレビューデータ読み込み
        reviews = self.load_batch_reviews(batch_dir)
        if not reviews:
            self.logger.error("レビューデータが見つかりません")
            return []
        
        # 疑似ユーザー生成
        pseudo_users = self.generate_pseudo_users(reviews)
        if not pseudo_users:
            self.logger.error("疑似ユーザーの生成に失敗しました")
            return []
        
        # 結果保存
        self.save_pseudo_users(pseudo_users)
        
        # 統計表示
        self.print_conversion_stats()
        
        self.logger.info("=== 評価ベース疑似ユーザー生成完了 ===")
        return pseudo_users

def main():
    """メイン実行"""
    import logging
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 評価ベース疑似ユーザー生成器
    generator = RatingBasedUserGenerator(like_threshold=4.0)
    pseudo_users = generator.run()
    
    if pseudo_users:
        print(f"\n成功: {len(pseudo_users)} 人の評価ベース疑似ユーザーを生成しました")

if __name__ == "__main__":
    main()