"""
疑似ユーザー生成システム

レビューデータからユーザープロファイル・行動データを生成し、
Two-Tower推薦モデルの初期学習データを作成する
"""

import pandas as pd
import numpy as np
import json
import hashlib
import re
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import random
from dataclasses import dataclass

@dataclass
class PseudoUser:
    """疑似ユーザーデータクラス"""
    user_id: str
    original_reviewer_id: str
    profile: Dict[str, Any]
    reviews: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    confidence_score: float

class PseudoUserGenerator:
    """レビューデータから疑似ユーザーを生成するクラス"""
    
    def __init__(self, salt: str = None):
        self.salt = salt or "pseudo_user_salt_2025"
        self.sentiment_analyzer = None  # 後で初期化
        self.genre_keywords = self._load_genre_keywords()
        self.performer_keywords = self._load_performer_keywords()
    
    def _load_genre_keywords(self) -> Dict[str, List[str]]:
        """ジャンルキーワード辞書をロード"""
        return {
            'amateur': ['素人', 'アマチュア', '一般人', '初撮り'],
            'mature': ['熟女', '人妻', '年上', 'ミセス'],
            'young': ['学生', '制服', 'JK', '若い', '10代'],
            'fetish': ['フェチ', 'M女', 'SM', '調教', '変態'],
            'group': ['乱交', '3P', '4P', '複数', 'ハーレム'],
            'outdoor': ['野外', '露出', '屋外', 'アウトドア'],
            'cosplay': ['コスプレ', 'コス', '制服', 'メイド'],
            'anal': ['アナル', 'お尻', '肛門'],
            'oral': ['フェラ', 'イラマ', 'ディープスロート'],
            'creampie': ['中出し', '生中', '膣内射精']
        }
    
    def _load_performer_keywords(self) -> List[str]:
        """出演者キーワードパターンをロード"""
        return [
            r'[^\s]{2,6}(?:ちゃん|さん|様)(?!は|が|の)',  # 名前+敬称
            r'[女優|出演者|モデル].*?[の|は].*?[^\s]{2,6}',  # 女優の○○
            r'[^\s]{2,6}(?:初出演|デビュー|新人)'  # デビュー情報
        ]
    
    def generate_pseudo_users(self, reviews: List[Dict[str, Any]]) -> List[PseudoUser]:
        """
        レビューデータから疑似ユーザーを生成
        
        Args:
            reviews: クリーニング済みレビューデータのリスト
            
        Returns:
            疑似ユーザーのリスト
        """
        print(f"疑似ユーザー生成開始: {len(reviews)} 件のレビューから")
        
        # Step 1: レビュアーごとにグループ化
        reviewer_groups = self._group_by_reviewer(reviews)
        print(f"レビュアー数: {len(reviewer_groups)}")
        
        # Step 2: 各レビュアーから疑似ユーザー生成
        pseudo_users = []
        for reviewer_id, reviewer_reviews in reviewer_groups.items():
            pseudo_user = self._create_pseudo_user(reviewer_id, reviewer_reviews)
            if pseudo_user:
                pseudo_users.append(pseudo_user)
        
        print(f"疑似ユーザー生成完了: {len(pseudo_users)} ユーザー")
        return pseudo_users
    
    def _group_by_reviewer(self, reviews: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """レビューをレビュアーIDごとにグループ化"""
        reviewer_groups = defaultdict(list)
        
        for review in reviews:
            # レビュアーIDの抽出（複数の方法を試行）
            reviewer_id = self._extract_reviewer_id(review)
            reviewer_groups[reviewer_id].append(review)
        
        # 最低レビュー数でフィルタ（1ユーザーあたり最低1件）
        filtered_groups = {
            reviewer_id: reviews 
            for reviewer_id, reviews in reviewer_groups.items() 
            if len(reviews) >= 1
        }
        
        return filtered_groups
    
    def _extract_reviewer_id(self, review: Dict[str, Any]) -> str:
        """レビューからレビュアーIDを抽出"""
        # 既存のIDがあればそれを使用
        if 'external_reviewer_id' in review:
            return review['external_reviewer_id']
        
        # レビューテキストの特徴からIDを生成
        review_text = review.get('review_text', '')
        element_info = review.get('element_info', {})
        
        # テキストの最初の100文字 + 文体特徴でIDを生成
        text_signature = review_text[:100]
        style_features = self._extract_writing_style(review_text)
        
        signature = f"{text_signature}_{style_features}_{element_info.get('class', [])}"
        return hashlib.md5(signature.encode()).hexdigest()[:12]
    
    def _extract_writing_style(self, text: str) -> str:
        """テキストから文体特徴を抽出"""
        features = []
        
        # 敬語使用
        if any(word in text for word in ['です', 'ます', 'である']):
            features.append('polite')
        
        # 感嘆符の使用
        exclamation_count = text.count('！') + text.count('!')
        if exclamation_count > 2:
            features.append('enthusiastic')
        
        # 長さ特徴
        if len(text) > 500:
            features.append('verbose')
        elif len(text) < 100:
            features.append('concise')
        
        return '_'.join(features)
    
    def _create_pseudo_user(self, reviewer_id: str, reviews: List[Dict[str, Any]]) -> PseudoUser:
        """個別の疑似ユーザーを作成"""
        
        # 疑似ユーザーID生成
        pseudo_user_id = self._generate_pseudo_user_id(reviewer_id)
        
        # ユーザープロファイル生成
        profile = self._generate_user_profile(reviews)
        
        # ユーザー行動データ生成
        actions = self._generate_user_actions(pseudo_user_id, reviews)
        
        # 信頼度スコア計算
        confidence_score = self._calculate_confidence_score(reviews, profile)
        
        return PseudoUser(
            user_id=pseudo_user_id,
            original_reviewer_id=reviewer_id,
            profile=profile,
            reviews=reviews,
            actions=actions,
            confidence_score=confidence_score
        )
    
    def _generate_pseudo_user_id(self, reviewer_id: str) -> str:
        """匿名化された疑似ユーザーIDを生成"""
        return hashlib.sha256(f"pseudo_{reviewer_id}_{self.salt}".encode()).hexdigest()
    
    def _generate_user_profile(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """レビュー履歴からユーザープロファイルを生成"""
        
        # 基本統計
        ratings = [r.get('rating', 3.0) for r in reviews if r.get('rating')]
        text_lengths = [len(r.get('review_text', '')) for r in reviews]
        
        # ジャンル嗜好分析
        genre_preferences = self._analyze_genre_preferences(reviews)
        
        # 感情パターン分析
        sentiment_profile = self._analyze_sentiment_profile(reviews)
        
        # 時間的活動パターン
        temporal_pattern = self._analyze_temporal_pattern(reviews)
        
        profile = {
            # 基本情報
            'review_count': len(reviews),
            'avg_rating': np.mean(ratings) if ratings else 3.0,
            'rating_std': np.std(ratings) if len(ratings) > 1 else 0.0,
            'avg_review_length': np.mean(text_lengths),
            'review_length_std': np.std(text_lengths),
            
            # 嗜好情報
            'genre_preferences': genre_preferences,
            'sentiment_profile': sentiment_profile,
            'temporal_pattern': temporal_pattern,
            
            # 品質指標
            'engagement_level': self._calculate_engagement_level(reviews),
            'expertise_level': self._calculate_expertise_level(reviews)
        }
        
        return profile
    
    def _analyze_genre_preferences(self, reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """レビューからジャンル嗜好を分析"""
        genre_scores = defaultdict(float)
        total_reviews = len(reviews)
        
        for review in reviews:
            text = review.get('review_text', '').lower()
            rating = review.get('rating', 3.0)
            
            # 各ジャンルキーワードの出現をチェック
            for genre, keywords in self.genre_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        # 評価値を重みとして使用
                        genre_scores[genre] += rating / total_reviews
        
        # 正規化（0-1スケール）
        max_score = max(genre_scores.values()) if genre_scores else 1.0
        normalized_scores = {
            genre: score / max_score 
            for genre, score in genre_scores.items()
        }
        
        return dict(normalized_scores)
    
    def _analyze_sentiment_profile(self, reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """感情パターンを分析"""
        sentiments = []
        
        for review in reviews:
            text = review.get('review_text', '')
            sentiment = self._simple_sentiment_analysis(text)
            sentiments.append(sentiment)
        
        return {
            'avg_sentiment': np.mean(sentiments),
            'sentiment_variance': np.var(sentiments),
            'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments)
        }
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """簡易感情分析（-1 to 1）"""
        positive_words = ['良い', 'すごい', '最高', '素晴らしい', '面白い', '楽しい', '満足', 'おすすめ']
        negative_words = ['悪い', 'つまらない', '最悪', 'ダメ', '嫌い', 'がっかり', '不満', '微妙']
        
        positive_count = sum(word in text for word in positive_words)
        negative_count = sum(word in text for word in negative_words)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
    def _analyze_temporal_pattern(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """時間的活動パターンを分析"""
        # 実際の実装では投稿日時を使用
        # ここではダミーデータで代替
        return {
            'activity_span_days': random.randint(30, 365),
            'avg_reviews_per_month': len(reviews) / 12,
            'seasonal_preference': random.choice(['spring', 'summer', 'autumn', 'winter']),
            'time_preference': random.choice(['morning', 'afternoon', 'evening', 'night'])
        }
    
    def _calculate_engagement_level(self, reviews: List[Dict[str, Any]]) -> float:
        """エンゲージメントレベルを計算（0-1）"""
        factors = []
        
        # レビュー数
        review_count_score = min(len(reviews) / 20, 1.0)  # 20件で満点
        factors.append(review_count_score)
        
        # レビューの詳細度
        avg_length = np.mean([len(r.get('review_text', '')) for r in reviews])
        length_score = min(avg_length / 500, 1.0)  # 500文字で満点
        factors.append(length_score)
        
        # 評価の分散（多様な評価をする = エンゲージメント高）
        ratings = [r.get('rating', 3.0) for r in reviews if r.get('rating')]
        if len(ratings) > 1:
            variance_score = min(np.var(ratings) / 2, 1.0)  # 分散2で満点
            factors.append(variance_score)
        
        return np.mean(factors)
    
    def _calculate_expertise_level(self, reviews: List[Dict[str, Any]]) -> float:
        """専門性レベルを計算（0-1）"""
        # 専門用語の使用頻度
        technical_terms = ['画質', '音質', 'カメラワーク', '演出', '構成', 'ストーリー', 'シナリオ']
        
        technical_mentions = 0
        total_words = 0
        
        for review in reviews:
            text = review.get('review_text', '')
            total_words += len(text)
            for term in technical_terms:
                technical_mentions += text.count(term)
        
        if total_words == 0:
            return 0.0
        
        expertise_score = min(technical_mentions / (total_words / 100), 1.0)
        return expertise_score
    
    def _generate_user_actions(self, user_id: str, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """レビューからユーザー行動データを生成"""
        actions = []
        
        for i, review in enumerate(reviews):
            # メインアクション（レビューした動画）
            main_action = self._convert_review_to_action(user_id, review, confidence=0.9)
            actions.append(main_action)
            
            # 関連動画への推定アクション（2-4個）
            related_actions = self._generate_related_actions(user_id, review, count=random.randint(2, 4))
            actions.extend(related_actions)
        
        return actions
    
    def _convert_review_to_action(self, user_id: str, review: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """レビュー1件をユーザーアクションに変換"""
        
        rating = review.get('rating', 3.0)
        sentiment = self._simple_sentiment_analysis(review.get('review_text', ''))
        
        # いいね確率の計算
        like_probability = self._calculate_like_probability(rating, sentiment)
        action = 'like' if random.random() < like_probability else 'skip'
        
        return {
            'user_id': user_id,
            'video_id': self._generate_video_id_from_review(review),
            'action': action,
            'confidence': confidence,
            'timestamp': self._generate_realistic_timestamp(),
            'session_id': f"session_{random.randint(1000, 9999)}",
            'source': 'review_conversion',
            'metadata': {
                'original_rating': rating,
                'sentiment_score': sentiment,
                'like_probability': like_probability,
                'review_length': len(review.get('review_text', ''))
            }
        }
    
    def _calculate_like_probability(self, rating: float, sentiment: float) -> float:
        """評価値と感情スコアからいいね確率を計算"""
        # 評価値ベースの確率
        rating_prob = {
            5.0: 0.95, 4.5: 0.85, 4.0: 0.75,
            3.5: 0.65, 3.0: 0.50, 2.5: 0.35,
            2.0: 0.25, 1.5: 0.15, 1.0: 0.05
        }
        
        # 最も近い評価値を選択
        base_prob = rating_prob.get(rating, 0.5)
        
        # 感情スコアで調整
        sentiment_adjustment = sentiment * 0.2  # -0.2 to +0.2の調整
        
        final_prob = np.clip(base_prob + sentiment_adjustment, 0.05, 0.95)
        return final_prob
    
    def _generate_related_actions(self, user_id: str, base_review: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """関連動画への推定アクションを生成"""
        actions = []
        
        # ベースレビューから嗜好を推定
        preferences = self._infer_preferences_from_review(base_review)
        
        for _ in range(count):
            # 嗜好に基づいてアクション確率を決定
            preference_match_score = random.uniform(0.3, 0.8)  # 関連度
            action_probability = preference_match_score * 0.7  # 0.7倍して控えめに
            
            action = 'like' if random.random() < action_probability else 'skip'
            
            actions.append({
                'user_id': user_id,
                'video_id': f"video_{random.randint(10000, 99999)}",  # ダミーID
                'action': action,
                'confidence': 0.6,  # 推定データなので低信頼度
                'timestamp': self._generate_realistic_timestamp(),
                'session_id': f"session_{random.randint(1000, 9999)}",
                'source': 'preference_inference',
                'metadata': {
                    'preference_match_score': preference_match_score,
                    'based_on_review': True,
                    'inference_confidence': 0.6
                }
            })
        
        return actions
    
    def _generate_video_id_from_review(self, review: Dict[str, Any]) -> str:
        """レビューからビデオIDを生成（ダミー）"""
        review_hash = hashlib.md5(
            review.get('review_text', '')[:50].encode()
        ).hexdigest()[:8]
        return f"video_{review_hash}"
    
    def _generate_realistic_timestamp(self) -> str:
        """リアルなタイムスタンプを生成"""
        # 過去1年間のランダムな日時
        start_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        
        timestamp = start_date + timedelta(
            days=random_days, 
            hours=random_hours, 
            minutes=random_minutes
        )
        
        return timestamp.isoformat()
    
    def _infer_preferences_from_review(self, review: Dict[str, Any]) -> Dict[str, float]:
        """レビューから嗜好を推定"""
        text = review.get('review_text', '').lower()
        preferences = {}
        
        # ジャンル嗜好の推定
        for genre, keywords in self.genre_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            preferences[f'genre_{genre}'] = min(score / len(keywords), 1.0)
        
        return preferences
    
    def _calculate_confidence_score(self, reviews: List[Dict[str, Any]], profile: Dict[str, Any]) -> float:
        """疑似ユーザーの信頼度スコアを計算"""
        factors = []
        
        # レビュー数（多いほど信頼度高）
        review_count_factor = min(len(reviews) / 10, 1.0)
        factors.append(review_count_factor)
        
        # レビューの詳細度
        avg_length = np.mean([len(r.get('review_text', '')) for r in reviews])
        detail_factor = min(avg_length / 300, 1.0)
        factors.append(detail_factor)
        
        # 評価値の一貫性
        ratings = [r.get('rating', 3.0) for r in reviews if r.get('rating')]
        if len(ratings) > 1:
            consistency_factor = 1.0 - min(np.std(ratings) / 2, 0.5)
            factors.append(consistency_factor)
        
        # エンゲージメント・専門性
        factors.append(profile.get('engagement_level', 0.5))
        factors.append(profile.get('expertise_level', 0.5))
        
        return np.mean(factors)

    def save_pseudo_users(self, pseudo_users: List[PseudoUser], output_file: str):
        """疑似ユーザーデータを保存"""
        data = []
        for user in pseudo_users:
            data.append({
                'user_id': user.user_id,
                'original_reviewer_id': user.original_reviewer_id,
                'profile': user.profile,
                'actions': user.actions,
                'confidence_score': user.confidence_score,
                'review_count': len(user.reviews)
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"疑似ユーザーデータ保存完了: {output_file}")

def main():
    """メイン処理"""
    generator = PseudoUserGenerator()
    
    # クリーニング済みレビューデータを読み込み
    input_file = "../processed_data/cleaned_reviews.json"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        
        print(f"レビューデータ読み込み: {len(reviews)} 件")
        
        # 疑似ユーザー生成
        pseudo_users = generator.generate_pseudo_users(reviews)
        
        # 結果保存
        output_file = "../processed_data/pseudo_users.json"
        generator.save_pseudo_users(pseudo_users, output_file)
        
        # 統計情報出力
        print("\n=== 疑似ユーザー生成統計 ===")
        print(f"生成ユーザー数: {len(pseudo_users)}")
        
        total_actions = sum(len(user.actions) for user in pseudo_users)
        print(f"総行動データ数: {total_actions}")
        
        avg_confidence = np.mean([user.confidence_score for user in pseudo_users])
        print(f"平均信頼度: {avg_confidence:.3f}")
        
        like_ratio = sum(
            sum(1 for action in user.actions if action['action'] == 'like')
            for user in pseudo_users
        ) / total_actions
        print(f"いいね率: {like_ratio:.3f}")
        
    except FileNotFoundError:
        print(f"入力ファイルが見つかりません: {input_file}")
        print("先に 'make data-clean' を実行してください")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()