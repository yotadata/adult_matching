"""
リアルユーザー特徴抽出器

user_video_decisionsテーブルからユーザー行動パターンを抽出し、
疑似ユーザーデータとの統合を可能にする特徴処理システム
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UserBehaviorProfile:
    """ユーザー行動プロファイル"""
    user_id: str
    total_decisions: int
    like_count: int
    nope_count: int
    like_ratio: float
    decision_frequency: float  # decisions per day
    genre_preferences: Dict[str, float]
    maker_preferences: Dict[str, float]
    price_preferences: Dict[str, float]  # price range preferences
    activity_recency: float  # days since last activity
    engagement_level: str  # high, medium, low
    diversity_score: float  # how diverse their likes are

@dataclass
class UserFeatureVector:
    """ユーザー特徴ベクター（768次元対応）"""
    user_id: str
    behavioral_features: np.ndarray  # 基本行動特徴 (10次元)
    genre_features: np.ndarray       # ジャンル嗜好 (50次元)
    maker_features: np.ndarray       # メーカー嗜好 (50次元)
    temporal_features: np.ndarray    # 時系列特徴 (8次元)
    diversity_features: np.ndarray   # 多様性特徴 (10次元)
    padding_features: np.ndarray     # 768次元への調整 (640次元)

class RealUserFeatureExtractor:
    """リアルユーザー特徴抽出器"""
    
    def __init__(self, 
                 db_connection_string: str,
                 target_feature_dim: int = 768,
                 cold_start_threshold: int = 3,
                 activity_window_days: int = 180):
        """
        初期化
        
        Args:
            db_connection_string: PostgreSQL接続文字列
            target_feature_dim: 目標特徴次元数（768次元）
            cold_start_threshold: コールドスタート閾値（最小決定数）
            activity_window_days: 活動期間ウィンドウ（日数）
        """
        self.db_connection_string = db_connection_string
        self.target_feature_dim = target_feature_dim
        self.cold_start_threshold = cold_start_threshold
        self.activity_window_days = activity_window_days
        
        # 特徴量の次元構成
        self.behavioral_dim = 10
        self.genre_dim = 50
        self.maker_dim = 50
        self.temporal_dim = 8
        self.diversity_dim = 10
        self.padding_dim = target_feature_dim - (
            self.behavioral_dim + self.genre_dim + 
            self.maker_dim + self.temporal_dim + self.diversity_dim
        )
        
        # エンコーダー
        self.genre_to_idx = {}
        self.maker_to_idx = {}
        
        # 統計情報
        self.extraction_stats = {
            'total_users_processed': 0,
            'cold_start_users': 0,
            'active_users': 0,
            'feature_extraction_time': 0,
            'avg_decisions_per_user': 0
        }

    def connect_db(self) -> psycopg2.extensions.connection:
        """データベース接続"""
        try:
            conn = psycopg2.connect(
                self.db_connection_string,
                cursor_factory=RealDictCursor
            )
            logger.info("PostgreSQL接続成功")
            return conn
        except Exception as e:
            logger.error(f"データベース接続エラー: {e}")
            raise

    def build_vocabulary(self, conn: psycopg2.extensions.connection) -> Tuple[Dict[str, int], Dict[str, int]]:
        """ジャンル・メーカーの語彙を構築"""
        logger.info("語彙構築開始...")
        
        cursor = conn.cursor()
        
        # ジャンル語彙の構築
        genre_query = """
        SELECT DISTINCT v.genre, COUNT(*) as count
        FROM videos v
        JOIN user_video_decisions uvd ON v.id = uvd.video_id
        WHERE v.genre IS NOT NULL
        GROUP BY v.genre
        ORDER BY count DESC
        LIMIT %s
        """
        
        cursor.execute(genre_query, (self.genre_dim,))
        genres = cursor.fetchall()
        self.genre_to_idx = {genre['genre']: idx for idx, genre in enumerate(genres)}
        
        # メーカー語彙の構築
        maker_query = """
        SELECT DISTINCT v.maker, COUNT(*) as count
        FROM videos v
        JOIN user_video_decisions uvd ON v.id = uvd.video_id
        WHERE v.maker IS NOT NULL
        GROUP BY v.maker
        ORDER BY count DESC
        LIMIT %s
        """
        
        cursor.execute(maker_query, (self.maker_dim,))
        makers = cursor.fetchall()
        self.maker_to_idx = {maker['maker']: idx for idx, maker in enumerate(makers)}
        
        logger.info(f"語彙構築完了 - ジャンル: {len(self.genre_to_idx)}, メーカー: {len(self.maker_to_idx)}")
        return self.genre_to_idx, self.maker_to_idx

    def extract_user_behavior_profile(self, 
                                    user_id: str, 
                                    conn: psycopg2.extensions.connection) -> UserBehaviorProfile:
        """単一ユーザーの行動プロファイル抽出"""
        
        cursor = conn.cursor()
        
        # 活動期間の設定
        activity_cutoff = datetime.now() - timedelta(days=self.activity_window_days)
        
        # 基本行動統計
        behavior_query = """
        SELECT 
            uvd.decision_type,
            COUNT(*) as count,
            MIN(uvd.created_at) as first_decision,
            MAX(uvd.created_at) as last_decision,
            v.genre,
            v.maker,
            v.price
        FROM user_video_decisions uvd
        JOIN videos v ON uvd.video_id = v.id
        WHERE uvd.user_id = %s AND uvd.created_at >= %s
        GROUP BY uvd.decision_type, v.genre, v.maker, v.price
        ORDER BY count DESC
        """
        
        cursor.execute(behavior_query, (user_id, activity_cutoff))
        decisions = cursor.fetchall()
        
        if not decisions:
            # コールドスタートユーザー
            return self._create_cold_start_profile(user_id)
        
        # 基本統計の計算
        like_count = sum(d['count'] for d in decisions if d['decision_type'] == 'like')
        nope_count = sum(d['count'] for d in decisions if d['decision_type'] == 'nope')
        total_decisions = like_count + nope_count
        like_ratio = like_count / total_decisions if total_decisions > 0 else 0.0
        
        # 時系列特徴
        first_decision = min(d['first_decision'] for d in decisions if d['first_decision'])
        last_decision = max(d['last_decision'] for d in decisions if d['last_decision'])
        activity_span = (last_decision - first_decision).days if first_decision and last_decision else 1
        decision_frequency = total_decisions / max(activity_span, 1)
        activity_recency = (datetime.now() - last_decision).days if last_decision else 999
        
        # ジャンル嗜好の計算（likeのみ）
        like_decisions = [d for d in decisions if d['decision_type'] == 'like']
        genre_preferences = self._calculate_genre_preferences(like_decisions)
        maker_preferences = self._calculate_maker_preferences(like_decisions)
        price_preferences = self._calculate_price_preferences(like_decisions)
        
        # 多様性スコア（Shannon entropy based）
        diversity_score = self._calculate_diversity_score(like_decisions)
        
        # エンゲージメントレベル
        engagement_level = self._determine_engagement_level(
            total_decisions, decision_frequency, like_ratio, diversity_score
        )
        
        return UserBehaviorProfile(
            user_id=user_id,
            total_decisions=total_decisions,
            like_count=like_count,
            nope_count=nope_count,
            like_ratio=like_ratio,
            decision_frequency=decision_frequency,
            genre_preferences=genre_preferences,
            maker_preferences=maker_preferences,
            price_preferences=price_preferences,
            activity_recency=activity_recency,
            engagement_level=engagement_level,
            diversity_score=diversity_score
        )

    def _create_cold_start_profile(self, user_id: str) -> UserBehaviorProfile:
        """コールドスタートユーザーのデフォルトプロファイル"""
        return UserBehaviorProfile(
            user_id=user_id,
            total_decisions=0,
            like_count=0,
            nope_count=0,
            like_ratio=0.5,  # ニュートラル
            decision_frequency=0.0,
            genre_preferences={},
            maker_preferences={},
            price_preferences={'low': 0.4, 'medium': 0.4, 'high': 0.2},
            activity_recency=999,
            engagement_level='cold_start',
            diversity_score=0.5
        )

    def _calculate_genre_preferences(self, like_decisions: List[Dict]) -> Dict[str, float]:
        """ジャンル嗜好の計算"""
        genre_counts = {}
        total_likes = len(like_decisions)
        
        for decision in like_decisions:
            genre = decision.get('genre')
            if genre and genre in self.genre_to_idx:
                genre_counts[genre] = genre_counts.get(genre, 0) + decision['count']
        
        # 正規化
        return {genre: count / total_likes 
                for genre, count in genre_counts.items()} if total_likes > 0 else {}

    def _calculate_maker_preferences(self, like_decisions: List[Dict]) -> Dict[str, float]:
        """メーカー嗜好の計算"""
        maker_counts = {}
        total_likes = len(like_decisions)
        
        for decision in like_decisions:
            maker = decision.get('maker')
            if maker and maker in self.maker_to_idx:
                maker_counts[maker] = maker_counts.get(maker, 0) + decision['count']
        
        # 正規化
        return {maker: count / total_likes 
                for maker, count in maker_counts.items()} if total_likes > 0 else {}

    def _calculate_price_preferences(self, like_decisions: List[Dict]) -> Dict[str, float]:
        """価格嗜好の計算"""
        prices = [d.get('price', 0) or 0 for d in like_decisions for _ in range(d['count'])]
        
        if not prices:
            return {'low': 0.33, 'medium': 0.33, 'high': 0.34}
        
        # 価格帯の分類（日本円基準）
        low_count = sum(1 for p in prices if p <= 500)
        medium_count = sum(1 for p in prices if 500 < p <= 1500)
        high_count = sum(1 for p in prices if p > 1500)
        total = len(prices)
        
        return {
            'low': low_count / total,
            'medium': medium_count / total,
            'high': high_count / total
        }

    def _calculate_diversity_score(self, like_decisions: List[Dict]) -> float:
        """多様性スコア（Shannon entropy）の計算"""
        if not like_decisions:
            return 0.5
        
        # ジャンル分布からエントロピーを計算
        genre_counts = {}
        total = sum(d['count'] for d in like_decisions)
        
        for decision in like_decisions:
            genre = decision.get('genre', 'unknown')
            genre_counts[genre] = genre_counts.get(genre, 0) + decision['count']
        
        # Shannon entropy
        entropy = 0.0
        for count in genre_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # 正規化（最大エントロピーで割る）
        max_entropy = np.log2(len(genre_counts)) if len(genre_counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.5

    def _determine_engagement_level(self, 
                                  total_decisions: int, 
                                  decision_frequency: float,
                                  like_ratio: float,
                                  diversity_score: float) -> str:
        """エンゲージメントレベルの決定"""
        
        # 複合スコアの計算
        activity_score = min(total_decisions / 100, 1.0)  # 100決定で満点
        frequency_score = min(decision_frequency / 5.0, 1.0)  # 5決定/日で満点
        quality_score = abs(like_ratio - 0.5) * 2  # 0.5から離れるほど明確な嗜好
        diversity_norm = diversity_score  # 既に0-1範囲
        
        engagement_score = (
            activity_score * 0.4 + 
            frequency_score * 0.3 + 
            quality_score * 0.2 + 
            diversity_norm * 0.1
        )
        
        if engagement_score >= 0.7:
            return 'high'
        elif engagement_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def convert_profile_to_feature_vector(self, profile: UserBehaviorProfile) -> UserFeatureVector:
        """行動プロファイルを768次元特徴ベクターに変換"""
        
        # 1. 基本行動特徴 (10次元)
        behavioral_features = np.array([
            profile.like_ratio,
            profile.decision_frequency / 10.0,  # 正規化
            min(profile.total_decisions / 1000, 1.0),  # 正規化
            1.0 / (profile.activity_recency + 1),  # 最近性
            profile.diversity_score,
            1.0 if profile.engagement_level == 'high' else 0.0,
            1.0 if profile.engagement_level == 'medium' else 0.0,
            1.0 if profile.engagement_level == 'low' else 0.0,
            1.0 if profile.engagement_level == 'cold_start' else 0.0,
            profile.like_count / max(profile.total_decisions, 1)  # Like density
        ])
        
        # 2. ジャンル特徴 (50次元)
        genre_features = np.zeros(self.genre_dim)
        for genre, score in profile.genre_preferences.items():
            if genre in self.genre_to_idx:
                genre_features[self.genre_to_idx[genre]] = score
        
        # 3. メーカー特徴 (50次元)
        maker_features = np.zeros(self.maker_dim)
        for maker, score in profile.maker_preferences.items():
            if maker in self.maker_to_idx:
                maker_features[self.maker_to_idx[maker]] = score
        
        # 4. 時系列特徴 (8次元)
        temporal_features = np.array([
            np.sin(2 * np.pi * datetime.now().hour / 24),  # 時間サイクル
            np.cos(2 * np.pi * datetime.now().hour / 24),
            np.sin(2 * np.pi * datetime.now().weekday() / 7),  # 曜日サイクル
            np.cos(2 * np.pi * datetime.now().weekday() / 7),
            profile.price_preferences.get('low', 0.33),
            profile.price_preferences.get('medium', 0.33),
            profile.price_preferences.get('high', 0.34),
            min(profile.decision_frequency, 10.0) / 10.0  # 活動頻度
        ])
        
        # 5. 多様性特徴 (10次元)
        diversity_features = np.array([
            profile.diversity_score,
            len(profile.genre_preferences) / 20.0,  # ジャンル多様性
            len(profile.maker_preferences) / 10.0,  # メーカー多様性
            np.std(list(profile.genre_preferences.values())) if profile.genre_preferences else 0.0,
            np.std(list(profile.maker_preferences.values())) if profile.maker_preferences else 0.0,
            min(profile.total_decisions / 500, 1.0),  # 経験値
            profile.like_ratio * (1 - profile.like_ratio) * 4,  # 嗜好明確度
            1.0 if profile.total_decisions >= self.cold_start_threshold else 0.0,
            profile.activity_recency / 365.0 if profile.activity_recency < 365 else 1.0,
            np.mean(list(profile.price_preferences.values()))
        ])
        
        # 6. パディング特徴（768次元にするため）
        padding_features = np.zeros(self.padding_dim)
        
        return UserFeatureVector(
            user_id=profile.user_id,
            behavioral_features=behavioral_features,
            genre_features=genre_features,
            maker_features=maker_features,
            temporal_features=temporal_features,
            diversity_features=diversity_features,
            padding_features=padding_features
        )

    def extract_batch_features(self, 
                             user_ids: List[str],
                             conn: psycopg2.extensions.connection) -> Tuple[List[UserFeatureVector], Dict]:
        """バッチでのユーザー特徴抽出"""
        logger.info(f"バッチ特徴抽出開始: {len(user_ids)} ユーザー")
        
        start_time = datetime.now()
        feature_vectors = []
        
        for user_id in user_ids:
            try:
                profile = self.extract_user_behavior_profile(user_id, conn)
                feature_vector = self.convert_profile_to_feature_vector(profile)
                feature_vectors.append(feature_vector)
                
                # 統計更新
                if profile.total_decisions < self.cold_start_threshold:
                    self.extraction_stats['cold_start_users'] += 1
                else:
                    self.extraction_stats['active_users'] += 1
                    
            except Exception as e:
                logger.error(f"ユーザー {user_id} の特徴抽出エラー: {e}")
                # デフォルト特徴ベクターを生成
                cold_profile = self._create_cold_start_profile(user_id)
                feature_vector = self.convert_profile_to_feature_vector(cold_profile)
                feature_vectors.append(feature_vector)
                self.extraction_stats['cold_start_users'] += 1
        
        # 統計情報の更新
        extraction_time = (datetime.now() - start_time).total_seconds()
        self.extraction_stats.update({
            'total_users_processed': len(user_ids),
            'feature_extraction_time': extraction_time,
            'avg_decisions_per_user': sum(
                profile.total_decisions for profile in [
                    self.extract_user_behavior_profile(uid, conn) for uid in user_ids[:10]
                ]
            ) / min(len(user_ids), 10)
        })
        
        logger.info(f"バッチ特徴抽出完了: {len(feature_vectors)} 特徴ベクター生成")
        return feature_vectors, self.extraction_stats

    def convert_to_numpy_array(self, feature_vectors: List[UserFeatureVector]) -> Tuple[np.ndarray, List[str]]:
        """特徴ベクターをNumPy配列に変換"""
        
        user_ids = [fv.user_id for fv in feature_vectors]
        feature_matrix = []
        
        for fv in feature_vectors:
            # 全特徴を結合して768次元ベクターを作成
            combined_features = np.concatenate([
                fv.behavioral_features,
                fv.genre_features,
                fv.maker_features,
                fv.temporal_features,
                fv.diversity_features,
                fv.padding_features
            ])
            
            # 768次元を確保
            if len(combined_features) != self.target_feature_dim:
                # サイズ調整
                if len(combined_features) > self.target_feature_dim:
                    combined_features = combined_features[:self.target_feature_dim]
                else:
                    padding_needed = self.target_feature_dim - len(combined_features)
                    combined_features = np.concatenate([
                        combined_features,
                        np.zeros(padding_needed)
                    ])
            
            feature_matrix.append(combined_features)
        
        return np.array(feature_matrix), user_ids

    def integrate_with_pseudo_users(self, 
                                   real_feature_vectors: List[UserFeatureVector],
                                   pseudo_user_file: str) -> Tuple[np.ndarray, List[str]]:
        """疑似ユーザーデータとの統合"""
        logger.info("疑似ユーザーデータとの統合開始...")
        
        # 疑似ユーザーデータの読み込み
        try:
            with open(pseudo_user_file, 'r', encoding='utf-8') as f:
                pseudo_users = json.load(f)
            logger.info(f"疑似ユーザーデータ読み込み: {len(pseudo_users)} ユーザー")
        except Exception as e:
            logger.error(f"疑似ユーザーデータ読み込みエラー: {e}")
            pseudo_users = []
        
        # リアルユーザー特徴の変換
        real_features, real_user_ids = self.convert_to_numpy_array(real_feature_vectors)
        
        # 疑似ユーザー特徴の変換（既存フォーマットからの変換）
        pseudo_features = []
        pseudo_user_ids = []
        
        for pseudo_user in pseudo_users:
            # 疑似ユーザーを同様の形式に変換
            pseudo_profile = self._convert_pseudo_to_profile(pseudo_user)
            pseudo_feature_vector = self.convert_profile_to_feature_vector(pseudo_profile)
            pseudo_features.append(pseudo_feature_vector)
            pseudo_user_ids.append(pseudo_user['user_id'])
        
        if pseudo_features:
            pseudo_feature_array, _ = self.convert_to_numpy_array(pseudo_features)
            
            # 統合
            integrated_features = np.vstack([real_features, pseudo_feature_array])
            integrated_user_ids = real_user_ids + pseudo_user_ids
        else:
            integrated_features = real_features
            integrated_user_ids = real_user_ids
        
        logger.info(f"統合完了: {len(integrated_user_ids)} ユーザー, {integrated_features.shape[1]} 次元")
        return integrated_features, integrated_user_ids

    def _convert_pseudo_to_profile(self, pseudo_user: Dict) -> UserBehaviorProfile:
        """疑似ユーザーデータをUserBehaviorProfileに変換"""
        profile_data = pseudo_user.get('profile', {})
        actions = pseudo_user.get('actions', [])
        
        like_count = sum(1 for a in actions if a.get('action') == 'like')
        total_actions = len(actions)
        
        return UserBehaviorProfile(
            user_id=pseudo_user['user_id'],
            total_decisions=total_actions,
            like_count=like_count,
            nope_count=total_actions - like_count,
            like_ratio=like_count / total_actions if total_actions > 0 else 0.5,
            decision_frequency=total_actions / 30.0,  # 仮定：30日での活動
            genre_preferences=profile_data.get('genre_preferences', {}),
            maker_preferences={},
            price_preferences={'low': 0.4, 'medium': 0.4, 'high': 0.2},
            activity_recency=1.0,  # 仮定：最近の活動
            engagement_level='medium',  # デフォルト
            diversity_score=profile_data.get('diversity_score', 0.5)
        )

    def save_extraction_results(self, 
                              feature_vectors: List[UserFeatureVector],
                              output_path: str = "data/processed/real_user_features.json"):
        """抽出結果の保存"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # JSON形式での保存
        results = {
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'feature_dim': self.target_feature_dim,
                'total_users': len(feature_vectors),
                'extraction_stats': self.extraction_stats
            },
            'feature_vectors': []
        }
        
        for fv in feature_vectors:
            feature_data = {
                'user_id': fv.user_id,
                'behavioral_features': fv.behavioral_features.tolist(),
                'genre_features': fv.genre_features.tolist(),
                'maker_features': fv.maker_features.tolist(),
                'temporal_features': fv.temporal_features.tolist(),
                'diversity_features': fv.diversity_features.tolist()
            }
            results['feature_vectors'].append(feature_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"抽出結果保存完了: {output_path}")

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='リアルユーザー特徴抽出')
    parser.add_argument('--db-url', required=True, help='PostgreSQL接続URL')
    parser.add_argument('--user-ids-file', help='対象ユーザーIDリストファイル')
    parser.add_argument('--pseudo-users-file', 
                       default='data/processed/rating_based_pseudo_users.json',
                       help='疑似ユーザーファイル')
    parser.add_argument('--output-path', 
                       default='data/processed/real_user_features.json',
                       help='出力パス')
    parser.add_argument('--sample-size', type=int, default=1000, 
                       help='サンプルユーザー数（指定なしの場合）')
    
    args = parser.parse_args()
    
    # 特徴抽出器の初期化
    extractor = RealUserFeatureExtractor(args.db_url)
    
    try:
        # データベース接続
        conn = extractor.connect_db()
        
        # 語彙構築
        extractor.build_vocabulary(conn)
        
        # ユーザーIDリストの取得
        if args.user_ids_file:
            with open(args.user_ids_file, 'r') as f:
                user_ids = [line.strip() for line in f if line.strip()]
        else:
            # アクティブユーザーをサンプリング
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT user_id 
                FROM user_video_decisions 
                WHERE created_at >= %s
                ORDER BY user_id
                LIMIT %s
            """, (datetime.now() - timedelta(days=180), args.sample_size))
            
            user_ids = [row['user_id'] for row in cursor.fetchall()]
        
        logger.info(f"対象ユーザー数: {len(user_ids)}")
        
        # バッチ特徴抽出
        feature_vectors, stats = extractor.extract_batch_features(user_ids, conn)
        
        # 疑似ユーザーとの統合
        integrated_features, integrated_user_ids = extractor.integrate_with_pseudo_users(
            feature_vectors, args.pseudo_users_file
        )
        
        # 結果保存
        extractor.save_extraction_results(feature_vectors, args.output_path)
        
        # 統計情報の表示
        logger.info("=== 抽出統計 ===")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        logger.info(f"統合後特徴行列形状: {integrated_features.shape}")
        logger.info("リアルユーザー特徴抽出完了")
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    main()