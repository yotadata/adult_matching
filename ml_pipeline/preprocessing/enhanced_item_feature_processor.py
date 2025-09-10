"""
強化されたアイテム特徴処理器

現在の32,304ビデオデータから高品質な特徴量を抽出し、
768次元Two-Towerモデルとの最適化された統合を実現
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
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """ビデオメタデータ構造"""
    video_id: str
    title: str
    description: str
    maker: str
    genre: str
    price: float
    duration_seconds: int
    performers: List[str]
    tags: List[str]
    external_id: str
    source: str
    created_at: datetime

@dataclass
class VideoFeatureVector:
    """ビデオ特徴ベクター（768次元対応）"""
    video_id: str
    text_features: np.ndarray        # TF-IDF特徴 (500次元)
    categorical_features: np.ndarray # カテゴリ特徴 (100次元)
    numerical_features: np.ndarray   # 数値特徴 (20次元)
    semantic_features: np.ndarray    # 意味的特徴 (100次元)
    popularity_features: np.ndarray  # 人気度特徴 (20次元)
    padding_features: np.ndarray     # 768次元への調整 (28次元)

class EnhancedItemFeatureProcessor:
    """強化されたアイテム特徴処理器"""
    
    def __init__(self, 
                 db_connection_string: str,
                 target_feature_dim: int = 768,
                 tfidf_max_features: int = 500,
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        初期化
        
        Args:
            db_connection_string: PostgreSQL接続文字列
            target_feature_dim: 目標特徴次元数（768次元）
            tfidf_max_features: TF-IDF最大特徴数
            min_df: TF-IDF最小文書頻度
            max_df: TF-IDF最大文書頻度
        """
        self.db_connection_string = db_connection_string
        self.target_feature_dim = target_feature_dim
        self.tfidf_max_features = tfidf_max_features
        
        # 特徴量の次元構成
        self.text_dim = 500        # TF-IDF特徴
        self.categorical_dim = 100  # カテゴリ特徴
        self.numerical_dim = 20     # 数値特徴
        self.semantic_dim = 100     # 意味的特徴
        self.popularity_dim = 20    # 人気度特徴
        self.padding_dim = target_feature_dim - (
            self.text_dim + self.categorical_dim + 
            self.numerical_dim + self.semantic_dim + self.popularity_dim
        )
        
        # 前処理器
        self.text_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            stop_words=None,  # 日本語対応のため無効化
            token_pattern=r'\b\w+\b',
            lowercase=True
        )
        
        self.genre_encoder = LabelEncoder()
        self.maker_encoder = LabelEncoder()
        self.source_encoder = LabelEncoder()
        self.numerical_scaler = StandardScaler()
        
        # 日本語特有の処理
        self.japanese_patterns = {
            'hiragana': re.compile(r'[\u3040-\u309F]'),
            'katakana': re.compile(r'[\u30A0-\u30FF]'),
            'kanji': re.compile(r'[\u4E00-\u9FAF]'),
            'numbers': re.compile(r'[\d]+'),
            'english': re.compile(r'[a-zA-Z]+'),
        }
        
        # ジャンル・特徴語彙
        self.genre_keywords = self._build_genre_keywords()
        self.quality_indicators = self._build_quality_indicators()
        
        # 統計情報
        self.processing_stats = {
            'total_videos_processed': 0,
            'text_features_extracted': 0,
            'categorical_features_encoded': 0,
            'missing_data_imputed': 0,
            'processing_time': 0
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

    def _build_genre_keywords(self) -> Dict[str, List[str]]:
        """ジャンル特徴語彙の構築"""
        return {
            'vr': ['vr', 'バーチャル', '仮想現実', '360度'],
            'amateur': ['素人', 'アマチュア', '一般人', '初撮り', 'ナンパ'],
            'mature': ['熟女', '人妻', 'ミセス', '年上', '30代', '40代', '50代'],
            'young': ['学生', '制服', 'jk', '若い', '10代', '新人', '18歳'],
            'big_breasts': ['爆乳', '巨乳', 'lカップ', 'パイズリ', 'おっぱい', 'gカップ'],
            'fetish': ['フェチ', 'm女', 'sm', '調教', '変態', 'ドm'],
            'anal': ['アナル', 'お尻', '肛門', 'アナル開発'],
            'group': ['乱交', '3p', '4p', '複数', 'ハーレム', 'ntr'],
            'cosplay': ['コスプレ', 'コス', 'メイド', 'ca', 'ol'],
            'creampie': ['中出し', 'クリームパイ', '生ハメ', '孕ませ'],
            'outdoor': ['野外', '屋外', '露出', '青姦'],
            'lesbian': ['レズ', 'レズビアン', '女同士', 'ビアン'],
            'married': ['人妻', '不倫', '浮気', 'ntr', '寝取り'],
            'idol': ['アイドル', '芸能人', '女優', 'av女優']
        }

    def _build_quality_indicators(self) -> Dict[str, List[str]]:
        """品質指標語彙の構築"""
        return {
            'high_quality': ['4k', 'vr', '高画質', 'プレミアム', '独占', '限定'],
            'popular_series': ['シリーズ', '続編', 'part', 'vol', '第'],
            'debut': ['デビュー', '初', '新人', 'debut', 'first'],
            'award': ['大賞', '受賞', 'グランプリ', 'ベスト', 'ランキング'],
            'exclusive': ['独占', '限定', '専属', 'exclusive', 'special'],
            'long_duration': ['長時間', '240分', '4時間', '大容量', 'best'],
        }

    def load_video_metadata(self, 
                           conn: psycopg2.extensions.connection,
                           batch_size: int = 1000,
                           limit: Optional[int] = None) -> List[VideoMetadata]:
        """ビデオメタデータのバッチ読み込み"""
        logger.info("ビデオメタデータ読み込み開始...")
        
        cursor = conn.cursor()
        
        # メタデータクエリ（関連テーブルを含む）
        query = """
        SELECT 
            v.id,
            v.title,
            v.description,
            v.maker,
            v.genre,
            v.price,
            v.duration_seconds,
            v.external_id,
            v.source,
            v.created_at,
            ARRAY_AGG(DISTINCT p.name) FILTER (WHERE p.name IS NOT NULL) as performers,
            ARRAY_AGG(DISTINCT t.name) FILTER (WHERE t.name IS NOT NULL) as tags
        FROM videos v
        LEFT JOIN video_performers vp ON v.id = vp.video_id
        LEFT JOIN performers p ON vp.performer_id = p.id
        LEFT JOIN video_tags vt ON v.id = vt.video_id
        LEFT JOIN tags t ON vt.tag_id = t.id
        GROUP BY v.id, v.title, v.description, v.maker, v.genre, 
                 v.price, v.duration_seconds, v.external_id, v.source, v.created_at
        ORDER BY v.created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        videos = []
        batch = cursor.fetchmany(batch_size)
        
        while batch:
            for row in batch:
                video = VideoMetadata(
                    video_id=str(row['id']),
                    title=row['title'] or '',
                    description=row['description'] or '',
                    maker=row['maker'] or 'unknown',
                    genre=row['genre'] or 'general',
                    price=float(row['price']) if row['price'] else 0.0,
                    duration_seconds=int(row['duration_seconds']) if row['duration_seconds'] else 0,
                    performers=row['performers'] or [],
                    tags=row['tags'] or [],
                    external_id=row['external_id'] or '',
                    source=row['source'] or '',
                    created_at=row['created_at']
                )
                videos.append(video)
            
            batch = cursor.fetchmany(batch_size)
        
        logger.info(f"ビデオメタデータ読み込み完了: {len(videos)} 件")
        return videos

    def extract_text_features(self, videos: List[VideoMetadata]) -> Tuple[np.ndarray, Dict]:
        """テキスト特徴の抽出（TF-IDF）"""
        logger.info("テキスト特徴抽出開始...")
        
        # テキストの結合と前処理
        combined_texts = []
        for video in videos:
            # タイトル、説明、パフォーマー、タグを結合
            text_parts = [
                video.title,
                video.description,
                ' '.join(video.performers),
                ' '.join(video.tags)
            ]
            combined_text = ' '.join(part for part in text_parts if part)
            
            # 日本語テキストの前処理
            processed_text = self._preprocess_japanese_text(combined_text)
            combined_texts.append(processed_text)
        
        # TF-IDF ベクトル化
        tfidf_matrix = self.text_vectorizer.fit_transform(combined_texts)
        
        # 密行列に変換してリサイズ
        text_features = tfidf_matrix.toarray()
        if text_features.shape[1] < self.text_dim:
            # パディング
            padding = np.zeros((text_features.shape[0], self.text_dim - text_features.shape[1]))
            text_features = np.hstack([text_features, padding])
        elif text_features.shape[1] > self.text_dim:
            # トランケーション
            text_features = text_features[:, :self.text_dim]
        
        # 統計情報
        text_stats = {
            'vocabulary_size': len(self.text_vectorizer.vocabulary_),
            'average_text_length': np.mean([len(text) for text in combined_texts]),
            'feature_sparsity': 1 - np.count_nonzero(text_features) / text_features.size
        }
        
        logger.info(f"テキスト特徴抽出完了: {text_features.shape}")
        self.processing_stats['text_features_extracted'] = text_features.shape[0]
        
        return text_features, text_stats

    def _preprocess_japanese_text(self, text: str) -> str:
        """日本語テキストの前処理"""
        if not text:
            return ""
        
        # 基本的なクリーニング
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # 記号の削除
        text = re.sub(r'\s+', ' ', text)      # 空白の正規化
        
        # 日本語特有の処理
        # 数字を統一
        text = re.sub(r'\d+', 'NUM', text)
        
        # 特殊なパターンの正規化
        text = re.sub(r'(vr|VR)', 'vr', text)
        text = re.sub(r'(4k|4K)', '4k', text)
        
        return text.strip()

    def extract_categorical_features(self, videos: List[VideoMetadata]) -> np.ndarray:
        """カテゴリ特徴の抽出"""
        logger.info("カテゴリ特徴抽出開始...")
        
        # カテゴリデータの準備
        genres = [video.genre for video in videos]
        makers = [video.maker for video in videos]
        sources = [video.source for video in videos]
        
        # エンコーディング
        genre_encoded = self.genre_encoder.fit_transform(genres)
        maker_encoded = self.maker_encoder.fit_transform(makers)
        source_encoded = self.source_encoder.fit_transform(sources)
        
        # ワンホットエンコーディング風の処理（多クラス対応）
        max_genre = len(self.genre_encoder.classes_)
        max_maker = len(self.maker_encoder.classes_)
        max_source = len(self.source_encoder.classes_)
        
        categorical_features = []
        
        for i, video in enumerate(videos):
            # ジャンル特徴（20次元）
            genre_feat = np.zeros(20)
            if genre_encoded[i] < 20:
                genre_feat[genre_encoded[i]] = 1.0
            
            # メーカー特徴（30次元）
            maker_feat = np.zeros(30)
            if maker_encoded[i] < 30:
                maker_feat[maker_encoded[i]] = 1.0
            
            # ソース特徴（5次元）
            source_feat = np.zeros(5)
            if source_encoded[i] < 5:
                source_feat[source_encoded[i]] = 1.0
            
            # ジャンルキーワード特徴（30次元）
            keyword_feat = self._extract_genre_keyword_features(video)
            
            # 品質指標特徴（15次元）
            quality_feat = self._extract_quality_features(video)
            
            # 結合（合計100次元）
            combined_feat = np.concatenate([
                genre_feat, maker_feat, source_feat, 
                keyword_feat, quality_feat
            ])
            
            categorical_features.append(combined_feat)
        
        categorical_matrix = np.array(categorical_features)
        
        logger.info(f"カテゴリ特徴抽出完了: {categorical_matrix.shape}")
        self.processing_stats['categorical_features_encoded'] = len(videos)
        
        return categorical_matrix

    def _extract_genre_keyword_features(self, video: VideoMetadata) -> np.ndarray:
        """ジャンルキーワード特徴の抽出"""
        combined_text = f"{video.title} {video.description} {' '.join(video.tags)}".lower()
        
        genre_features = np.zeros(len(self.genre_keywords))
        
        for i, (genre, keywords) in enumerate(self.genre_keywords.items()):
            # キーワードマッチング
            score = sum(1 for keyword in keywords if keyword in combined_text)
            genre_features[i] = min(score / len(keywords), 1.0)  # 正規化
        
        # 30次元にリサイズ
        if len(genre_features) > 30:
            genre_features = genre_features[:30]
        else:
            padding = np.zeros(30 - len(genre_features))
            genre_features = np.concatenate([genre_features, padding])
        
        return genre_features

    def _extract_quality_features(self, video: VideoMetadata) -> np.ndarray:
        """品質特徴の抽出"""
        combined_text = f"{video.title} {video.description}".lower()
        
        quality_features = np.zeros(len(self.quality_indicators))
        
        for i, (quality_type, indicators) in enumerate(self.quality_indicators.items()):
            score = sum(1 for indicator in indicators if indicator in combined_text)
            quality_features[i] = min(score / len(indicators), 1.0)  # 正規化
        
        # 15次元にリサイズ
        if len(quality_features) > 15:
            quality_features = quality_features[:15]
        else:
            padding = np.zeros(15 - len(quality_features))
            quality_features = np.concatenate([quality_features, padding])
        
        return quality_features

    def extract_numerical_features(self, videos: List[VideoMetadata]) -> np.ndarray:
        """数値特徴の抽出"""
        logger.info("数値特徴抽出開始...")
        
        numerical_features = []
        
        for video in videos:
            # 基本数値特徴
            price = video.price if video.price else 0.0
            duration = video.duration_seconds if video.duration_seconds else 0
            
            # 派生特徴
            price_per_minute = price / (duration / 60) if duration > 0 else 0.0
            
            # テキスト統計
            title_length = len(video.title) if video.title else 0
            desc_length = len(video.description) if video.description else 0
            performer_count = len(video.performers)
            tag_count = len(video.tags)
            
            # 日本語テキスト特徴
            title_hiragana_ratio = self._calculate_character_ratio(video.title, 'hiragana')
            title_katakana_ratio = self._calculate_character_ratio(video.title, 'katakana')
            title_kanji_ratio = self._calculate_character_ratio(video.title, 'kanji')
            title_english_ratio = self._calculate_character_ratio(video.title, 'english')
            
            # 時系列特徴
            days_since_creation = (datetime.now() - video.created_at).days if video.created_at else 0
            
            # 20次元の数値特徴ベクトル
            numerical_vector = np.array([
                price,
                duration / 3600.0,  # 時間単位に変換
                price_per_minute,
                title_length / 100.0,  # 正規化
                desc_length / 1000.0,  # 正規化
                performer_count,
                tag_count,
                title_hiragana_ratio,
                title_katakana_ratio,
                title_kanji_ratio,
                title_english_ratio,
                days_since_creation / 365.0,  # 年単位に変換
                # 追加の統計特徴
                np.log1p(price),  # 対数変換価格
                np.sqrt(duration) / 100.0,  # 平方根変換時間
                min(performer_count / 5.0, 1.0),  # パフォーマー密度
                min(tag_count / 10.0, 1.0),  # タグ密度
                1.0 if price > 1000 else 0.0,  # 高価格フラグ
                1.0 if duration > 7200 else 0.0,  # 長時間フラグ
                1.0 if performer_count > 2 else 0.0,  # 複数パフォーマーフラグ
                title_length * desc_length / 10000.0  # コンテンツ充実度
            ])
            
            numerical_features.append(numerical_vector)
        
        numerical_matrix = np.array(numerical_features)
        
        # 正規化
        numerical_matrix = self.numerical_scaler.fit_transform(numerical_matrix)
        
        logger.info(f"数値特徴抽出完了: {numerical_matrix.shape}")
        
        return numerical_matrix

    def _calculate_character_ratio(self, text: str, char_type: str) -> float:
        """文字種別比率の計算"""
        if not text:
            return 0.0
        
        pattern = self.japanese_patterns.get(char_type)
        if not pattern:
            return 0.0
        
        matches = pattern.findall(text)
        return len(matches) / len(text) if text else 0.0

    def extract_semantic_features(self, videos: List[VideoMetadata]) -> np.ndarray:
        """意味的特徴の抽出"""
        logger.info("意味的特徴抽出開始...")
        
        semantic_features = []
        
        for video in videos:
            # 意味的特徴の構築（100次元）
            semantic_vector = np.zeros(100)
            
            # テキスト複雑度特徴（20次元）
            complexity_features = self._extract_text_complexity_features(video)
            semantic_vector[:20] = complexity_features
            
            # コンテンツ特徴（30次元）
            content_features = self._extract_content_semantic_features(video)
            semantic_vector[20:50] = content_features
            
            # 関連性特徴（25次元）
            relation_features = self._extract_relation_features(video)
            semantic_vector[50:75] = relation_features
            
            # 品質推定特徴（25次元）
            quality_features = self._extract_semantic_quality_features(video)
            semantic_vector[75:100] = quality_features
            
            semantic_features.append(semantic_vector)
        
        semantic_matrix = np.array(semantic_features)
        
        logger.info(f"意味的特徴抽出完了: {semantic_matrix.shape}")
        
        return semantic_matrix

    def _extract_text_complexity_features(self, video: VideoMetadata) -> np.ndarray:
        """テキスト複雑度特徴"""
        title = video.title or ""
        desc = video.description or ""
        
        features = np.zeros(20)
        
        if title:
            # タイトル特徴
            features[0] = len(title.split())  # 単語数
            features[1] = np.mean([len(word) for word in title.split()]) if title.split() else 0  # 平均単語長
            features[2] = len(set(title.split())) / max(len(title.split()), 1)  # 語彙多様性
            features[3] = title.count('!') + title.count('？') + title.count('♪')  # 感嘆符数
        
        if desc:
            # 説明文特徴
            features[4] = len(desc.split())  # 単語数
            features[5] = np.mean([len(word) for word in desc.split()]) if desc.split() else 0
            features[6] = len(set(desc.split())) / max(len(desc.split()), 1)
            features[7] = desc.count('\n')  # 改行数
        
        # 日本語特有の特徴
        combined_text = title + " " + desc
        features[8] = len(self.japanese_patterns['hiragana'].findall(combined_text))
        features[9] = len(self.japanese_patterns['katakana'].findall(combined_text))
        features[10] = len(self.japanese_patterns['kanji'].findall(combined_text))
        features[11] = len(self.japanese_patterns['numbers'].findall(combined_text))
        features[12] = len(self.japanese_patterns['english'].findall(combined_text))
        
        # 意味的密度
        features[13] = len([w for w in combined_text.split() if len(w) > 2])  # 長い単語数
        features[14] = combined_text.count('、') + combined_text.count('。')  # 句読点数
        features[15] = 1.0 if any(keyword in combined_text.lower() for keyword in ['限定', '初回', '特典']) else 0.0
        features[16] = 1.0 if any(keyword in combined_text.lower() for keyword in ['人気', 'ランキング', '話題']) else 0.0
        features[17] = len(video.performers) / 10.0  # パフォーマー正規化数
        features[18] = len(video.tags) / 20.0  # タグ正規化数
        features[19] = min(len(combined_text) / 200.0, 1.0)  # テキスト長正規化
        
        return features

    def _extract_content_semantic_features(self, video: VideoMetadata) -> np.ndarray:
        """コンテンツ意味的特徴"""
        features = np.zeros(30)
        combined_text = f"{video.title} {video.description}".lower()
        
        # コンテンツタイプ特徴
        content_types = {
            'drama': ['ドラマ', '物語', 'ストーリー', 'シナリオ'],
            'reality': ['リアル', 'ドキュメント', '本物', 'ガチ'],
            'fantasy': ['ファンタジー', '夢', '妄想', 'if'],
            'amateur': ['素人', 'ナンパ', '一般人'],
            'professional': ['女優', 'プロ', '撮影'],
            'series': ['シリーズ', '連続', '続編'],
            'special': ['特別', 'スペシャル', '限定'],
            'compilation': ['総集編', 'ベスト', 'まとめ'],
        }
        
        for i, (content_type, keywords) in enumerate(content_types.items()):
            if i < 15:  # 最初の15次元
                features[i] = 1.0 if any(kw in combined_text for kw in keywords) else 0.0
        
        # ジャンル横断特徴
        genre_cross_features = [
            ('action', ['激しい', 'ハード', '激烈']),
            ('romantic', ['ロマンス', '恋愛', 'ラブ']),
            ('comedy', ['コメディ', '笑い', 'ギャグ']),
            ('artistic', ['芸術', 'アート', '美しい']),
            ('extreme', ['過激', 'エクストリーム', '限界']),
        ]
        
        for i, (genre, keywords) in enumerate(genre_cross_features):
            if 15 + i < 30:
                features[15 + i] = 1.0 if any(kw in combined_text for kw in keywords) else 0.0
        
        # 残りの次元は統計的特徴
        features[20] = len(video.performers) / 5.0  # パフォーマー密度
        features[21] = len(video.tags) / 15.0  # タグ密度
        features[22] = 1.0 if video.price > 500 else 0.0  # 価格層
        features[23] = 1.0 if video.duration_seconds > 3600 else 0.0  # 長さ層
        features[24] = len(set(video.performers)) / max(len(video.performers), 1)  # パフォーマー多様性
        
        return features

    def _extract_relation_features(self, video: VideoMetadata) -> np.ndarray:
        """関連性特徴"""
        features = np.zeros(25)
        
        # パフォーマー関連特徴
        performer_names = [p.lower() for p in video.performers]
        features[0] = len(set(performer_names))  # ユニークパフォーマー数
        features[1] = len(video.performers) - len(set(performer_names))  # 重複数
        
        # タグ関連特徴
        tag_names = [t.lower() for t in video.tags]
        features[2] = len(set(tag_names))  # ユニークタグ数
        features[3] = len(video.tags) - len(set(tag_names))  # タグ重複数
        
        # 意味的関連性（簡易版）
        title_words = set(video.title.lower().split()) if video.title else set()
        desc_words = set(video.description.lower().split()) if video.description else set()
        
        features[4] = len(title_words & desc_words) / max(len(title_words | desc_words), 1)  # 語彙重複率
        features[5] = len(title_words) / max(len(desc_words), 1) if desc_words else 0  # タイトル/説明比
        
        # カテゴリ一貫性
        features[6] = 1.0 if video.genre in video.title.lower() else 0.0
        features[7] = 1.0 if video.maker.lower() in video.title.lower() else 0.0
        
        return features

    def _extract_semantic_quality_features(self, video: VideoMetadata) -> np.ndarray:
        """意味的品質特徴"""
        features = np.zeros(25)
        
        combined_text = f"{video.title} {video.description}".lower()
        
        # 品質指標
        quality_indicators = {
            'high_production': ['hd', '4k', 'ハイビジョン', '高画質', 'クリア'],
            'exclusive': ['独占', '限定', '先行', 'プレミアム', 'vip'],
            'popular': ['人気', 'ベスト', 'ランキング', 'top', '#1'],
            'award': ['受賞', '大賞', 'グランプリ', 'award', '表彰'],
            'series_quality': ['シリーズ', '第', 'vol', 'part', '編'],
        }
        
        for i, (quality_type, keywords) in enumerate(quality_indicators.items()):
            if i < 10:
                features[i] = sum(1 for kw in keywords if kw in combined_text) / len(keywords)
        
        # メタデータ品質
        features[10] = 1.0 if video.title and len(video.title) > 10 else 0.0  # タイトル充実度
        features[11] = 1.0 if video.description and len(video.description) > 50 else 0.0  # 説明充実度
        features[12] = min(len(video.performers) / 3.0, 1.0)  # パフォーマー情報充実度
        features[13] = min(len(video.tags) / 8.0, 1.0)  # タグ情報充実度
        features[14] = 1.0 if video.duration_seconds > 1800 else 0.0  # 十分な長さ
        
        return features

    def extract_popularity_features(self, videos: List[VideoMetadata], conn: psycopg2.extensions.connection) -> np.ndarray:
        """人気度特徴の抽出"""
        logger.info("人気度特徴抽出開始...")
        
        cursor = conn.cursor()
        popularity_features = []
        
        for video in videos:
            # 人気度統計の取得
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN decision_type = 'like' THEN 1 END) as likes,
                    COUNT(CASE WHEN decision_type = 'nope' THEN 1 END) as nopes,
                    COUNT(*) as total_decisions
                FROM user_video_decisions 
                WHERE video_id = %s
            """, (video.video_id,))
            
            result = cursor.fetchone()
            likes = result['likes'] if result else 0
            nopes = result['nopes'] if result else 0
            total_decisions = result['total_decisions'] if result else 0
            
            # 人気度特徴ベクトル（20次元）
            popularity_vector = np.array([
                likes,  # いいね数
                nopes,  # ダメ数
                total_decisions,  # 総決定数
                likes / max(total_decisions, 1),  # いいね率
                nopes / max(total_decisions, 1),  # ダメ率
                likes / max(likes + nopes, 1),  # 相対いいね率
                np.log1p(likes),  # 対数いいね数
                np.log1p(total_decisions),  # 対数総決定数
                1.0 if likes > 10 else 0.0,  # 人気閾値1
                1.0 if likes > 50 else 0.0,  # 人気閾値2
                1.0 if likes > 100 else 0.0,  # 人気閾値3
                1.0 if total_decisions > 20 else 0.0,  # 認知度閾値1
                1.0 if total_decisions > 100 else 0.0,  # 認知度閾値2
                likes * 2 + total_decisions,  # 複合人気スコア
                (datetime.now() - video.created_at).days if video.created_at else 0,  # 経過日数
                likes / max((datetime.now() - video.created_at).days, 1) if video.created_at else 0,  # 日次いいね率
                min(video.price / 1000.0, 2.0) if video.price else 0.0,  # 価格正規化
                video.duration_seconds / 3600.0 if video.duration_seconds else 0.0,  # 時間正規化
                len(video.performers),  # パフォーマー数
                len(video.tags)  # タグ数
            ])
            
            popularity_features.append(popularity_vector)
        
        popularity_matrix = np.array(popularity_features)
        
        # 正規化（人気度特徴は大きく変動するため）
        scaler = StandardScaler()
        popularity_matrix = scaler.fit_transform(popularity_matrix)
        
        logger.info(f"人気度特徴抽出完了: {popularity_matrix.shape}")
        
        return popularity_matrix

    def convert_to_feature_vectors(self, 
                                 videos: List[VideoMetadata],
                                 conn: psycopg2.extensions.connection) -> List[VideoFeatureVector]:
        """ビデオメタデータを768次元特徴ベクターに変換"""
        logger.info("特徴ベクター変換開始...")
        
        start_time = datetime.now()
        
        # 各種特徴の抽出
        text_features, _ = self.extract_text_features(videos)
        categorical_features = self.extract_categorical_features(videos)
        numerical_features = self.extract_numerical_features(videos)
        semantic_features = self.extract_semantic_features(videos)
        popularity_features = self.extract_popularity_features(videos, conn)
        
        # パディング特徴の生成
        padding_features = np.zeros((len(videos), self.padding_dim))
        
        # 特徴ベクターの構築
        feature_vectors = []
        
        for i, video in enumerate(videos):
            feature_vector = VideoFeatureVector(
                video_id=video.video_id,
                text_features=text_features[i],
                categorical_features=categorical_features[i],
                numerical_features=numerical_features[i],
                semantic_features=semantic_features[i],
                popularity_features=popularity_features[i],
                padding_features=padding_features[i]
            )
            feature_vectors.append(feature_vector)
        
        # 統計更新
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_stats.update({
            'total_videos_processed': len(videos),
            'processing_time': processing_time
        })
        
        logger.info(f"特徴ベクター変換完了: {len(feature_vectors)} ベクター, {processing_time:.2f}秒")
        
        return feature_vectors

    def convert_to_numpy_array(self, feature_vectors: List[VideoFeatureVector]) -> Tuple[np.ndarray, List[str]]:
        """特徴ベクターをNumPy配列に変換"""
        
        video_ids = [fv.video_id for fv in feature_vectors]
        feature_matrix = []
        
        for fv in feature_vectors:
            # 全特徴を結合して768次元ベクターを作成
            combined_features = np.concatenate([
                fv.text_features,
                fv.categorical_features,
                fv.numerical_features,
                fv.semantic_features,
                fv.popularity_features,
                fv.padding_features
            ])
            
            # 768次元を確保
            if len(combined_features) != self.target_feature_dim:
                if len(combined_features) > self.target_feature_dim:
                    combined_features = combined_features[:self.target_feature_dim]
                else:
                    padding_needed = self.target_feature_dim - len(combined_features)
                    combined_features = np.concatenate([
                        combined_features,
                        np.zeros(padding_needed)
                    ])
            
            feature_matrix.append(combined_features)
        
        return np.array(feature_matrix), video_ids

    def save_processing_artifacts(self, output_dir: str = "data/processed/item_features"):
        """前処理アーティファクトの保存"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 前処理器の保存
        artifacts = {
            'text_vectorizer': self.text_vectorizer,
            'genre_encoder': self.genre_encoder,
            'maker_encoder': self.maker_encoder,
            'source_encoder': self.source_encoder,
            'numerical_scaler': self.numerical_scaler,
            'genre_keywords': self.genre_keywords,
            'quality_indicators': self.quality_indicators,
            'target_feature_dim': self.target_feature_dim,
            'processing_stats': self.processing_stats
        }
        
        with open(f"{output_dir}/preprocessing_artifacts.pkl", 'wb') as f:
            pickle.dump(artifacts, f)
        
        # メタデータの保存
        metadata = {
            'created_at': datetime.now().isoformat(),
            'feature_dimensions': {
                'text_dim': self.text_dim,
                'categorical_dim': self.categorical_dim,
                'numerical_dim': self.numerical_dim,
                'semantic_dim': self.semantic_dim,
                'popularity_dim': self.popularity_dim,
                'total_dim': self.target_feature_dim
            },
            'processing_stats': self.processing_stats
        }
        
        with open(f"{output_dir}/feature_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"前処理アーティファクト保存完了: {output_dir}")

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='強化されたアイテム特徴処理')
    parser.add_argument('--db-url', required=True, help='PostgreSQL接続URL')
    parser.add_argument('--output-path', 
                       default='data/processed/enhanced_item_features.json',
                       help='出力パス')
    parser.add_argument('--limit', type=int, help='処理するビデオ数の制限')
    parser.add_argument('--tfidf-features', type=int, default=500,
                       help='TF-IDF特徴数')
    
    args = parser.parse_args()
    
    # 特徴処理器の初期化
    processor = EnhancedItemFeatureProcessor(
        db_connection_string=args.db_url,
        tfidf_max_features=args.tfidf_features
    )
    
    try:
        # データベース接続
        conn = processor.connect_db()
        
        # ビデオメタデータの読み込み
        videos = processor.load_video_metadata(conn, limit=args.limit)
        
        # 特徴ベクターの生成
        feature_vectors = processor.convert_to_feature_vectors(videos, conn)
        
        # NumPy配列への変換
        feature_matrix, video_ids = processor.convert_to_numpy_array(feature_vectors)
        
        # 結果の保存
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_videos': len(videos),
                'feature_dimension': processor.target_feature_dim,
                'processing_stats': processor.processing_stats
            },
            'video_ids': video_ids,
            'features': feature_matrix.tolist()
        }
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # アーティファクトの保存
        processor.save_processing_artifacts()
        
        # 統計情報の表示
        logger.info("=== 処理統計 ===")
        for key, value in processor.processing_stats.items():
            logger.info(f"{key}: {value}")
        
        logger.info(f"特徴行列形状: {feature_matrix.shape}")
        logger.info("アイテム特徴処理完了")
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    main()