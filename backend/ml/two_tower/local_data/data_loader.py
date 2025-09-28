#!/usr/bin/env python3
"""
Enhanced Data Loader for Two-Tower Model
豊富な特徴量を活用したデータローダー
- タグ情報のワンホットエンコーディング
- テキスト特徴量の活用
- メーカー・シリーズ・監督の埋め込み
- 全メタデータの活用
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from collections import Counter
import re

class SupabaseDataLoader:
    """豊富な特徴量を活用するデータローダー"""

    def __init__(self):
        # Supabase商用DB互換データを読み込み
        self.data_dir = Path("../../../data_processing/local_compatible_data")

        # エンコーダーを保存
        self.encoders = {}
        self.scalers = {}

        # データ
        self.profiles_df = None
        self.videos_df = None
        self.decisions_df = None

        # 特徴量関連
        self.tag_binarizer = MultiLabelBinarizer()
        self.all_tags = set()
        self.frequent_tags = set()  # 頻出タグセット
        self.text_features = {}

    def load_all_data(self):
        """全データを読み込み"""
        print("📊 Enhanced Data Loading...")

        self.profiles_df = self._load_profiles()
        self.videos_df = self._load_videos()
        self.decisions_df = self._load_decisions()

        print(f"✅ Loaded: {len(self.profiles_df)} users, {len(self.videos_df)} videos, {len(self.decisions_df)} decisions")

        # タグ分析
        self._analyze_tags()

        # テキスト特徴量分析
        self._analyze_text_features()

    def _load_profiles(self):
        """ユーザープロファイル読み込み"""
        with open(self.data_dir / "profiles.json", 'r', encoding='utf-8') as f:
            profiles = json.load(f)

        # 日付カラムをパース
        for profile in profiles:
            if profile.get('created_at'):
                profile['created_at'] = pd.to_datetime(profile['created_at'])

        return pd.DataFrame(profiles)

    def _load_videos(self):
        """動画データ読み込み"""
        with open(self.data_dir / "videos_subset.json", 'r', encoding='utf-8') as f:
            videos = json.load(f)

        # 日付カラムをパース
        for video in videos:
            if video.get('product_released_at'):
                video['product_released_at'] = pd.to_datetime(video['product_released_at'])
            if video.get('created_at'):
                video['created_at'] = pd.to_datetime(video['created_at'])

        return pd.DataFrame(videos)

    def _load_decisions(self):
        """ユーザー動画決定データ読み込み"""
        with open(self.data_dir / "user_video_decisions.json", 'r', encoding='utf-8') as f:
            decisions = json.load(f)

        # 日付カラムをパース
        for decision in decisions:
            if decision.get('created_at'):
                decision['created_at'] = pd.to_datetime(decision['created_at'])

        return pd.DataFrame(decisions)

    def _analyze_tags(self):
        """タグ分析とエンコーディング準備"""
        print("🏷️ Analyzing tags...")

        # 全タグを収集
        all_tags = []
        for _, video in self.videos_df.iterrows():
            if isinstance(video['tags'], list):
                all_tags.extend(video['tags'])
                self.all_tags.update(video['tags'])

        # タグ頻度分析
        tag_counts = Counter(all_tags)
        print(f"📈 Total unique tags: {len(tag_counts)}")
        print(f"📈 Top 10 tags: {tag_counts.most_common(10)}")

        # 頻度の低いタグを除外（3回未満）
        frequent_tags = {tag for tag, count in tag_counts.items() if count >= 3}
        self.frequent_tags = frequent_tags
        print(f"📈 Frequent tags (≥3 occurrences): {len(frequent_tags)}")

        # MultiLabelBinarizerを準備
        video_tags = []
        for _, video in self.videos_df.iterrows():
            if isinstance(video['tags'], list):
                # 頻度の高いタグのみ使用
                filtered_tags = [tag for tag in video['tags'] if tag in frequent_tags]
                video_tags.append(filtered_tags)
            else:
                video_tags.append([])

        self.tag_binarizer.fit(video_tags)
        print(f"📈 Tag binarizer vocabulary: {len(self.tag_binarizer.classes_)}")

    def _analyze_text_features(self):
        """テキスト特徴量分析"""
        print("📝 Analyzing text features...")

        # タイトル長の統計
        title_lengths = self.videos_df['title'].str.len()
        print(f"📈 Title length: mean={title_lengths.mean():.1f}, max={title_lengths.max()}")

        # メーカー、監督、シリーズの統計
        for col in ['maker', 'director', 'series']:
            if col in self.videos_df.columns:
                unique_count = self.videos_df[col].nunique()
                print(f"📈 {col}: {unique_count} unique values")

    def get_realtime_training_data(self):
        """リアルタイム型学習データを作成（LIKE/NOPE履歴ベース）"""
        print("🔄 Creating realtime training data...")

        # 各ユーザーのLIKE/NOPE履歴を取得
        training_samples = []

        for user_id in self.decisions_df['user_id'].unique():
            user_decisions = self.decisions_df[self.decisions_df['user_id'] == user_id]

            # LIKE履歴とNOPE履歴を分別
            liked_videos = user_decisions[user_decisions['decision_type'] == 'like']['video_id'].tolist()
            noped_videos = user_decisions[user_decisions['decision_type'] == 'nope']['video_id'].tolist()

            # 各決定に対してサンプル作成
            for _, decision in user_decisions.iterrows():
                # その決定時点での履歴を作成（時系列考慮）
                decision_time = decision['created_at']

                # その時点より前のLIKE/NOPE履歴
                prior_decisions = user_decisions[user_decisions['created_at'] < decision_time]
                prior_likes = prior_decisions[prior_decisions['decision_type'] == 'like']['video_id'].tolist()
                prior_nopes = prior_decisions[prior_decisions['decision_type'] == 'nope']['video_id'].tolist()

                # 動画情報を取得
                video_info = self.videos_df[self.videos_df['id'] == decision['video_id']].iloc[0]

                sample = {
                    'user_id': user_id,
                    'video_id': decision['video_id'],
                    'user_like_history': prior_likes,
                    'user_nope_history': prior_nopes,
                    'video_features': video_info,
                    'label': 1 if decision['decision_type'] == 'like' else 0
                }
                training_samples.append(sample)

        print(f"📈 Realtime training data: {len(training_samples)} samples")
        return training_samples

    def get_enhanced_user_features(self, training_data: pd.DataFrame) -> np.ndarray:
        """拡張ユーザー特徴量を生成"""
        print("👤 Creating enhanced user features...")

        user_features = []

        for user_id in training_data['user_id'].unique():
            user_data = training_data[training_data['user_id'] == user_id].iloc[0]
            user_decisions = training_data[training_data['user_id'] == user_id]

            # 基本特徴量
            basic_features = [
                len(user_data['display_name']) if pd.notna(user_data['display_name']) else 0,  # display_name_length
                (datetime.now() - user_data['created_at_user']).days,  # account_age_days
                len(user_decisions),  # total_decisions
                user_decisions['decision_type'].apply(lambda x: 1 if x == 'like' else 0).mean()  # like_ratio
            ]

            # 行動特徴量（ユーザーの好みを反映）
            liked_videos = user_decisions[user_decisions['decision_type'] == 'like']

            # 価格帯の好み
            if len(liked_videos) > 0:
                avg_liked_price = liked_videos['price'].mean() if 'price' in liked_videos.columns else 0
                price_preference_high = (liked_videos['price'] > 1000).mean() if 'price' in liked_videos.columns else 0
            else:
                avg_liked_price = 0
                price_preference_high = 0

            behavioral_features = [
                avg_liked_price,  # 平均価格帯
                price_preference_high,  # 高価格帯好み率
                len(liked_videos),  # like数
                len(user_decisions) - len(liked_videos)  # nope数
            ]

            # タグの好み分析
            if len(liked_videos) > 0:
                # ユーザーがlikeした動画のタグを集計
                liked_tags = []
                for _, video in liked_videos.iterrows():
                    if isinstance(video['tags'], list):
                        liked_tags.extend([tag for tag in video['tags'] if tag in self.frequent_tags])

                tag_preferences = Counter(liked_tags)
                # 上位5つのタグ好みを特徴量に
                top_tag_prefs = [tag_preferences.get(tag, 0) for tag in list(self.frequent_tags)[:5]]
            else:
                top_tag_prefs = [0] * 5

            # 全特徴量結合
            features = basic_features + behavioral_features + top_tag_prefs
            user_features.append(features)

        user_features_array = np.array(user_features)
        print(f"👤 User features shape: {user_features_array.shape}")

        return user_features_array

    def get_enhanced_video_features(self, training_data: pd.DataFrame) -> np.ndarray:
        """拡張動画特徴量を生成"""
        print("🎬 Creating enhanced video features...")

        video_features = []

        for video_id in training_data['video_id'].unique():
            video_data = training_data[training_data['video_id'] == video_id].iloc[0]
            video_decisions = training_data[training_data['video_id'] == video_id]

            # 基本特徴量
            basic_features = [
                len(video_data['title']) if pd.notna(video_data['title']) else 0,  # title_length
                video_data['price'] if pd.notna(video_data['price']) else 0,  # price
                video_data['product_released_at'].year if pd.notna(video_data['product_released_at']) else 2024,  # release_year
                1 if pd.notna(video_data['thumbnail_url']) else 0,  # has_thumbnail
                len(video_decisions),  # total_decisions
                video_decisions['decision_type'].apply(lambda x: 1 if x == 'like' else 0).mean()  # like_ratio
            ]

            # 追加メタデータ特徴量
            metadata_features = [
                len(video_data['maker']) if pd.notna(video_data['maker']) else 0,  # maker_length
                len(video_data['director']) if pd.notna(video_data['director']) else 0,  # director_length
                len(video_data['series']) if pd.notna(video_data['series']) else 0,  # series_length
                1 if pd.notna(video_data['sample_video_url']) else 0,  # has_sample
                1 if pd.notna(video_data['preview_video_url']) else 0,  # has_preview
                len(video_data['image_urls']) if isinstance(video_data['image_urls'], list) else 0,  # image_count
                video_data['duration_seconds'] if pd.notna(video_data['duration_seconds']) else 0  # duration
            ]

            # タグ特徴量（ワンホット）
            if isinstance(video_data['tags'], list):
                filtered_tags = [tag for tag in video_data['tags'] if tag in self.frequent_tags]
                tag_features = self.tag_binarizer.transform([filtered_tags])[0]
            else:
                tag_features = self.tag_binarizer.transform([[]])[0]

            # 全特徴量結合
            features = basic_features + metadata_features + tag_features.tolist()
            video_features.append(features)

        video_features_array = np.array(video_features)
        print(f"🎬 Video features shape: {video_features_array.shape}")

        return video_features_array

    def get_single_user_features(self, row: pd.Series) -> np.ndarray:
        """単一ユーザーの拡張特徴量を生成"""
        user_data = row

        # 基本特徴量
        basic_features = [
            len(user_data['display_name']) if pd.notna(user_data['display_name']) else 0,  # display_name_length
            (datetime.now() - user_data['created_at_user']).days,  # account_age_days
            1,  # このサンプルでの決定数
            1 if user_data['decision_type'] == 'like' else 0  # この決定のlike/nope
        ]

        # 行動特徴量（簡易版）
        behavioral_features = [
            user_data['price'] if pd.notna(user_data['price']) else 0,  # この動画の価格（好み推定用）
            1 if user_data['price'] > 1000 else 0,  # 高価格帯判定
            1 if user_data['decision_type'] == 'like' else 0,  # like判定
            1 if user_data['decision_type'] == 'nope' else 0   # nope判定
        ]

        # タグの好み分析（簡易版）
        if isinstance(user_data['tags'], list) and user_data['decision_type'] == 'like':
            liked_tags = [tag for tag in user_data['tags'] if tag in self.frequent_tags]
            top_tag_prefs = [1 if tag in liked_tags else 0 for tag in list(self.frequent_tags)[:5]]
        else:
            top_tag_prefs = [0] * 5

        # 全特徴量結合
        features = basic_features + behavioral_features + top_tag_prefs
        return np.array(features)

    def get_single_video_features(self, row: pd.Series) -> np.ndarray:
        """単一動画の拡張特徴量を生成"""
        video_data = row

        # 基本特徴量
        basic_features = [
            len(video_data['title']) if pd.notna(video_data['title']) else 0,  # title_length
            video_data['price'] if pd.notna(video_data['price']) else 0,  # price
            video_data['product_released_at'].year if pd.notna(video_data['product_released_at']) else 2024,  # release_year
            1 if pd.notna(video_data['thumbnail_url']) else 0,  # has_thumbnail
            1,  # この動画での決定数（1つ）
            1 if video_data['decision_type'] == 'like' else 0  # like率（0または1）
        ]

        # 追加メタデータ特徴量
        metadata_features = [
            len(video_data['maker']) if pd.notna(video_data['maker']) else 0,  # maker_length
            len(video_data['director']) if pd.notna(video_data['director']) else 0,  # director_length
            len(video_data['series']) if pd.notna(video_data['series']) else 0,  # series_length
            1 if pd.notna(video_data['sample_video_url']) else 0,  # has_sample
            1 if pd.notna(video_data['preview_video_url']) else 0,  # has_preview
            len(video_data['image_urls']) if isinstance(video_data['image_urls'], list) else 0,  # image_count
            video_data['duration_seconds'] if pd.notna(video_data['duration_seconds']) else 0  # duration
        ]

        # タグ特徴量（ワンホット）
        if isinstance(video_data['tags'], list):
            filtered_tags = [tag for tag in video_data['tags'] if tag in self.frequent_tags]
            tag_features = self.tag_binarizer.transform([filtered_tags])[0]
        else:
            tag_features = self.tag_binarizer.transform([[]])[0]

        # 全特徴量結合
        features = basic_features + metadata_features + tag_features.tolist()
        return np.array(features)

if __name__ == "__main__":
    # テスト実行
    loader = SupabaseDataLoader()
    loader.load_all_data()

    training_data, labels = loader.get_enhanced_training_data()
    user_features = loader.get_enhanced_user_features(training_data)
    video_features = loader.get_enhanced_video_features(training_data)

    print(f"\n🎯 Final Summary:")
    print(f"   Training samples: {len(labels)}")
    print(f"   User features: {user_features.shape[1]} dimensions")
    print(f"   Video features: {video_features.shape[1]} dimensions")
    print(f"   Tag vocabulary: {len(loader.tag_binarizer.classes_)} tags")