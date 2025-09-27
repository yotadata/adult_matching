"""
Supabase互換構造でローカルデータを構築

生成するファイル:
1. profiles.json - ユーザープロファイル
2. user_video_decisions.json - ユーザー判定データ
3. videos_subset.json - マッチした動画データ
4. compatible_schema.sql - テーブル構造定義
5. import_data.sql - データインポート用SQL
"""

import asyncio
import aiohttp
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import os

class LocalCompatibleDataBuilder:
    def __init__(self):
        self.supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
        self.supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

        # 出力ディレクトリ
        self.output_dir = Path("../local_compatible_data")
        self.output_dir.mkdir(exist_ok=True)

        # データディレクトリ
        self.processed_data_dir = Path("../processed_data")
        self.converted_data_dir = Path("../converted_data")

        # キャッシュ
        self.video_mapping = {}  # content_id -> video_info
        self.user_mapping = {}   # reviewer_id -> user_info
        self.matched_reviews = []

        # 統計
        self.stats = {
            'total_users': 0,
            'total_videos': 0,
            'total_decisions': 0,
            'like_decisions': 0,
            'nope_decisions': 0,
            'processing_time': datetime.now().isoformat()
        }

    def load_created_users(self) -> List[Dict[str, Any]]:
        """作成済みユーザー情報を読み込み"""
        users_file = self.converted_data_dir / "created_users.json"

        if not users_file.exists():
            print("作成済みユーザーファイルが見つかりません")
            return []

        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)

        # reviewer_id -> user_info のマッピング
        for user in users:
            self.user_mapping[user['reviewer_id']] = {
                'user_id': user['user_id'],
                'display_name': user['display_name'],
                'email': user['email'],
                'created_at': user['created_at']
            }

        self.stats['total_users'] = len(users)
        print(f"ユーザー読み込み: {len(users)}件")
        return users

    async def fetch_matching_videos(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """マッチする動画データを取得"""
        headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}'
        }

        # 全動画データを取得してマッピング作成
        video_mapping = {}
        limit = 1000
        offset = 0

        print("動画データ取得中...")

        while True:
            videos_url = f"{self.supabase_url}/rest/v1/videos"
            params = {
                'select': 'id,external_id,title,thumbnail_url,price,product_released_at',
                'limit': limit,
                'offset': offset
            }

            try:
                async with session.get(videos_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        videos = await response.json()
                        if not videos:
                            break

                        for video in videos:
                            external_id = video.get('external_id')
                            if external_id:
                                video_mapping[external_id] = video

                        offset += limit

                        if len(videos) < limit:
                            break
                    else:
                        print(f"動画取得失敗: {response.status}")
                        break

            except Exception as e:
                print(f"動画取得エラー: {e}")
                break

        self.video_mapping = video_mapping
        self.stats['total_videos'] = len(video_mapping)
        print(f"動画データ取得完了: {len(video_mapping)}件")

        return video_mapping

    def process_matching_reviews(self) -> List[Dict[str, Any]]:
        """マッチするレビューを処理してuser_video_decisions形式に変換"""
        reviews_file = self.processed_data_dir / "integrated_reviews.json"

        with open(reviews_file, 'r', encoding='utf-8') as f:
            all_reviews = json.load(f)

        decisions = []
        matched_videos = set()

        print("レビューデータ処理中...")

        for review in all_reviews:
            reviewer_id = review.get('reviewer_id')
            content_id = review.get('content_id')
            rating = review.get('rating', 0)
            write_date = review.get('write_date')

            # ユーザーマッチング確認
            if reviewer_id not in self.user_mapping:
                continue

            # 動画マッチング確認
            if content_id not in self.video_mapping:
                continue

            matched_videos.add(content_id)
            user_info = self.user_mapping[reviewer_id]
            video_info = self.video_mapping[content_id]

            # 判定タイプ決定
            if rating >= 4:
                decision_type = 'like'
                self.stats['like_decisions'] += 1
            elif rating <= 3:
                decision_type = 'nope'
                self.stats['nope_decisions'] += 1
            else:
                continue

            # 日時変換
            created_at = write_date
            if isinstance(write_date, str) and 'T' not in write_date:
                try:
                    dt = datetime.strptime(write_date, "%Y-%m-%d %H:%M:%S")
                    created_at = dt.isoformat()
                except:
                    created_at = datetime.now().isoformat()

            decision = {
                'user_id': user_info['user_id'],
                'video_id': video_info['id'],
                'decision_type': decision_type,
                'created_at': created_at,
                # 追加メタデータ
                'original_rating': rating,
                'original_reviewer_id': reviewer_id,
                'original_content_id': content_id
            }

            decisions.append(decision)

        self.stats['total_decisions'] = len(decisions)
        self.matched_reviews = decisions

        print(f"マッチング完了:")
        print(f"  判定データ: {len(decisions)}件")
        print(f"  Like判定: {self.stats['like_decisions']}件")
        print(f"  Nope判定: {self.stats['nope_decisions']}件")
        print(f"  マッチ動画: {len(matched_videos)}件")

        return decisions

    def generate_profiles_data(self, users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """profiles テーブル互換形式でデータ生成"""
        profiles = []

        for user in users:
            profile = {
                'user_id': user['user_id'],
                'display_name': user['display_name'],
                'created_at': user['created_at'],
                # 追加フィールド（必要に応じて）
                'original_reviewer_id': user.get('reviewer_id'),
                'email': user['email']
            }
            profiles.append(profile)

        return profiles

    def generate_videos_subset(self) -> List[Dict[str, Any]]:
        """マッチした動画のサブセットを生成"""
        matched_content_ids = {decision['original_content_id'] for decision in self.matched_reviews}

        videos_subset = []
        for content_id in matched_content_ids:
            if content_id in self.video_mapping:
                video = self.video_mapping[content_id].copy()
                video['original_content_id'] = content_id  # 参照用
                videos_subset.append(video)

        return videos_subset

    def save_json_files(self, users: List[Dict[str, Any]], decisions: List[Dict[str, Any]]):
        """JSON ファイルとして保存"""

        # 1. Profiles データ
        profiles = self.generate_profiles_data(users)
        profiles_file = self.output_dir / "profiles.json"
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

        # 2. User Video Decisions データ
        decisions_file = self.output_dir / "user_video_decisions.json"
        with open(decisions_file, 'w', encoding='utf-8') as f:
            json.dump(decisions, f, ensure_ascii=False, indent=2)

        # 3. Videos サブセット
        videos_subset = self.generate_videos_subset()
        videos_file = self.output_dir / "videos_subset.json"
        with open(videos_file, 'w', encoding='utf-8') as f:
            json.dump(videos_subset, f, ensure_ascii=False, indent=2)

        # 4. 統計情報
        stats_file = self.output_dir / "data_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        print(f"\\nJSON ファイル保存完了:")
        print(f"  profiles: {profiles_file}")
        print(f"  decisions: {decisions_file}")
        print(f"  videos: {videos_file}")
        print(f"  stats: {stats_file}")

    def generate_sql_schema(self):
        """Supabase互換のテーブル構造定義SQLを生成"""
        schema_sql = '''-- Supabase互換テーブル構造定義
-- ローカル環境用

-- UUIDサポート
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- プロファイルテーブル
CREATE TABLE IF NOT EXISTS public.profiles (
    user_id UUID PRIMARY KEY,
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    original_reviewer_id TEXT,
    email TEXT
);

-- 動画テーブル（サブセット）
CREATE TABLE IF NOT EXISTS public.videos_subset (
    id UUID PRIMARY KEY,
    external_id TEXT UNIQUE,
    title TEXT,
    thumbnail_url TEXT,
    price NUMERIC,
    product_released_at TIMESTAMPTZ,
    original_content_id TEXT
);

-- ユーザー動画判定テーブル
CREATE TABLE IF NOT EXISTS public.user_video_decisions (
    user_id UUID REFERENCES public.profiles(user_id),
    video_id UUID REFERENCES public.videos_subset(id),
    decision_type TEXT NOT NULL CHECK (decision_type IN ('like', 'nope')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    original_rating NUMERIC,
    original_reviewer_id TEXT,
    original_content_id TEXT,
    PRIMARY KEY (user_id, video_id)
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_uvd_user_decision ON public.user_video_decisions(user_id, decision_type);
CREATE INDEX IF NOT EXISTS idx_uvd_created_at ON public.user_video_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_videos_external_id ON public.videos_subset(external_id);
'''
        schema_file = self.output_dir / "compatible_schema.sql"
        with open(schema_file, 'w', encoding='utf-8') as f:
            f.write(schema_sql)

        print(f"スキーマファイル生成: {schema_file}")

    def generate_import_sql(self, users: List[Dict[str, Any]], decisions: List[Dict[str, Any]]):
        """データインポート用SQLファイルを生成"""

        import_sql = "-- データインポート用SQL\\n\\n"

        # Profiles INSERT
        import_sql += "-- Profiles データ\\n"
        profiles = self.generate_profiles_data(users)

        if profiles:
            import_sql += "INSERT INTO public.profiles (user_id, display_name, created_at, original_reviewer_id, email) VALUES\\n"
            profile_values = []
            for profile in profiles:
                values = f"('{profile['user_id']}', '{profile['display_name']}', '{profile['created_at']}', '{profile.get('original_reviewer_id', '')}', '{profile.get('email', '')}')"
                profile_values.append(values)
            import_sql += ",\\n".join(profile_values) + ";\\n\\n"

        # Videos INSERT
        import_sql += "-- Videos サブセット\\n"
        videos_subset = self.generate_videos_subset()

        if videos_subset:
            import_sql += "INSERT INTO public.videos_subset (id, external_id, title, thumbnail_url, price, product_released_at, original_content_id) VALUES\\n"
            video_values = []
            for video in videos_subset[:100]:  # 最初の100件のみ（例）
                title = video.get('title', '').replace("'", "''")  # SQLエスケープ
                price = video.get('price', 'NULL')
                released_at = f"'{video.get('product_released_at')}'" if video.get('product_released_at') else 'NULL'
                values = f"('{video['id']}', '{video.get('external_id', '')}', '{title}', '{video.get('thumbnail_url', '')}', {price}, {released_at}, '{video.get('original_content_id', '')}')"
                video_values.append(values)
            import_sql += ",\\n".join(video_values) + ";\\n\\n"

        # Decisions INSERT (サンプル)
        import_sql += "-- User Video Decisions （サンプル100件）\\n"
        if decisions:
            import_sql += "INSERT INTO public.user_video_decisions (user_id, video_id, decision_type, created_at, original_rating, original_reviewer_id, original_content_id) VALUES\\n"
            decision_values = []
            for decision in decisions[:100]:  # サンプル100件
                values = f"('{decision['user_id']}', '{decision['video_id']}', '{decision['decision_type']}', '{decision['created_at']}', {decision.get('original_rating', 'NULL')}, '{decision.get('original_reviewer_id', '')}', '{decision.get('original_content_id', '')}')"
                decision_values.append(values)
            import_sql += ",\\n".join(decision_values) + ";\\n\\n"

        import_sql += "-- データインポート完了\\n"

        import_file = self.output_dir / "import_data.sql"
        with open(import_file, 'w', encoding='utf-8') as f:
            f.write(import_sql)

        print(f"インポートSQLファイル生成: {import_file}")

    async def build_complete_dataset(self):
        """完全なデータセットを構築"""
        print("=== Supabase互換ローカルデータ構築開始 ===")

        # 1. ユーザーデータ読み込み
        print("\\n1. ユーザーデータ読み込み中...")
        users = self.load_created_users()

        if not users:
            print("ユーザーデータが見つかりません")
            return

        # 2. 動画データ取得
        print("\\n2. 動画データ取得中...")
        async with aiohttp.ClientSession() as session:
            await self.fetch_matching_videos(session)

        # 3. マッチングレビュー処理
        print("\\n3. レビューマッチング処理中...")
        decisions = self.process_matching_reviews()

        if not decisions:
            print("マッチするレビューデータがありません")
            return

        # 4. JSON ファイル生成
        print("\\n4. JSON ファイル生成中...")
        self.save_json_files(users, decisions)

        # 5. SQL ファイル生成
        print("\\n5. SQL ファイル生成中...")
        self.generate_sql_schema()
        self.generate_import_sql(users, decisions)

        # 6. サマリー表示
        print("\\n=== 構築完了 ===")
        print(f"ユーザー数: {self.stats['total_users']}")
        print(f"動画数: {self.stats['total_videos']}")
        print(f"判定データ数: {self.stats['total_decisions']}")
        print(f"Like判定: {self.stats['like_decisions']}")
        print(f"Nope判定: {self.stats['nope_decisions']}")
        print(f"\\n出力ディレクトリ: {self.output_dir}")

async def main():
    builder = LocalCompatibleDataBuilder()
    await builder.build_complete_dataset()

if __name__ == "__main__":
    asyncio.run(main())