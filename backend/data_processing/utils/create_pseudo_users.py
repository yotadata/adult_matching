"""
Supabase認証システムで疑似ユーザーを作成し、レビューデータを紐付けるスクリプト

手順:
1. レビューデータから一意のレビュワーIDを抽出
2. 各レビュワーにメールアドレスを生成
3. Supabase Auth APIでユーザー登録
4. 生成されたuser_idでprofilesとuser_video_decisionsを作成
"""

import json
import asyncio
import aiohttp
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class PseudoUserCreator:
    def __init__(self, supabase_url: str = None, supabase_anon_key: str = None, supabase_service_key: str = None):
        self.supabase_url = supabase_url or os.getenv('NEXT_PUBLIC_SUPABASE_URL', 'http://127.0.0.1:54321')
        self.supabase_anon_key = supabase_anon_key or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
        self.supabase_service_key = supabase_service_key or os.getenv('SUPABASE_SERVICE_ROLE_KEY')

        # データディレクトリ
        self.processed_data_dir = Path("../processed_data")
        self.output_dir = Path("../converted_data")
        self.output_dir.mkdir(exist_ok=True)

        # 統計情報
        self.stats = {
            'total_reviewers': 0,
            'successful_registrations': 0,
            'failed_registrations': 0,
            'created_profiles': 0,
            'created_decisions': 0,
            'processing_start': datetime.now().isoformat()
        }

        # ユーザーマッピング（reviewer_id -> user_id）
        self.user_mapping = {}

    def load_integrated_reviews(self) -> List[Dict[str, Any]]:
        """統合済みレビューデータを読み込み"""
        integrated_file = self.processed_data_dir / "integrated_reviews.json"

        if not integrated_file.exists():
            print(f"統合レビューファイルが見つかりません: {integrated_file}")
            return []

        with open(integrated_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_unique_reviewers(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """レビューデータから一意のレビュワーIDを抽出"""
        reviewer_ids = set()

        for review in reviews:
            reviewer_id = review.get('reviewer_id')
            if reviewer_id:
                reviewer_ids.add(reviewer_id)

        self.stats['total_reviewers'] = len(reviewer_ids)
        print(f"一意のレビュワー数: {len(reviewer_ids)}")

        return list(reviewer_ids)

    def generate_user_credentials(self, reviewer_ids: List[str]) -> List[Dict[str, str]]:
        """各レビュワーにメールアドレスとパスワードを生成"""
        credentials = []

        for reviewer_id in reviewer_ids:
            email = f"reviewer_{reviewer_id}@pseudo.local"
            password = f"pseudo_pass_{reviewer_id}_2025"  # 簡単なパスワード生成
            display_name = f"レビュワー_{reviewer_id}"

            credentials.append({
                'reviewer_id': reviewer_id,
                'email': email,
                'password': password,
                'display_name': display_name
            })

        return credentials

    async def create_single_user(self, session: aiohttp.ClientSession, credential: Dict[str, str]) -> Optional[str]:
        """
        個別ユーザーを作成
        """
        try:
            # Supabase Auth API - ユーザー登録
            auth_url = f"{self.supabase_url}/auth/v1/signup"

            headers = {
                'apikey': self.supabase_anon_key,
                'Content-Type': 'application/json'
            }

            payload = {
                'email': credential['email'],
                'password': credential['password']
            }

            async with session.post(auth_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    user_id = data.get('user', {}).get('id')

                    if user_id:
                        # プロファイル作成
                        await self.create_user_profile(session, user_id, credential['display_name'])

                        self.user_mapping[credential['reviewer_id']] = user_id
                        self.stats['successful_registrations'] += 1

                        print(f"✓ ユーザー作成成功: {credential['email']} -> {user_id}")
                        return user_id
                    else:
                        print(f"✗ ユーザーID取得失敗: {credential['email']}")
                        self.stats['failed_registrations'] += 1

                else:
                    error_text = await response.text()
                    print(f"✗ ユーザー作成失敗 ({response.status}): {credential['email']} - {error_text}")
                    self.stats['failed_registrations'] += 1

        except Exception as e:
            print(f"✗ ユーザー作成例外: {credential['email']} - {e}")
            self.stats['failed_registrations'] += 1

        return None

    async def create_user_profile(self, session: aiohttp.ClientSession, user_id: str, display_name: str):
        """
        ユーザープロファイルを作成
        """
        try:
            profile_url = f"{self.supabase_url}/rest/v1/profiles"

            headers = {
                'apikey': self.supabase_anon_key,
                'Authorization': f'Bearer {self.supabase_anon_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                'user_id': user_id,
                'display_name': display_name,
                'created_at': datetime.now().isoformat()
            }

            async with session.post(profile_url, headers=headers, json=payload) as response:
                if response.status in [200, 201]:
                    self.stats['created_profiles'] += 1
                    print(f"  ✓ プロファイル作成: {display_name}")
                else:
                    error_text = await response.text()
                    print(f"  ✗ プロファイル作成失敗 ({response.status}): {error_text}")

        except Exception as e:
            print(f"  ✗ プロファイル作成例外: {e}")

    async def create_users_batch(self, credentials: List[Dict[str, str]], batch_size: int = 5):
        """
        バッチでユーザーを作成（レート制限対策）
        """
        print(f"\\n=== ユーザー登録開始 (バッチサイズ: {batch_size}) ===")

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(credentials), batch_size):
                batch = credentials[i:i + batch_size]

                print(f"\\nバッチ {i//batch_size + 1}/{(len(credentials) + batch_size - 1)//batch_size} 処理中...")

                # 並行でユーザー作成
                tasks = [self.create_single_user(session, cred) for cred in batch]
                await asyncio.gather(*tasks)

                # レート制限対策のための待機
                if i + batch_size < len(credentials):
                    await asyncio.sleep(2)  # 2秒待機

    def get_video_mapping(self) -> Dict[str, str]:
        """
        データベースからvideosテーブルの external_id -> id マッピングを取得
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor

        video_mapping = {}
        db_url = os.getenv('DATABASE_URL')

        if not db_url:
            print("DATABASE_URL環境変数が設定されていません")
            return video_mapping

        try:
            with psycopg2.connect(db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT id, external_id FROM public.videos WHERE external_id IS NOT NULL")

                    for row in cur.fetchall():
                        video_mapping[row['external_id']] = str(row['id'])

            print(f"データベースから{len(video_mapping)}件の動画マッピングを取得")

        except Exception as e:
            print(f"データベース接続エラー: {e}")

        return video_mapping

    def create_video_decisions_data(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        レビューデータからuser_video_decisions用のデータを作成
        """
        decisions = []
        matched = 0
        unmatched = 0

        # 動画マッピングを取得
        video_mapping = self.get_video_mapping()
        print(f"利用可能な動画マッピング: {len(video_mapping)}件")

        for review in reviews:
            reviewer_id = review.get('reviewer_id')
            content_id = review.get('content_id')
            rating = review.get('rating', 0)
            write_date = review.get('write_date')

            if not reviewer_id or not content_id:
                continue

            # ユーザーIDを取得
            user_id = self.user_mapping.get(reviewer_id)
            if not user_id:
                unmatched += 1
                continue

            # 動画IDを取得
            video_id = video_mapping.get(content_id)
            if not video_id:
                unmatched += 1
                continue

            matched += 1

            # 評価に基づいてdecision_typeを決定
            decision_type = None
            if rating >= 4:
                decision_type = 'like'
            elif rating <= 2:
                decision_type = 'nope'
            else:
                continue  # 中間評価は除外

            # created_atを変換
            created_at = write_date
            if isinstance(write_date, str):
                try:
                    if 'T' not in write_date:
                        dt = datetime.strptime(write_date, "%Y-%m-%d %H:%M:%S")
                        created_at = dt.isoformat()
                except:
                    created_at = datetime.now().isoformat()

            decision = {
                'user_id': user_id,
                'video_id': video_id,  # 実際のvideo_id
                'decision_type': decision_type,
                'created_at': created_at,
                'original_rating': rating,
                'original_reviewer_id': reviewer_id,
                'original_content_id': content_id
            }

            decisions.append(decision)

        self.stats['matched_reviews'] = matched
        self.stats['unmatched_reviews'] = unmatched
        self.stats['created_decisions'] = len(decisions)

        print(f"\\n判定データ生成: {len(decisions)}件 (マッチ: {matched}, 非マッチ: {unmatched})")

        return decisions

    def save_results(self, credentials: List[Dict[str, str]], decisions: List[Dict[str, Any]]):
        """
        結果をファイルに保存
        """
        # ユーザー情報保存
        users_file = self.output_dir / "created_users.json"
        user_data = []

        for cred in credentials:
            user_id = self.user_mapping.get(cred['reviewer_id'])
            if user_id:
                user_data.append({
                    'user_id': user_id,
                    'reviewer_id': cred['reviewer_id'],
                    'email': cred['email'],
                    'display_name': cred['display_name'],
                    'created_at': datetime.now().isoformat()
                })

        with open(users_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)

        # 判定データ保存
        decisions_file = self.output_dir / "user_video_decisions_with_users.json"
        with open(decisions_file, 'w', encoding='utf-8') as f:
            json.dump(decisions, f, ensure_ascii=False, indent=2)

        # 統計情報保存
        stats_file = self.output_dir / "user_creation_stats.json"
        self.stats['processing_end'] = datetime.now().isoformat()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        print(f"\\n=== 結果保存完了 ===")
        print(f"作成ユーザー: {users_file}")
        print(f"判定データ: {decisions_file}")
        print(f"統計情報: {stats_file}")

    async def run_full_process(self, max_users: int = 10):
        """
        フルプロセス実行（テスト用に最大ユーザー数制限）
        """
        print("=== 疑似ユーザー作成・データ紐付けプロセス開始 ===")

        # 1. レビューデータ読み込み
        print("\\n1. レビューデータ読み込み中...")
        reviews = self.load_integrated_reviews()
        if not reviews:
            print("レビューデータが見つかりません")
            return

        print(f"レビューデータ: {len(reviews)}件")

        # 2. レビュワー抽出
        print("\\n2. レビュワー抽出中...")
        reviewer_ids = self.extract_unique_reviewers(reviews)

        # テスト用に制限
        if max_users and len(reviewer_ids) > max_users:
            reviewer_ids = reviewer_ids[:max_users]
            print(f"テスト用に{max_users}人に制限")

        # 3. 認証情報生成
        print("\\n3. 認証情報生成中...")
        credentials = self.generate_user_credentials(reviewer_ids)

        # 4. ユーザー作成
        print("\\n4. Supabaseユーザー作成中...")
        await self.create_users_batch(credentials)

        # 5. 判定データ生成
        print("\\n5. 判定データ生成中...")
        decisions = self.create_video_decisions_data(reviews)

        # 6. 結果保存
        print("\\n6. 結果保存中...")
        self.save_results(credentials, decisions)

        # 7. 統計表示
        print("\\n=== 処理完了 ===")
        print(f"対象レビュワー: {self.stats['total_reviewers']}")
        print(f"成功登録: {self.stats['successful_registrations']}")
        print(f"失敗登録: {self.stats['failed_registrations']}")
        print(f"作成プロファイル: {self.stats['created_profiles']}")
        print(f"生成判定データ: {self.stats['created_decisions']}")

async def main():
    """
    メイン実行関数
    """
    # 環境変数チェック
    supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
    supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

    if not supabase_url or not supabase_key:
        print("環境変数が設定されていません:")
        print("NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY")
        print("\\nローカル開発環境のデフォルト値を使用します")

    creator = PseudoUserCreator()

    # テスト実行（最初は少数で）
    await creator.run_full_process(max_users=5)

if __name__ == "__main__":
    asyncio.run(main())