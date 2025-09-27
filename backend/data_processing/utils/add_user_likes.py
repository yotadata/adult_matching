"""
作成済みユーザーにレビューデータをlike判定として追加するスクリプト

手順:
1. 作成済みユーザー情報を読み込み
2. Supabase REST APIで動画データを取得
3. レビューデータを評価に基づいてlike/nope判定に変換
4. user_video_decisionsテーブルに挿入
"""

import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class UserLikesAdder:
    def __init__(self, supabase_url: str = None, supabase_anon_key: str = None):
        self.supabase_url = supabase_url or os.getenv('NEXT_PUBLIC_SUPABASE_URL')
        self.supabase_anon_key = supabase_anon_key or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

        # データディレクトリ
        self.processed_data_dir = Path("../processed_data")
        self.converted_data_dir = Path("../converted_data")

        # 統計情報
        self.stats = {
            'loaded_users': 0,
            'loaded_videos': 0,
            'total_reviews': 0,
            'matched_videos': 0,
            'unmatched_videos': 0,
            'like_decisions': 0,
            'nope_decisions': 0,
            'inserted_decisions': 0,
            'failed_inserts': 0
        }

        # キャッシュ
        self.video_mapping = {}
        self.user_mapping = {}

    def load_created_users(self) -> List[Dict[str, Any]]:
        """作成済みユーザー情報を読み込み"""
        users_file = self.converted_data_dir / "created_users.json"

        if not users_file.exists():
            print(f"作成済みユーザーファイルが見つかりません: {users_file}")
            return []

        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)

        # reviewer_id -> user_id のマッピングを作成
        for user in users:
            self.user_mapping[user['reviewer_id']] = user['user_id']

        self.stats['loaded_users'] = len(users)
        print(f"作成済みユーザー読み込み: {len(users)}件")

        return users

    def load_integrated_reviews(self) -> List[Dict[str, Any]]:
        """統合済みレビューデータを読み込み"""
        integrated_file = self.processed_data_dir / "integrated_reviews.json"

        if not integrated_file.exists():
            print(f"統合レビューファイルが見つかりません: {integrated_file}")
            return []

        with open(integrated_file, 'r', encoding='utf-8') as f:
            reviews = json.load(f)

        self.stats['total_reviews'] = len(reviews)
        print(f"レビューデータ読み込み: {len(reviews)}件")

        return reviews

    async def fetch_videos_from_api(self, session: aiohttp.ClientSession) -> Dict[str, str]:
        """
        Supabase REST APIから動画データを取得してマッピングを作成
        """
        videos_url = f"{self.supabase_url}/rest/v1/videos"

        headers = {
            'apikey': self.supabase_anon_key,
            'Authorization': f'Bearer {self.supabase_anon_key}',
            'Content-Type': 'application/json'
        }

        params = {
            'select': 'id,external_id',
            'external_id': 'not.is.null'
        }

        try:
            async with session.get(videos_url, headers=headers, params=params) as response:
                if response.status == 200:
                    videos = await response.json()

                    video_mapping = {}
                    for video in videos:
                        if video.get('external_id'):
                            video_mapping[video['external_id']] = video['id']

                    self.video_mapping = video_mapping
                    self.stats['loaded_videos'] = len(video_mapping)

                    print(f"API経由で動画データ取得: {len(video_mapping)}件")
                    return video_mapping

                else:
                    error_text = await response.text()
                    print(f"動画データ取得失敗 ({response.status}): {error_text}")

        except Exception as e:
            print(f"動画データ取得例外: {e}")

        return {}

    def create_decisions_from_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        レビューデータからuser_video_decisions用のデータを作成
        """
        decisions = []
        matched = 0
        unmatched_video = 0
        unmatched_user = 0

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
                unmatched_user += 1
                continue

            # 動画IDを取得
            video_id = self.video_mapping.get(content_id)
            if not video_id:
                unmatched_video += 1
                continue

            matched += 1

            # 評価に基づいてdecision_typeを決定
            decision_type = None
            if rating >= 4:
                decision_type = 'like'
                self.stats['like_decisions'] += 1
            elif rating <= 3:  # 評価3以下をnopeに変更
                decision_type = 'nope'
                self.stats['nope_decisions'] += 1
            else:
                continue  # この条件は実際には到達しない

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
                'video_id': video_id,
                'decision_type': decision_type,
                'created_at': created_at
            }

            decisions.append(decision)

        self.stats['matched_videos'] = matched
        self.stats['unmatched_videos'] = unmatched_video + unmatched_user

        print(f"\\n判定データ生成:")
        print(f"  マッチ成功: {matched}件")
        print(f"  動画未マッチ: {unmatched_video}件")
        print(f"  ユーザー未マッチ: {unmatched_user}件")
        print(f"  Like判定: {self.stats['like_decisions']}件")
        print(f"  Nope判定: {self.stats['nope_decisions']}件")

        return decisions

    async def insert_decisions_batch(self, session: aiohttp.ClientSession, decisions: List[Dict[str, Any]], batch_size: int = 100):
        """
        判定データをバッチでuser_video_decisionsテーブルに挿入
        """
        decisions_url = f"{self.supabase_url}/rest/v1/user_video_decisions"

        headers = {
            'apikey': self.supabase_anon_key,
            'Authorization': f'Bearer {self.supabase_anon_key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }

        print(f"\\n=== 判定データ挿入開始 (バッチサイズ: {batch_size}) ===")

        for i in range(0, len(decisions), batch_size):
            batch = decisions[i:i + batch_size]

            print(f"バッチ {i//batch_size + 1}/{(len(decisions) + batch_size - 1)//batch_size} 挿入中... ({len(batch)}件)")

            try:
                async with session.post(decisions_url, headers=headers, json=batch) as response:
                    if response.status in [200, 201]:
                        self.stats['inserted_decisions'] += len(batch)
                        print(f"  ✓ バッチ挿入成功: {len(batch)}件")
                    else:
                        error_text = await response.text()
                        print(f"  ✗ バッチ挿入失敗 ({response.status}): {error_text}")
                        self.stats['failed_inserts'] += len(batch)

            except Exception as e:
                print(f"  ✗ バッチ挿入例外: {e}")
                self.stats['failed_inserts'] += len(batch)

            # レート制限対策
            if i + batch_size < len(decisions):
                await asyncio.sleep(1)

    def save_results(self, decisions: List[Dict[str, Any]]):
        """
        結果をファイルに保存
        """
        # 判定データ保存
        decisions_file = self.converted_data_dir / "final_user_video_decisions.json"
        with open(decisions_file, 'w', encoding='utf-8') as f:
            json.dump(decisions, f, ensure_ascii=False, indent=2)

        # 統計情報保存
        stats_file = self.converted_data_dir / "likes_insertion_stats.json"
        self.stats['processing_end'] = datetime.now().isoformat()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        print(f"\\n=== 結果保存完了 ===")
        print(f"判定データ: {decisions_file}")
        print(f"統計情報: {stats_file}")

    async def run_full_process(self):
        """
        フルプロセス実行
        """
        print("=== レビューデータをlike判定として追加開始 ===")

        # 1. 作成済みユーザー読み込み
        print("\\n1. 作成済みユーザー読み込み中...")
        users = self.load_created_users()
        if not users:
            print("作成済みユーザーが見つかりません")
            return

        # 2. レビューデータ読み込み
        print("\\n2. レビューデータ読み込み中...")
        reviews = self.load_integrated_reviews()
        if not reviews:
            print("レビューデータが見つかりません")
            return

        # 3. 動画データ取得
        print("\\n3. 動画データ取得中...")
        async with aiohttp.ClientSession() as session:
            video_mapping = await self.fetch_videos_from_api(session)

            if not video_mapping:
                print("動画データの取得に失敗しました")
                return

            # 4. 判定データ生成
            print("\\n4. 判定データ生成中...")
            decisions = self.create_decisions_from_reviews(reviews)

            if not decisions:
                print("生成された判定データがありません")
                return

            # 5. データベース挿入
            print("\\n5. データベース挿入中...")
            await self.insert_decisions_batch(session, decisions)

        # 6. 結果保存
        print("\\n6. 結果保存中...")
        self.save_results(decisions)

        # 7. 統計表示
        print("\\n=== 処理完了 ===")
        print(f"読み込みユーザー: {self.stats['loaded_users']}")
        print(f"読み込み動画: {self.stats['loaded_videos']}")
        print(f"処理レビュー: {self.stats['total_reviews']}")
        print(f"Like判定: {self.stats['like_decisions']}")
        print(f"Nope判定: {self.stats['nope_decisions']}")
        print(f"挿入成功: {self.stats['inserted_decisions']}")
        print(f"挿入失敗: {self.stats['failed_inserts']}")

async def main():
    """
    メイン実行関数
    """
    adder = UserLikesAdder()
    await adder.run_full_process()

if __name__ == "__main__":
    asyncio.run(main())