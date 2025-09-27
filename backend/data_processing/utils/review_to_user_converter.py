"""
レビューデータをSupabaseユーザーテーブル形式に変換するスクリプト

レビューデータから：
1. 疑似ユーザープロファイル (profiles)
2. ユーザーの動画判定 (user_video_decisions)
3. 既存のvideosテーブルとの連携

を生成する
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class ReviewToUserConverter:
    def __init__(self, db_connection_string: Optional[str] = None):
        self.db_conn_str = db_connection_string or os.getenv('DATABASE_URL')

        # レビューデータのパス
        self.raw_data_dir = Path("../raw_data")
        self.processed_data_dir = Path("../processed_data")

        # 変換結果を保存するディレクトリ
        self.output_dir = Path("../converted_data")
        self.output_dir.mkdir(exist_ok=True)

        # 統計情報
        self.stats = {
            'total_reviewers': 0,
            'total_reviews': 0,
            'matched_videos': 0,
            'unmatched_videos': 0,
            'created_profiles': 0,
            'created_decisions': 0
        }

    def load_integrated_reviews(self) -> List[Dict[str, Any]]:
        """統合済みレビューデータを読み込み"""
        integrated_file = self.processed_data_dir / "integrated_reviews.json"

        if not integrated_file.exists():
            print(f"統合レビューファイルが見つかりません: {integrated_file}")
            return []

        with open(integrated_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_existing_videos(self) -> Dict[str, str]:
        """
        既存のvideosテーブルからexternal_idとuuidのマッピングを取得
        """
        if not self.db_conn_str:
            print("データベース接続情報がありません")
            return {}

        video_mapping = {}

        try:
            with psycopg2.connect(self.db_conn_str) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT id, external_id FROM public.videos WHERE external_id IS NOT NULL")

                    for row in cur.fetchall():
                        video_mapping[row['external_id']] = str(row['id'])

            print(f"データベースから{len(video_mapping)}件の動画情報を取得")

        except Exception as e:
            print(f"データベース接続エラー: {e}")

        return video_mapping

    def create_pseudo_user_profiles(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        レビューデータから疑似ユーザープロファイルを生成
        """
        profiles = []
        reviewer_ids = set()

        for review in reviews:
            reviewer_id = review.get('reviewer_id')
            if reviewer_id and reviewer_id not in reviewer_ids:
                reviewer_ids.add(reviewer_id)

                # UUIIDを生成（疑似ユーザー用）
                user_uuid = str(uuid.uuid4())

                profile = {
                    'user_id': user_uuid,
                    'display_name': f"レビュワー_{reviewer_id}",
                    'created_at': datetime.now().isoformat(),
                    'original_reviewer_id': reviewer_id  # 元のIDを保持
                }

                profiles.append(profile)

        self.stats['created_profiles'] = len(profiles)
        print(f"疑似ユーザープロファイル生成: {len(profiles)}件")

        return profiles

    def create_user_video_decisions(self,
                                   reviews: List[Dict[str, Any]],
                                   profiles: List[Dict[str, Any]],
                                   video_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        レビューデータからuser_video_decisions形式のデータを生成
        """
        decisions = []

        # reviewer_id -> user_uuid のマッピングを作成
        reviewer_to_uuid = {
            profile['original_reviewer_id']: profile['user_id']
            for profile in profiles
        }

        matched_videos = 0
        unmatched_videos = 0

        for review in reviews:
            reviewer_id = review.get('reviewer_id')
            content_id = review.get('content_id')
            rating = review.get('rating', 0)
            write_date = review.get('write_date')

            if not reviewer_id or not content_id:
                continue

            # ユーザーUUIDを取得
            user_uuid = reviewer_to_uuid.get(reviewer_id)
            if not user_uuid:
                continue

            # 動画UUIDを取得
            video_uuid = video_mapping.get(content_id)
            if not video_uuid:
                unmatched_videos += 1
                continue

            matched_videos += 1

            # 評価に基づいてdecision_typeを決定
            # 4-5: like, 1-2: nope, 3: 中間なので除外
            decision_type = None
            if rating >= 4:
                decision_type = 'like'
            elif rating <= 2:
                decision_type = 'nope'
            else:
                # 中間評価は除外
                continue

            # created_atを変換
            created_at = write_date
            if isinstance(write_date, str):
                try:
                    # 様々な日付フォーマットに対応
                    if 'T' in write_date:
                        created_at = write_date
                    else:
                        # "2025-09-05 15:13:48" 形式
                        dt = datetime.strptime(write_date, "%Y-%m-%d %H:%M:%S")
                        created_at = dt.isoformat()
                except:
                    created_at = datetime.now().isoformat()

            decision = {
                'user_id': user_uuid,
                'video_id': video_uuid,
                'decision_type': decision_type,
                'created_at': created_at,
                'original_rating': rating,  # 元の評価を保持
                'original_reviewer_id': reviewer_id,
                'original_content_id': content_id
            }

            decisions.append(decision)

        self.stats['matched_videos'] = matched_videos
        self.stats['unmatched_videos'] = unmatched_videos
        self.stats['created_decisions'] = len(decisions)

        print(f"動画マッチング: {matched_videos}件成功, {unmatched_videos}件失敗")
        print(f"ユーザー判定データ生成: {len(decisions)}件")

        return decisions

    def save_converted_data(self, profiles: List[Dict[str, Any]], decisions: List[Dict[str, Any]]):
        """
        変換済みデータをJSONファイルに保存
        """
        # プロファイル保存
        profiles_file = self.output_dir / "pseudo_user_profiles.json"
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

        # 判定データ保存
        decisions_file = self.output_dir / "user_video_decisions.json"
        with open(decisions_file, 'w', encoding='utf-8') as f:
            json.dump(decisions, f, ensure_ascii=False, indent=2)

        # 統計情報保存
        stats_file = self.output_dir / "conversion_stats.json"
        self.stats['conversion_time'] = datetime.now().isoformat()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        print(f"\n変換結果を保存:")
        print(f"  プロファイル: {profiles_file}")
        print(f"  判定データ: {decisions_file}")
        print(f"  統計情報: {stats_file}")

    def generate_sql_inserts(self, profiles: List[Dict[str, Any]], decisions: List[Dict[str, Any]]):
        """
        データベース挿入用のSQLファイルを生成
        """
        sql_file = self.output_dir / "insert_pseudo_users.sql"

        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write("-- レビューデータから生成された疑似ユーザーデータ\n")
            f.write("-- 注意: 実際のauth.usersテーブルに挿入するには認証システムとの統合が必要\n\n")

            # プロファイル挿入SQL（参考用）
            f.write("-- profiles テーブル用（auth.usersが必要）\n")
            f.write("/*\n")
            for profile in profiles[:5]:  # 最初の5件のみサンプル表示
                f.write(f"INSERT INTO public.profiles (user_id, display_name, created_at) VALUES\n")
                f.write(f"  ('{profile['user_id']}', '{profile['display_name']}', '{profile['created_at']}');\n")
            f.write("*/\n\n")

            # 判定データ挿入SQL
            f.write("-- user_video_decisions テーブル用\n")
            f.write("INSERT INTO public.user_video_decisions (user_id, video_id, decision_type, created_at) VALUES\n")

            sql_values = []
            for decision in decisions:
                sql_values.append(
                    f"('{decision['user_id']}', '{decision['video_id']}', "
                    f"'{decision['decision_type']}', '{decision['created_at']}')"
                )

            f.write(",\n".join(sql_values))
            f.write(";\n")

        print(f"SQL挿入ファイル生成: {sql_file}")

    def run_conversion(self):
        """
        メイン変換処理
        """
        print("=== レビューデータ → ユーザーテーブル形式変換開始 ===")

        # 1. レビューデータ読み込み
        print("\n1. レビューデータ読み込み中...")
        reviews = self.load_integrated_reviews()
        if not reviews:
            print("レビューデータが見つかりません")
            return

        self.stats['total_reviews'] = len(reviews)
        print(f"レビューデータ読み込み完了: {len(reviews)}件")

        # 2. 既存動画データ取得
        print("\n2. データベースから既存動画情報取得中...")
        video_mapping = self.get_existing_videos()

        # 3. 疑似ユーザープロファイル生成
        print("\n3. 疑似ユーザープロファイル生成中...")
        profiles = self.create_pseudo_user_profiles(reviews)

        # 4. ユーザー判定データ生成
        print("\n4. ユーザー動画判定データ生成中...")
        decisions = self.create_user_video_decisions(reviews, profiles, video_mapping)

        # 5. 結果保存
        print("\n5. 変換結果保存中...")
        self.save_converted_data(profiles, decisions)

        # 6. SQL生成
        print("\n6. SQL挿入ファイル生成中...")
        self.generate_sql_inserts(profiles, decisions)

        # 7. 統計表示
        print("\n=== 変換完了 ===")
        print(f"処理したレビュー数: {self.stats['total_reviews']}")
        print(f"生成プロファイル数: {self.stats['created_profiles']}")
        print(f"生成判定データ数: {self.stats['created_decisions']}")
        print(f"動画マッチング成功: {self.stats['matched_videos']}")
        print(f"動画マッチング失敗: {self.stats['unmatched_videos']}")

def main():
    # 環境変数からデータベース接続情報を取得
    # 実際の使用時は適切な接続文字列に変更
    db_conn = os.getenv('DATABASE_URL', None)

    converter = ReviewToUserConverter(db_conn)
    converter.run_conversion()

if __name__ == "__main__":
    main()