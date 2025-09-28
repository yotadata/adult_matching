#!/usr/bin/env python3
"""
50人の仮想ユーザーをローカルで作成

商用Supabaseに影響を与えずに、50人のレビュアー分の
仮想UUIDとプロファイルをローカルで生成する
"""

import json
import uuid
from datetime import datetime, timedelta
import random
from pathlib import Path
from typing import Dict, List, Any

class VirtualUserCreator:
    """50人の仮想ユーザーをローカル生成"""

    def __init__(self):
        self.processed_data_dir = Path("../archive/processed_data")
        self.output_dir = Path("../archive/converted_data")
        self.output_dir.mkdir(exist_ok=True)

    def extract_reviewer_ids(self) -> List[str]:
        """レビューデータから全レビュアーIDを抽出"""
        reviews_file = self.processed_data_dir / "integrated_reviews.json"

        with open(reviews_file, 'r', encoding='utf-8') as f:
            all_reviews = json.load(f)

        # 全レビュアーIDを抽出
        reviewer_ids = set()
        for review in all_reviews:
            reviewer_id = review.get('reviewer_id')
            if reviewer_id:
                reviewer_ids.add(reviewer_id)

        reviewer_list = sorted(list(reviewer_ids))
        print(f"レビュアー総数: {len(reviewer_list)}人")

        return reviewer_list

    def generate_virtual_users(self, reviewer_ids: List[str]) -> List[Dict[str, Any]]:
        """50人の仮想ユーザーを生成"""
        virtual_users = []

        # 基準日時（適度に過去）
        base_date = datetime.now() - timedelta(days=365)

        for i, reviewer_id in enumerate(reviewer_ids):
            # 仮想UUID生成（実際のSupabaseユーザーではない）
            virtual_uuid = str(uuid.uuid4())

            # 適度にランダムな作成日時
            created_at = base_date + timedelta(days=random.randint(0, 300))

            # 仮想プロファイル
            virtual_user = {
                'user_id': virtual_uuid,
                'reviewer_id': reviewer_id,
                'display_name': f'Virtual_User_{reviewer_id}',
                'email': f'virtual_{reviewer_id}@local.test',
                'created_at': created_at.isoformat(),
                'is_virtual': True,  # 仮想ユーザーフラグ
                'local_only': True   # ローカル専用フラグ
            }

            virtual_users.append(virtual_user)

        print(f"仮想ユーザー生成: {len(virtual_users)}人")
        return virtual_users

    def save_virtual_users(self, virtual_users: List[Dict[str, Any]]):
        """仮想ユーザーをローカルファイルに保存"""

        # created_users.json を上書き（既存5人→50人に拡張）
        output_file = self.output_dir / "created_users.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(virtual_users, f, ensure_ascii=False, indent=2)

        # 統計情報も生成
        stats = {
            'total_virtual_users': len(virtual_users),
            'creation_method': 'local_virtual_generation',
            'created_at': datetime.now().isoformat(),
            'note': 'これらは仮想UUIDです。実際のSupabaseユーザーではありません。'
        }

        stats_file = self.output_dir / "virtual_user_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"✅ 仮想ユーザー保存完了:")
        print(f"   ファイル: {output_file}")
        print(f"   統計: {stats_file}")
        print(f"   ユーザー数: {len(virtual_users)}人")

    def create_full_virtual_dataset(self):
        """フル仮想データセット作成の全工程"""
        print("🚀 50人仮想ユーザー作成開始...")

        # Step 1: レビュアーID抽出
        print("\n📊 Step 1: レビュアーID抽出")
        reviewer_ids = self.extract_reviewer_ids()

        # Step 2: 仮想ユーザー生成
        print("\n👥 Step 2: 仮想ユーザー生成")
        virtual_users = self.generate_virtual_users(reviewer_ids)

        # Step 3: 保存
        print("\n💾 Step 3: 仮想ユーザー保存")
        self.save_virtual_users(virtual_users)

        print(f"\n🎉 完了! {len(virtual_users)}人の仮想ユーザーが作成されました")
        print("次のステップ: build_local_compatible_data.py を実行してフルデータセットを構築")

        return virtual_users

if __name__ == "__main__":
    creator = VirtualUserCreator()
    virtual_users = creator.create_full_virtual_dataset()

    # レビュアー別レビュー数統計も表示
    with open(Path("../archive/processed_data/integrated_reviews.json"), 'r', encoding='utf-8') as f:
        all_reviews = json.load(f)

    reviewer_counts = {}
    for review in all_reviews:
        reviewer_id = review.get('reviewer_id')
        if reviewer_id:
            reviewer_counts[reviewer_id] = reviewer_counts.get(reviewer_id, 0) + 1

    print(f"\n📈 レビュー分布:")
    print(f"   平均レビュー数: {len(all_reviews) // len(virtual_users):.1f}件/人")
    print(f"   最大レビュー数: {max(reviewer_counts.values())}件")
    print(f"   最小レビュー数: {min(reviewer_counts.values())}件")