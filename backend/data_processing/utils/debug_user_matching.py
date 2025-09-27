"""
ユーザーマッチングのデバッグスクリプト
"""

import json
from pathlib import Path

def debug_user_matching():
    processed_data_dir = Path("../processed_data")
    converted_data_dir = Path("../converted_data")

    # 作成済みユーザー読み込み
    users_file = converted_data_dir / "created_users.json"
    with open(users_file, 'r', encoding='utf-8') as f:
        users = json.load(f)

    # レビューデータ読み込み
    reviews_file = processed_data_dir / "integrated_reviews.json"
    with open(reviews_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    print("=== デバッグ情報 ===")
    print(f"作成済みユーザー: {len(users)}件")
    print(f"レビューデータ: {len(reviews)}件")

    # 作成されたユーザーのレビュワーID
    user_reviewer_ids = {user['reviewer_id'] for user in users}
    print(f"\\n作成されたユーザーのレビュワーID: {user_reviewer_ids}")

    # レビューデータのレビュワーID（ユニーク）
    review_reviewer_ids = {review.get('reviewer_id') for review in reviews if review.get('reviewer_id')}
    print(f"\\nレビューデータのユニークレビュワーID数: {len(review_reviewer_ids)}")
    print(f"最初の10件: {list(review_reviewer_ids)[:10]}")

    # マッチングチェック
    matched_ids = user_reviewer_ids.intersection(review_reviewer_ids)
    print(f"\\nマッチするレビュワーID: {matched_ids}")

    # 各ユーザーのレビュー数をカウント
    for user_id in user_reviewer_ids:
        count = sum(1 for review in reviews if review.get('reviewer_id') == user_id)
        print(f"レビュワー {user_id}: {count}件のレビュー")

        if count > 0:
            # サンプルレビューを表示
            sample_reviews = [review for review in reviews if review.get('reviewer_id') == user_id][:3]
            for i, review in enumerate(sample_reviews):
                print(f"  サンプル{i+1}: content_id={review.get('content_id')}, rating={review.get('rating')}")

if __name__ == "__main__":
    debug_user_matching()