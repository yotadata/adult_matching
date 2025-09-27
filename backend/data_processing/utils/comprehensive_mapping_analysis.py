"""
レビューデータと動画データベースの包括的マッピング分析
"""

import asyncio
import aiohttp
import json
import os
from pathlib import Path
from collections import Counter

async def comprehensive_analysis():
    supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
    supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

    print("=== 包括的マッピング分析 ===")

    # 1. レビューデータの全content_id取得
    processed_data_dir = Path("../processed_data")
    reviews_file = processed_data_dir / "integrated_reviews.json"

    with open(reviews_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    review_content_ids = set()
    for review in reviews:
        content_id = review.get('content_id')
        if content_id:
            review_content_ids.add(content_id)

    print(f"1. レビューデータ")
    print(f"   総レビュー数: {len(reviews):,}")
    print(f"   ユニークcontent_id数: {len(review_content_ids):,}")

    # content_idの形式パターン分析
    patterns = Counter()
    for content_id in list(review_content_ids)[:1000]:  # サンプル
        if content_id:
            # 文字数による分類
            length = len(content_id)
            patterns[f"文字数{length}"] += 1

            # 数字の有無
            has_numbers = any(c.isdigit() for c in content_id)
            patterns[f"数字{'有' if has_numbers else '無'}"] += 1

    print(f"   content_id形式パターン (サンプル1000件):")
    for pattern, count in patterns.most_common(10):
        print(f"     {pattern}: {count}")

    print(f"   content_idサンプル: {list(review_content_ids)[:15]}")

    # 2. データベースから全動画データ取得
    async with aiohttp.ClientSession() as session:
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}'
        }

        # 全動画数を取得
        count_url = f"{supabase_url}/rest/v1/videos"
        count_params = {'select': 'count'}

        try:
            async with session.head(count_url, headers=headers) as response:
                total_videos = int(response.headers.get('Content-Range', '0').split('/')[-1])

        except:
            total_videos = 0

        print(f"\\n2. データベース動画データ")
        print(f"   総動画数: {total_videos:,}")

        # 全external_idを取得（ページネーション）
        db_external_ids = set()
        limit = 1000
        offset = 0

        while True:
            videos_url = f"{supabase_url}/rest/v1/videos"
            params = {
                'select': 'external_id',
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
                                db_external_ids.add(external_id)

                        offset += limit

                        if len(videos) < limit:
                            break

                    else:
                        print(f"API取得失敗: {response.status}")
                        break

            except Exception as e:
                print(f"エラー: {e}")
                break

        print(f"   external_id有り動画数: {len(db_external_ids):,}")

        # external_idの形式パターン分析
        db_patterns = Counter()
        for external_id in list(db_external_ids)[:1000]:  # サンプル
            if external_id:
                length = len(external_id)
                db_patterns[f"文字数{length}"] += 1

                has_numbers = any(c.isdigit() for c in external_id)
                db_patterns[f"数字{'有' if has_numbers else '無'}"] += 1

        print(f"   external_id形式パターン (サンプル1000件):")
        for pattern, count in db_patterns.most_common(10):
            print(f"     {pattern}: {count}")

        print(f"   external_idサンプル: {list(db_external_ids)[:15]}")

        # 3. マッチング分析
        print(f"\\n3. マッチング分析")
        exact_matches = review_content_ids.intersection(db_external_ids)
        print(f"   完全一致: {len(exact_matches):,}件")

        if exact_matches:
            print(f"   一致例: {list(exact_matches)[:10]}")

        # 部分一致分析（先頭部分）
        partial_matches = 0
        for review_id in list(review_content_ids)[:100]:  # サンプル
            for db_id in list(db_external_ids)[:100]:  # サンプル
                if review_id in db_id or db_id in review_id:
                    partial_matches += 1
                    break

        print(f"   部分一致（サンプル100x100）: {partial_matches}件")

        # 4. 統計サマリー
        print(f"\\n4. 統計サマリー")
        review_coverage = len(exact_matches) / len(review_content_ids) * 100 if review_content_ids else 0
        db_coverage = len(exact_matches) / len(db_external_ids) * 100 if db_external_ids else 0

        print(f"   レビューデータカバー率: {review_coverage:.2f}%")
        print(f"   DBデータカバー率: {db_coverage:.2f}%")

        # 5. 推奨対応策
        print(f"\\n5. 推奨対応策")
        if len(exact_matches) == 0:
            print("   ❌ 完全一致なし - データソースが異なる可能性")
            print("   📋 対策案:")
            print("     1. content_idの形式変換ルール調査")
            print("     2. 別のマッピングキー使用（タイトル等）")
            print("     3. DMM APIでの動画データ再同期")
        elif len(exact_matches) < len(review_content_ids) * 0.5:
            print("   ⚠️  一致率低い - 部分的データ不足")
        else:
            print("   ✅ 一致率良好 - 処理可能")

if __name__ == "__main__":
    asyncio.run(comprehensive_analysis())