"""
動画データと レビューデータのcontent_idマッピングを確認
"""

import asyncio
import aiohttp
import json
import os
from pathlib import Path

async def check_video_mapping():
    supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
    supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

    # レビューデータのcontent_idサンプルを取得
    processed_data_dir = Path("../processed_data")
    reviews_file = processed_data_dir / "integrated_reviews.json"

    with open(reviews_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    # レビューデータのcontent_idサンプル
    review_content_ids = {review.get('content_id') for review in reviews[:100] if review.get('content_id')}
    print(f"レビューデータのcontent_idサンプル (最初の100件から): {list(review_content_ids)[:10]}")

    # Supabaseから動画データを取得
    async with aiohttp.ClientSession() as session:
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}'
        }

        videos_url = f"{supabase_url}/rest/v1/videos"
        params = {'select': 'external_id', 'limit': '50'}

        try:
            async with session.get(videos_url, headers=headers, params=params) as response:
                if response.status == 200:
                    videos = await response.json()
                    db_external_ids = {video.get('external_id') for video in videos if video.get('external_id')}
                    print(f"\\nDBの動画external_idサンプル: {list(db_external_ids)[:10]}")

                    # マッチング確認
                    matches = review_content_ids.intersection(db_external_ids)
                    print(f"\\nマッチするcontent_id: {matches}")
                    print(f"マッチ数: {len(matches)}")

                else:
                    print(f"API取得失敗: {response.status}")
                    print(await response.text())

        except Exception as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    asyncio.run(check_video_mapping())