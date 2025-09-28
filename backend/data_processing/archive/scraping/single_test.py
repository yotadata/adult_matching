"""
単一レビュワー・単一ページテスト
"""

from batch_review_scraper import BatchReviewScraper
import json

def test_single_reviewer():
    print("=== 単一レビュワーテスト ===")
    
    scraper = BatchReviewScraper()
    
    # Cookie読み込み
    scraper.load_cookies_from_json("../config/dmm_cookies.json")
    
    # テスト対象: ID 185585 (エロの極み乙女。) の1ページ目
    test_reviewer_id = "185585"
    
    print(f"テスト対象: レビュワーID {test_reviewer_id}")
    
    # 1ページ目を取得
    soup = scraper.fetch_reviewer_page(test_reviewer_id, 1)
    if not soup:
        print("エラー: ページ取得失敗")
        return
    
    print("ページ取得成功")
    
    # レビューデータ抽出
    reviews = scraper.extract_reviews_from_page(soup, test_reviewer_id, 1)
    
    print(f"抽出されたレビュー数: {len(reviews)}")
    
    # 結果を表示
    if reviews:
        print("\n=== サンプルレビュー ===")
        for i, review in enumerate(reviews[:3]):  # 最初の3件を表示
            print(f"{i+1}. タイトル: {review.get('title', 'N/A')[:50]}")
            print(f"   レビューテキスト: {review.get('review_text', 'N/A')[:100]}...")
            print(f"   評価: {review.get('rating', 'N/A')}")
            print()
        
        # JSON保存
        with open("../raw_data/single_test.json", 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        
        print(f"テスト結果を保存: single_test.json")
    else:
        print("レビューデータが抽出されませんでした")
        
        # デバッグ用: ページの一部を表示
        print("\n=== ページ内容サンプル ===")
        print(soup.get_text()[:500])

if __name__ == "__main__":
    test_single_reviewer()