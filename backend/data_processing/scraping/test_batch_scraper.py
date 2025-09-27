"""
バッチレビュー収集テストスクリプト

最初の数人のレビュワーでテスト実行
"""

import json
from batch_review_scraper import BatchReviewScraper

def create_test_reviewers_file():
    """テスト用の小さなレビュワーリストを作成"""
    try:
        # 元のファイルを読み込み
        with open("../raw_data/top_reviewers.json", 'r', encoding='utf-8') as f:
            all_reviewers = json.load(f)
        
        # 最初の3人のみを選択
        test_reviewers = all_reviewers[:3]
        
        # テスト用ファイルに保存
        with open("../raw_data/test_reviewers.json", 'w', encoding='utf-8') as f:
            json.dump(test_reviewers, f, ensure_ascii=False, indent=2)
        
        print(f"テスト用レビュワーファイル作成: {len(test_reviewers)} 人")
        for reviewer in test_reviewers:
            print(f"  - {reviewer['reviewer_id']}: {reviewer['username'][:30]}...")
        
        return True
        
    except Exception as e:
        print(f"テストファイル作成エラー: {e}")
        return False

def main():
    print("=== バッチレビュー収集テスト ===")
    
    # テスト用ファイル作成
    if not create_test_reviewers_file():
        return
    
    # テスト実行
    scraper = BatchReviewScraper()
    scraper.run("../raw_data/test_reviewers.json")

if __name__ == "__main__":
    main()