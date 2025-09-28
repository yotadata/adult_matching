"""
バッチレビュー収集スクリプト

トップレビュワーの個別ページから全レビューデータを収集する
"""

import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Iterator
import time
from datetime import datetime
import os
from urllib.parse import urljoin, urlparse, parse_qs
import concurrent.futures
from threading import Lock
import math

class BatchReviewScraper:
    def __init__(self, cookie_file: str = "../config/dmm_cookies.json"):
        self.session = requests.Session()
        self.cookie_file = cookie_file
        self.base_url = "https://review.dmm.co.jp"
        self.lock = Lock()  # スレッドセーフティ用
        
        # ヘッダー設定
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        })
        
        # 収集統計
        self.stats = {
            'total_reviewers': 0,
            'completed_reviewers': 0,
            'total_reviews': 0,
            'failed_reviewers': 0,
            'start_time': None
        }
        
    def load_cookies_from_json(self, cookie_file: str) -> bool:
        """JSONファイルからCookieを読み込む"""
        try:
            if not os.path.exists(cookie_file):
                print(f"警告: Cookieファイル {cookie_file} が見つかりません")
                return False
                
            with open(cookie_file, 'r', encoding='utf-8') as f:
                cookies_data = json.load(f)
            
            # 複数の形式に対応
            if isinstance(cookies_data, list):
                for cookie in cookies_data:
                    if 'name' in cookie and 'value' in cookie:
                        domain = cookie.get('domain', '.dmm.co.jp')
                        self.session.cookies.set(
                            cookie['name'], 
                            cookie['value'], 
                            domain=domain
                        )
            elif isinstance(cookies_data, dict):
                for name, value in cookies_data.items():
                    self.session.cookies.set(name, value, domain='.dmm.co.jp')
                    
            return True
            
        except Exception as e:
            print(f"Cookie読み込みエラー: {e}")
            return False
    
    def load_reviewers_list(self, input_file: str = "../raw_data/top_reviewers.json") -> List[Dict[str, Any]]:
        """レビュワーリストをJSONファイルから読み込み"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                reviewers = json.load(f)
            print(f"レビュワーリスト読み込み: {len(reviewers)} 人")
            return reviewers
        except Exception as e:
            print(f"レビュワーリスト読み込みエラー: {e}")
            return []
    
    def fetch_reviewer_page(self, reviewer_id: str, page: int = 1) -> Optional[BeautifulSoup]:
        """レビュワーの特定ページを取得"""
        url = f"https://review.dmm.co.jp/review-front/reviewer/list/{reviewer_id}?page={page}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            print(f"ページ取得エラー {reviewer_id} (page {page}): {e}")
            return None
    
    def extract_reviews_from_page(self, soup: BeautifulSoup, reviewer_id: str, page: int) -> List[Dict[str, Any]]:
        """ページからレビューデータを抽出"""
        reviews = []
        
        try:
            # ページ内のJSONデータを検索
            scripts = soup.find_all('script')
            review_data = None
            
            for script in scripts:
                script_text = script.get_text()
                
                # reviewListを含むJSONデータを検索
                if 'reviewList' in script_text:
                    # JSONデータの抽出パターンを試行
                    patterns = [
                        r'reviewList["\']?:\s*(\[.*?\])',
                        r'"reviewList":\s*(\[.*?\])',
                        r'reviewList:\s*(\[.*?\])',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, script_text, re.DOTALL)
                        if match:
                            try:
                                review_data = json.loads(match.group(1))
                                break
                            except json.JSONDecodeError:
                                continue
                    
                    if review_data:
                        break
            
            # レビューデータが見つからない場合の代替手段
            if not review_data:
                print(f"警告: レビュワー {reviewer_id} (page {page}) でJSONデータが見つかりません")
                # HTMLから直接パースを試行
                review_data = self.parse_reviews_from_html(soup)
            
            # レビューデータを処理
            if review_data and isinstance(review_data, list):
                for review_item in review_data:
                    if isinstance(review_item, dict):
                        processed_review = self.process_review_data(review_item, reviewer_id, page)
                        if processed_review:
                            reviews.append(processed_review)
            
            return reviews
            
        except Exception as e:
            print(f"レビュー抽出エラー {reviewer_id} (page {page}): {e}")
            return []
    
    def parse_reviews_from_html(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """HTMLから直接レビューデータをパース（フォールバック）"""
        reviews = []
        
        try:
            # レビューアイテムの検索（複数のセレクタを試行）
            selectors = [
                '.review-item',
                '[class*="review"]',
                '[class*="item"]',
                'article',
                'div[class*="turtle"]'
            ]
            
            for selector in selectors:
                review_elements = soup.select(selector)
                if review_elements:
                    print(f"HTMLパース: {len(review_elements)} 件のレビューを発見")
                    
                    for element in review_elements:
                        review_data = {
                            'title': '',
                            'text': '',
                            'value': None,
                            'contentTitle': '',
                            'writeDate': None,
                        }
                        
                        # テキスト抽出
                        text_content = element.get_text(strip=True)
                        if len(text_content) > 50:  # 最低文字数チェック
                            review_data['text'] = text_content[:1000]  # 制限
                            reviews.append(review_data)
                    
                    break
            
            return reviews
            
        except Exception as e:
            print(f"HTMLパースエラー: {e}")
            return []
    
    def process_review_data(self, raw_review: Dict[str, Any], reviewer_id: str, page: int) -> Optional[Dict[str, Any]]:
        """生のレビューデータを処理して標準化"""
        try:
            processed = {
                'reviewer_id': reviewer_id,
                'page_number': page,
                'extracted_at': datetime.now().isoformat(),
            }
            
            # 必要なフィールドをマッピング
            field_mapping = {
                'contentId': 'content_id',
                'displayShopName': 'category',
                'title': 'title',
                'text': 'review_text',
                'value': 'rating',
                'writeDate': 'write_date',
                'contentTitle': 'content_title',
                'contentUrl': 'content_url',
                'contentImageSrc': 'content_image',
                'evaluateCount': 'helpful_count',
            }
            
            for orig_key, new_key in field_mapping.items():
                processed[new_key] = raw_review.get(orig_key, None)
            
            # データ品質チェック
            review_text = processed.get('review_text', '')
            if isinstance(review_text, str) and len(review_text.strip()) < 10:
                return None  # 短すぎるレビューは除外
            
            # 評価の正規化
            rating = processed.get('rating')
            if rating is not None:
                try:
                    processed['rating'] = float(rating)
                except (ValueError, TypeError):
                    processed['rating'] = None
            
            return processed
            
        except Exception as e:
            print(f"レビュー処理エラー: {e}")
            return None
    
    def get_total_pages(self, reviewer_id: str) -> int:
        """レビュワーの総ページ数を取得"""
        try:
            soup = self.fetch_reviewer_page(reviewer_id, 1)
            if not soup:
                return 1
            
            # ページネーション要素を検索
            pagination_selectors = [
                '.pagination a',
                '[class*="page"] a',
                'a[href*="page="]'
            ]
            
            max_page = 1
            
            for selector in pagination_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    page_match = re.search(r'page=([0-9]+)', href)
                    if page_match:
                        page_num = int(page_match.group(1))
                        max_page = max(max_page, page_num)
            
            # テキストからも推定
            page_text = soup.get_text()
            page_patterns = [
                r'([0-9]+)\s*ページ目',
                r'(\d+)\s*/\s*(\d+)',
                r'全\s*(\d+)\s*ページ'
            ]
            
            for pattern in page_patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            max_page = max(max_page, max(int(x) for x in match))
                        else:
                            max_page = max(max_page, int(match))
            
            return min(max_page, 100)  # 安全のため上限設定
            
        except Exception as e:
            print(f"総ページ数取得エラー {reviewer_id}: {e}")
            return 1
    
    def collect_reviewer_reviews(self, reviewer: Dict[str, Any], max_pages: int = None) -> List[Dict[str, Any]]:
        """特定レビュワーの全レビューを収集"""
        reviewer_id = reviewer['reviewer_id']
        reviewer_name = reviewer.get('username', 'Unknown')[:20]
        
        try:
            print(f"開始: {reviewer_name} (ID: {reviewer_id})")
            
            # 総ページ数を取得
            if max_pages is None:
                total_pages = self.get_total_pages(reviewer_id)
            else:
                total_pages = max_pages
            
            print(f"  推定ページ数: {total_pages}")
            
            all_reviews = []
            
            # 各ページを順次処理
            for page in range(1, total_pages + 1):
                time.sleep(2)  # レート制限
                
                soup = self.fetch_reviewer_page(reviewer_id, page)
                if not soup:
                    print(f"  ページ {page} の取得に失敗")
                    continue
                
                reviews = self.extract_reviews_from_page(soup, reviewer_id, page)
                all_reviews.extend(reviews)
                
                print(f"  ページ {page}/{total_pages}: {len(reviews)} 件のレビュー")
                
                # 空ページが出たら終了
                if len(reviews) == 0 and page > 1:
                    print(f"  空ページを検出、終了")
                    break
            
            print(f"完了: {reviewer_name} - 総レビュー数: {len(all_reviews)}")
            
            # 統計更新
            with self.lock:
                self.stats['total_reviews'] += len(all_reviews)
                self.stats['completed_reviewers'] += 1
            
            return all_reviews
            
        except Exception as e:
            print(f"レビュワー収集エラー {reviewer_id}: {e}")
            with self.lock:
                self.stats['failed_reviewers'] += 1
            return []
    
    def save_reviewer_reviews(self, reviews: List[Dict[str, Any]], reviewer_id: str) -> bool:
        """レビュワーのレビューデータを保存"""
        try:
            output_dir = "../raw_data/batch_reviews"
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"reviewer_{reviewer_id}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(reviews, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"保存エラー {reviewer_id}: {e}")
            return False
    
    def collect_all_reviewers(self, reviewers: List[Dict[str, Any]], max_workers: int = 3) -> None:
        """全レビュワーのデータを並列収集"""
        self.stats['total_reviewers'] = len(reviewers)
        self.stats['start_time'] = datetime.now()
        
        print(f"=== バッチ収集開始: {len(reviewers)} 人のレビュワー ===")
        
        # シーケンシャル処理（安全性重視）
        for i, reviewer in enumerate(reviewers):
            reviewer_id = reviewer['reviewer_id']
            
            try:
                # レビューデータ収集
                reviews = self.collect_reviewer_reviews(reviewer)
                
                # 保存
                if reviews:
                    success = self.save_reviewer_reviews(reviews, reviewer_id)
                    if success:
                        print(f"保存完了: reviewer_{reviewer_id}.json ({len(reviews)} 件)")
                
                # 進捗表示
                progress = (i + 1) / len(reviewers) * 100
                elapsed = datetime.now() - self.stats['start_time']
                print(f"進捗: {progress:.1f}% ({i+1}/{len(reviewers)}) - 経過時間: {elapsed}")
                
                # 長時間間隔でのレート制限
                if (i + 1) % 5 == 0:
                    print("レート制限: 30秒待機")
                    time.sleep(30)
                
            except Exception as e:
                print(f"レビュワー処理エラー {reviewer_id}: {e}")
                with self.lock:
                    self.stats['failed_reviewers'] += 1
                continue
        
        self.print_final_stats()
    
    def print_final_stats(self):
        """最終統計を表示"""
        elapsed = datetime.now() - self.stats['start_time']
        
        print("\n=== 収集完了統計 ===")
        print(f"対象レビュワー数: {self.stats['total_reviewers']}")
        print(f"成功: {self.stats['completed_reviewers']}")
        print(f"失敗: {self.stats['failed_reviewers']}")
        print(f"総レビュー数: {self.stats['total_reviews']:,}")
        print(f"実行時間: {elapsed}")
        print(f"平均レビュー数/人: {self.stats['total_reviews'] / max(1, self.stats['completed_reviewers']):.1f}")
    
    def run(self, input_file: str = "../raw_data/top_reviewers.json") -> None:
        """メイン実行関数"""
        print("=== バッチレビュー収集開始 ===")
        
        # Cookie読み込み
        if not self.load_cookies_from_json(self.cookie_file):
            print("Cookie認証なしで継続します")
        
        # レビュワーリスト読み込み
        reviewers = self.load_reviewers_list(input_file)
        if not reviewers:
            print("エラー: レビュワーリストの読み込みに失敗")
            return
        
        # 収集実行
        self.collect_all_reviewers(reviewers)

def main():
    scraper = BatchReviewScraper()
    scraper.run()

if __name__ == "__main__":
    main()