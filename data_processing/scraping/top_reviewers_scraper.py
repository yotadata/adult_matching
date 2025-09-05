"""
トップレビュワー抽出スクリプト

DMM年間ランキングページから上位50名のレビュワー情報を抽出する
"""

import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import os
from urllib.parse import urljoin, urlparse, parse_qs

class TopReviewersScraper:
    def __init__(self, cookie_file: str = "../config/dmm_cookies.json"):
        self.session = requests.Session()
        self.cookie_file = cookie_file
        self.base_url = "https://review.dmm.co.jp"
        self.ranking_url = "https://review.dmm.co.jp/review-front/ranking/1year"
        
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
                # Chrome拡張機能形式
                for cookie in cookies_data:
                    if 'name' in cookie and 'value' in cookie:
                        domain = cookie.get('domain', '.dmm.co.jp')
                        self.session.cookies.set(
                            cookie['name'], 
                            cookie['value'], 
                            domain=domain
                        )
            elif isinstance(cookies_data, dict):
                # 辞書形式
                for name, value in cookies_data.items():
                    self.session.cookies.set(name, value, domain='.dmm.co.jp')
                    
            print(f"Cookieを読み込みました: {len(self.session.cookies)} 件")
            return True
            
        except Exception as e:
            print(f"Cookie読み込みエラー: {e}")
            return False
    
    def fetch_ranking_page(self) -> Optional[BeautifulSoup]:
        """ランキングページを取得してパース"""
        try:
            print(f"ランキングページ取得中: {self.ranking_url}")
            
            response = self.session.get(self.ranking_url, timeout=30)
            response.raise_for_status()
            
            print(f"レスポンス受信: {response.status_code}, サイズ: {len(response.content)} bytes")
            
            # BeautifulSoupでパース
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            print(f"ページ取得エラー: {e}")
            return None
        except Exception as e:
            print(f"パースエラー: {e}")
            return None
    
    def extract_reviewers(self, soup: BeautifulSoup, max_reviewers: int = 50) -> List[Dict[str, Any]]:
        """ランキングページからレビュワー情報を抽出"""
        reviewers = []
        
        try:
            # ランキング要素を検索（複数のセレクタを試行）
            selectors = [
                'a.css-g65o95',  # 仕様書で確認したセレクタ
                'a[href*="/review/-/list/reviewer/"]',  # URLパターンでの検索
                '.css-g65o95',  # クラス名のみ
            ]
            
            reviewer_elements = []
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    reviewer_elements = elements
                    print(f"レビュワー要素を発見: {len(elements)} 件 (セレクタ: {selector})")
                    break
            
            if not reviewer_elements:
                print("警告: レビュワー要素が見つかりませんでした")
                # デバッグ用: ページ構造を出力
                print("ページ構造の一部:")
                print(soup.get_text()[:500])
                return []
            
            # 各レビュワー要素から情報を抽出
            for i, element in enumerate(reviewer_elements[:max_reviewers]):
                reviewer_info = self.parse_reviewer_element(element, i + 1)
                if reviewer_info:
                    reviewers.append(reviewer_info)
                    print(f"  {i+1:2d}. {reviewer_info['username']} (ID: {reviewer_info['reviewer_id']}, レビュー: {reviewer_info['review_count']})")
                
                # レート制限
                if i > 0 and i % 10 == 0:
                    time.sleep(1)
            
            print(f"\n抽出完了: {len(reviewers)} 人のレビュワー情報")
            return reviewers
            
        except Exception as e:
            print(f"レビュワー抽出エラー: {e}")
            return []
    
    def parse_reviewer_element(self, element, rank: int) -> Optional[Dict[str, Any]]:
        """個別レビュワー要素から情報を抽出"""
        try:
            reviewer_info = {
                'rank': rank,
                'username': '',
                'reviewer_id': '',
                'profile_url': '',
                'review_count': 0,
                'helpful_count': 0,
                'extracted_at': datetime.now().isoformat()
            }
            
            # URLからレビュワーIDを抽出
            if element.name == 'a' and element.get('href'):
                profile_url = element.get('href')
                reviewer_info['profile_url'] = profile_url
                
                # URLパターン: https://www.dmm.co.jp/review/-/list/reviewer/=/id=12345/
                id_match = re.search(r'/id=([0-9]+)/', profile_url)
                if id_match:
                    reviewer_info['reviewer_id'] = id_match.group(1)
            
            # ユーザー名を抽出
            username_selectors = ['.css-735aui', '.username', 'span', 'div']
            for selector in username_selectors:
                username_element = element.select_one(selector)
                if username_element and username_element.get_text(strip=True):
                    reviewer_info['username'] = username_element.get_text(strip=True)
                    break
            
            # レビュー数と参考になった数を抽出
            text_content = element.get_text()
            
            # レビュー数を抽出（複数のパターンを試行）
            review_patterns = [
                r'レビュー.*?([0-9,]+).*?件',
                r'投稿.*?([0-9,]+).*?件',
                r'([0-9,]+).*?レビュー',
                r'([0-9,]+).*?件'
            ]
            
            for pattern in review_patterns:
                match = re.search(pattern, text_content)
                if match:
                    review_count_str = match.group(1).replace(',', '')
                    reviewer_info['review_count'] = int(review_count_str)
                    break
            
            # 参考になった数を抽出
            helpful_patterns = [
                r'参考になった.*?([0-9,]+).*?件',
                r'([0-9,]+).*?参考',
                r'helpful.*?([0-9,]+)',
            ]
            
            for pattern in helpful_patterns:
                match = re.search(pattern, text_content)
                if match:
                    helpful_count_str = match.group(1).replace(',', '')
                    reviewer_info['helpful_count'] = int(helpful_count_str)
                    break
            
            # 必須項目のチェック
            if not reviewer_info['reviewer_id'] or not reviewer_info['username']:
                print(f"警告: 必須情報が不足しています - {element.get_text()[:100]}")
                return None
            
            return reviewer_info
            
        except Exception as e:
            print(f"レビュワー解析エラー: {e}")
            return None
    
    def save_reviewers(self, reviewers: List[Dict[str, Any]], output_file: str = "../raw_data/top_reviewers.json") -> bool:
        """レビュワー情報をJSONファイルに保存"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(reviewers, f, ensure_ascii=False, indent=2)
            
            print(f"レビュワー情報を保存しました: {output_file}")
            print(f"保存件数: {len(reviewers)} 件")
            
            return True
            
        except Exception as e:
            print(f"保存エラー: {e}")
            return False
    
    def run(self) -> List[Dict[str, Any]]:
        """メイン実行関数"""
        print("=== DMM トップレビュワー抽出開始 ===")
        
        # Cookie読み込み
        if not self.load_cookies_from_json(self.cookie_file):
            print("Cookie認証なしで継続します")
        
        # ランキングページ取得
        soup = self.fetch_ranking_page()
        if not soup:
            print("エラー: ランキングページの取得に失敗しました")
            return []
        
        # レビュワー情報抽出
        reviewers = self.extract_reviewers(soup)
        if not reviewers:
            print("エラー: レビュワー情報の抽出に失敗しました")
            return []
        
        # 結果保存
        self.save_reviewers(reviewers)
        
        print("=== 抽出完了 ===")
        return reviewers

def main():
    scraper = TopReviewersScraper()
    reviewers = scraper.run()
    
    if reviewers:
        print(f"\n成功: {len(reviewers)} 人のレビュワー情報を取得しました")
        
        # 統計情報
        total_reviews = sum(r['review_count'] for r in reviewers)
        total_helpful = sum(r['helpful_count'] for r in reviewers)
        
        print(f"総レビュー数: {total_reviews:,} 件")
        print(f"総参考になった数: {total_helpful:,} 件")
        print(f"平均レビュー数: {total_reviews / len(reviewers):.1f} 件/人")
        
        # 上位5人を表示
        print("\n=== 上位5人 ===")
        for i, reviewer in enumerate(reviewers[:5]):
            print(f"{reviewer['rank']:2d}. {reviewer['username']} (ID: {reviewer['reviewer_id']}) - {reviewer['review_count']:,}件")
    else:
        print("エラー: レビュワー情報の取得に失敗しました")

if __name__ == "__main__":
    main()