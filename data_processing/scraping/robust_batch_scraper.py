"""
堅牢なバッチレビュー収集スクリプト

バックグラウンド実行・中断再開・進捗管理対応
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
import signal
import sys
import logging
from pathlib import Path

class RobustBatchScraper:
    def __init__(self, cookie_file: str = "../config/dmm_cookies.json"):
        self.session = requests.Session()
        self.cookie_file = cookie_file
        self.base_url = "https://review.dmm.co.jp"
        
        # 作業ディレクトリ設定
        self.output_dir = Path("../raw_data/batch_reviews")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = Path("../raw_data/scraping_progress.json")
        self.log_file = Path("../raw_data/scraping.log")
        
        # ログ設定
        self.setup_logging()
        
        # ヘッダー設定
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 進捗管理
        self.progress = {
            'start_time': None,
            'total_reviewers': 0,
            'completed_reviewers': [],
            'failed_reviewers': [],
            'current_reviewer': None,
            'total_reviews_collected': 0,
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # シグナルハンドラ設定（Ctrl+C対応）
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """シグナルハンドラ（中断時の処理）"""
        self.logger.info(f"中断シグナル受信: {signum}")
        self.save_progress()
        self.logger.info("進捗を保存しました。再開時は --resume オプションを使用してください")
        sys.exit(0)
    
    def load_cookies_from_json(self, cookie_file: str) -> bool:
        """JSONファイルからCookieを読み込む"""
        try:
            if not os.path.exists(cookie_file):
                self.logger.warning(f"Cookieファイル {cookie_file} が見つかりません")
                return False
                
            with open(cookie_file, 'r', encoding='utf-8') as f:
                cookies_data = json.load(f)
            
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
                    
            self.logger.info(f"Cookie読み込み完了: {len(self.session.cookies)} 件")
            return True
            
        except Exception as e:
            self.logger.error(f"Cookie読み込みエラー: {e}")
            return False
    
    def load_progress(self) -> bool:
        """進捗状況を読み込み"""
        try:
            if not self.progress_file.exists():
                return False
                
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                saved_progress = json.load(f)
                
            self.progress.update(saved_progress)
            self.logger.info(f"進捗を復元しました: {len(self.progress['completed_reviewers'])} 人完了済み")
            return True
            
        except Exception as e:
            self.logger.error(f"進捗読み込みエラー: {e}")
            return False
    
    def save_progress(self):
        """進捗状況を保存"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
            self.logger.debug("進捗保存完了")
        except Exception as e:
            self.logger.error(f"進捗保存エラー: {e}")
    
    def is_reviewer_completed(self, reviewer_id: str) -> bool:
        """レビュワーが既に処理済みかチェック"""
        # 完了リストに含まれているか
        if reviewer_id in self.progress['completed_reviewers']:
            return True
            
        # 出力ファイルが存在するかチェック
        output_file = self.output_dir / f"reviewer_{reviewer_id}.json"
        if output_file.exists():
            # ファイルサイズが十分あるかチェック（空ファイル対策）
            if output_file.stat().st_size > 100:
                return True
        
        return False
    
    def load_reviewers_list(self, input_file: str = "../raw_data/top_reviewers.json") -> List[Dict[str, Any]]:
        """レビュワーリストを読み込み、未完了分のみを返す"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                all_reviewers = json.load(f)
            
            # 未完了分のみフィルタ
            pending_reviewers = [
                r for r in all_reviewers 
                if not self.is_reviewer_completed(r['reviewer_id'])
            ]
            
            self.logger.info(f"レビュワーリスト読み込み: 全{len(all_reviewers)}人中、未完了{len(pending_reviewers)}人")
            return pending_reviewers
            
        except Exception as e:
            self.logger.error(f"レビュワーリスト読み込みエラー: {e}")
            return []
    
    def fetch_reviewer_page(self, reviewer_id: str, page: int = 1) -> Optional[BeautifulSoup]:
        """レビュワーページを取得（リトライ機能付き）"""
        url = f"https://review.dmm.co.jp/review-front/reviewer/list/{reviewer_id}?page={page}"
        
        for attempt in range(3):  # 最大3回リトライ
            try:
                self.logger.debug(f"ページ取得: {reviewer_id} page={page} (試行{attempt+1})")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                return soup
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"ページ取得エラー {reviewer_id} page={page} (試行{attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(5)  # リトライ前に待機
                continue
        
        self.logger.error(f"ページ取得失敗: {reviewer_id} page={page}")
        return None
    
    def extract_reviews_from_page(self, soup: BeautifulSoup, reviewer_id: str, page: int) -> List[Dict[str, Any]]:
        """ページからレビューデータを抽出"""
        reviews = []
        
        try:
            # JSONデータの検索
            scripts = soup.find_all('script')
            review_data = None
            
            for script in scripts:
                script_text = script.get_text()
                
                if 'reviewList' in script_text:
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
            
            if review_data and isinstance(review_data, list):
                for review_item in review_data:
                    if isinstance(review_item, dict):
                        processed_review = self.process_review_data(review_item, reviewer_id, page)
                        if processed_review:
                            reviews.append(processed_review)
            
            return reviews
            
        except Exception as e:
            self.logger.error(f"レビュー抽出エラー {reviewer_id} page={page}: {e}")
            return []
    
    def process_review_data(self, raw_review: Dict[str, Any], reviewer_id: str, page: int) -> Optional[Dict[str, Any]]:
        """レビューデータを処理"""
        try:
            processed = {
                'reviewer_id': reviewer_id,
                'page_number': page,
                'extracted_at': datetime.now().isoformat(),
            }
            
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
                return None
            
            # 評価の正規化
            rating = processed.get('rating')
            if rating is not None:
                try:
                    processed['rating'] = float(rating)
                except (ValueError, TypeError):
                    processed['rating'] = None
            
            return processed
            
        except Exception as e:
            self.logger.error(f"レビュー処理エラー: {e}")
            return None
    
    def get_total_pages(self, reviewer_id: str) -> int:
        """総ページ数を取得"""
        try:
            soup = self.fetch_reviewer_page(reviewer_id, 1)
            if not soup:
                return 1
            
            max_page = 1
            
            # ページネーション解析
            pagination_selectors = [
                '.pagination a',
                '[class*="page"] a',
                'a[href*="page="]'
            ]
            
            for selector in pagination_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    page_match = re.search(r'page=([0-9]+)', href)
                    if page_match:
                        page_num = int(page_match.group(1))
                        max_page = max(max_page, page_num)
            
            return min(max_page, 200)  # 安全上限
            
        except Exception as e:
            self.logger.error(f"総ページ数取得エラー {reviewer_id}: {e}")
            return 1
    
    def collect_reviewer_reviews(self, reviewer: Dict[str, Any]) -> List[Dict[str, Any]]:
        """レビュワーの全レビューを収集"""
        reviewer_id = reviewer['reviewer_id']
        reviewer_name = reviewer.get('username', 'Unknown')[:20]
        
        self.progress['current_reviewer'] = reviewer_id
        self.save_progress()
        
        try:
            self.logger.info(f"収集開始: {reviewer_name} (ID: {reviewer_id})")
            
            total_pages = self.get_total_pages(reviewer_id)
            self.logger.info(f"推定ページ数: {total_pages}")
            
            all_reviews = []
            
            for page in range(1, total_pages + 1):
                time.sleep(3)  # レート制限
                
                soup = self.fetch_reviewer_page(reviewer_id, page)
                if not soup:
                    self.logger.warning(f"ページ {page} スキップ")
                    continue
                
                reviews = self.extract_reviews_from_page(soup, reviewer_id, page)
                all_reviews.extend(reviews)
                
                self.logger.info(f"ページ {page}/{total_pages}: {len(reviews)}件収集")
                
                # 空ページで終了
                if len(reviews) == 0 and page > 1:
                    self.logger.info("空ページ検出、収集終了")
                    break
                
                # 中間保存（10ページごと）
                if page % 10 == 0:
                    self.save_reviewer_reviews(all_reviews, reviewer_id)
                    self.logger.info(f"中間保存完了: {len(all_reviews)}件")
            
            self.logger.info(f"収集完了: {reviewer_name} - 総数: {len(all_reviews)}件")
            
            # 最終保存
            self.save_reviewer_reviews(all_reviews, reviewer_id)
            
            # 進捗更新
            self.progress['completed_reviewers'].append(reviewer_id)
            self.progress['total_reviews_collected'] += len(all_reviews)
            self.progress['current_reviewer'] = None
            self.save_progress()
            
            return all_reviews
            
        except Exception as e:
            self.logger.error(f"レビュワー収集エラー {reviewer_id}: {e}")
            self.progress['failed_reviewers'].append(reviewer_id)
            self.save_progress()
            return []
    
    def save_reviewer_reviews(self, reviews: List[Dict[str, Any]], reviewer_id: str) -> bool:
        """レビューデータを保存"""
        try:
            output_file = self.output_dir / f"reviewer_{reviewer_id}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(reviews, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"保存完了: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存エラー {reviewer_id}: {e}")
            return False
    
    def collect_all_reviewers(self, reviewers: List[Dict[str, Any]]) -> None:
        """全レビュワー収集"""
        self.progress['total_reviewers'] = len(reviewers)
        self.progress['start_time'] = datetime.now().isoformat()
        
        self.logger.info(f"=== バッチ収集開始: {len(reviewers)}人 ===")
        
        for i, reviewer in enumerate(reviewers):
            try:
                self.collect_reviewer_reviews(reviewer)
                
                # 進捗表示
                completed = len(self.progress['completed_reviewers'])
                progress_pct = completed / len(reviewers) * 100 if reviewers else 0
                
                elapsed = datetime.now() - datetime.fromisoformat(self.progress['start_time'])
                self.logger.info(f"進捗: {progress_pct:.1f}% ({completed}/{len(reviewers)}) - 経過: {elapsed}")
                
                # 長時間インターバル
                if (i + 1) % 5 == 0:
                    self.logger.info("レート制限: 60秒待機")
                    time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"処理エラー: {e}")
                continue
        
        self.print_final_stats()
    
    def print_final_stats(self):
        """最終統計表示"""
        completed = len(self.progress['completed_reviewers'])
        failed = len(self.progress['failed_reviewers'])
        total_reviews = self.progress['total_reviews_collected']
        
        if self.progress['start_time']:
            elapsed = datetime.now() - datetime.fromisoformat(self.progress['start_time'])
        else:
            elapsed = "不明"
        
        self.logger.info("\n=== 収集完了統計 ===")
        self.logger.info(f"成功レビュワー: {completed}")
        self.logger.info(f"失敗レビュワー: {failed}")
        self.logger.info(f"総レビュー数: {total_reviews:,}")
        self.logger.info(f"実行時間: {elapsed}")
        self.logger.info(f"平均レビュー数/人: {total_reviews / max(1, completed):.1f}")
    
    def run(self, input_file: str = "../raw_data/top_reviewers.json", resume: bool = False) -> None:
        """メイン実行"""
        self.logger.info("=== 堅牢バッチスクレイパー開始 ===")
        
        if resume and self.load_progress():
            self.logger.info("中断から再開します")
        else:
            self.logger.info("新規実行開始")
        
        # Cookie読み込み
        if not self.load_cookies_from_json(self.cookie_file):
            self.logger.warning("Cookie認証なしで継続")
        
        # レビュワーリスト読み込み
        reviewers = self.load_reviewers_list(input_file)
        if not reviewers:
            self.logger.error("レビュワーリスト読み込み失敗")
            return
        
        # 収集実行
        self.collect_all_reviewers(reviewers)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='堅牢バッチレビュースクレイパー')
    parser.add_argument('--resume', action='store_true', help='中断から再開')
    parser.add_argument('--input', default='../raw_data/top_reviewers.json', help='入力ファイル')
    
    args = parser.parse_args()
    
    scraper = RobustBatchScraper()
    scraper.run(args.input, args.resume)

if __name__ == "__main__":
    main()