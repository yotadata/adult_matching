import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import csv
from urllib.parse import urljoin, urlparse, parse_qs
import pickle
import os

class CookieDMMScraper:
    def __init__(self, cookie_file: str = "dmm_cookies.json"):
        self.session = requests.Session()
        self.cookie_file = cookie_file
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
            with open(cookie_file, 'r', encoding='utf-8') as f:
                cookies_data = json.load(f)
            
            # 複数の形式に対応
            if isinstance(cookies_data, list):
                # Chrome拡張機能形式 [{"name": "...", "value": "...", "domain": "..."}]
                for cookie in cookies_data:
                    if 'name' in cookie and 'value' in cookie:
                        domain = cookie.get('domain', '.dmm.co.jp')
                        self.session.cookies.set(
                            cookie['name'], 
                            cookie['value'], 
                            domain=domain
                        )
            elif isinstance(cookies_data, dict):
                # 辞書形式 {"cookie_name": "cookie_value"}
                for name, value in cookies_data.items():
                    self.session.cookies.set(name, value, domain='.dmm.co.jp')
            
            print(f"Cookieを読み込みました: {len(self.session.cookies)} 個")
            return True
            
        except FileNotFoundError:
            print(f"Cookieファイルが見つかりません: {cookie_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"Cookieファイルの形式が正しくありません: {e}")
            return False

    def load_cookies_from_browser_format(self, cookie_string: str):
        """ブラウザからコピーしたCookie文字列を読み込む"""
        # "name1=value1; name2=value2;" 形式
        cookies = {}
        for item in cookie_string.split(';'):
            if '=' in item:
                name, value = item.strip().split('=', 1)
                cookies[name] = value
        
        for name, value in cookies.items():
            self.session.cookies.set(name, value, domain='.dmm.co.jp')
        
        print(f"ブラウザ形式のCookieを読み込みました: {len(cookies)} 個")

    def save_cookies_to_json(self, cookie_file: str):
        """現在のCookieをJSONファイルに保存"""
        cookies_data = []
        for cookie in self.session.cookies:
            cookies_data.append({
                'name': cookie.name,
                'value': cookie.value,
                'domain': cookie.domain,
                'path': cookie.path,
                'secure': cookie.secure
            })
        
        with open(cookie_file, 'w', encoding='utf-8') as f:
            json.dump(cookies_data, f, ensure_ascii=False, indent=2)
        
        print(f"Cookieを保存しました: {cookie_file}")

    def test_cookie_validity(self, test_url: str = "https://www.dmm.co.jp/") -> bool:
        """Cookieが有効かテスト"""
        try:
            response = self.session.get(test_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 年齢認証ページかチェック
            if '年齢認証' in soup.get_text() or 'age_check' in response.url:
                print("Cookie認証失敗: まだ年齢認証ページです")
                return False
            
            # ログイン状態のチェック（ログインボタンがないかチェック）
            login_indicators = soup.find_all(['a', 'button'], string=re.compile(r'ログイン|Login', re.I))
            if not login_indicators:
                print("Cookie認証成功: サイトにアクセスできます")
                return True
            else:
                print("Cookie認証成功: ゲストアクセスが可能です")
                return True
                
        except Exception as e:
            print(f"Cookie検証エラー: {e}")
            return False

    def extract_reviews_from_page(self, url: str) -> List[Dict[str, Any]]:
        """Cookieを使用してページからレビューを抽出"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            print(f"URL: {url}")
            print(f"Response status: {response.status_code}")
            print(f"Final URL: {response.url}")
            print(f"Content length: {len(response.content)}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # デバッグ用にHTMLを保存
            with open('cookie_debug_page.html', 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            print("デバッグ用HTMLを保存: cookie_debug_page.html")
            
            # 年齢認証チェック
            if '年齢認証' in soup.get_text()[:1000]:
                print("警告: まだ年齢認証ページです")
                return []
            
            reviews = []
            
            # Method 1: JSONデータからの抽出
            script_reviews = self.extract_from_scripts(soup)
            reviews.extend(script_reviews)
            print(f"スクリプトから {len(script_reviews)} 件のレビューを抽出")
            
            # Method 2: HTMLからの抽出
            if not reviews:
                html_reviews = self.extract_from_html(soup)
                reviews.extend(html_reviews)
                print(f"HTMLから {len(html_reviews)} 件のレビューを抽出")
            
            # Method 3: より広範囲なHTML検索
            if not reviews:
                broad_reviews = self.extract_with_broad_search(soup)
                reviews.extend(broad_reviews)
                print(f"広範囲検索から {len(broad_reviews)} 件のレビューを抽出")
                
            return reviews
            
        except requests.RequestException as e:
            print(f"リクエストエラー: {e}")
            return []

    def extract_from_scripts(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """JavaScriptからレビューデータを抽出"""
        reviews = []
        script_tags = soup.find_all('script')
        
        print(f"スクリプトタグ数: {len(script_tags)}")
        
        for i, script in enumerate(script_tags):
            if not script.string:
                continue
                
            script_content = script.string
            
            # より多くのパターンを試行
            patterns = [
                r'reviewList\s*[:=]\s*(\[.*?\])',
                r'reviews\s*[:=]\s*(\[.*?\])', 
                r'items\s*[:=]\s*(\[.*?\])',
                r'data\s*[:=]\s*({.*?"reviewList".*?})',
                r'window\.__INITIAL_STATE__\s*=\s*({.*?})',
                r'window\.__NEXT_DATA__\s*=\s*({.*?})',
                r'__NUXT__\s*=\s*({.*?})',
                r'props\s*[:=]\s*({.*?"reviews".*?})',
                r'initialState\s*[:=]\s*({.*?})'
            ]
            
            for j, pattern in enumerate(patterns):
                try:
                    matches = re.finditer(pattern, script_content, re.DOTALL | re.IGNORECASE)
                    for match in matches:
                        json_str = match.group(1)
                        # JSONの長さ制限（メモリ保護）
                        if len(json_str) > 1000000:  # 1MB制限
                            continue
                            
                        try:
                            data = json.loads(json_str)
                            extracted = self.find_reviews_in_data(data)
                            if extracted:
                                print(f"パターン {j} (スクリプト {i}) から {len(extracted)} 件発見")
                                reviews.extend(extracted)
                        except json.JSONDecodeError:
                            continue
                            
                except Exception as e:
                    continue
        
        return reviews

    def extract_from_html(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """HTMLからレビューを抽出"""
        reviews = []
        
        # より具体的なセレクターを追加
        selectors = [
            # 一般的なレビューセレクター
            'div[class*="review"]',
            'li[class*="review"]', 
            'article[class*="review"]',
            'div[class*="comment"]',
            'div[class*="item"]',
            
            # DMMspecific selectors (推測)
            'div[class*="css-"]',  # CSS modules
            '.reviewItem',
            '.review-item', 
            '.comment-item',
            '[data-review]',
            '[data-comment]',
            
            # より広範囲
            'div[class*="list"] > div',
            'ul[class*="list"] > li',
            'section > div',
        ]
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    print(f"セレクター '{selector}' で {len(elements)} 要素発見")
                    
                    for element in elements[:20]:  # 最大20要素まで処理
                        review = self.extract_single_review(element)
                        if review and review.get('review_text') and len(review['review_text']) > 10:
                            reviews.append(review)
            except Exception as e:
                continue
        
        # 重複除去
        unique_reviews = []
        seen_texts = set()
        for review in reviews:
            text = review.get('review_text', '').strip()[:100]  # 最初の100文字で重複チェック
            if text and text not in seen_texts:
                unique_reviews.append(review)
                seen_texts.add(text)
        
        return unique_reviews

    def extract_with_broad_search(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """より広範囲でテキストを検索"""
        reviews = []
        
        # 長いテキストを含む要素を探す
        all_divs = soup.find_all(['div', 'p', 'span', 'li', 'td'])
        
        for element in all_divs:
            text = element.get_text(strip=True)
            
            # レビューらしいテキストの条件
            if (len(text) > 50 and  # 50文字以上
                len(text) < 2000 and  # 2000文字以下
                ('。' in text or '！' in text or '?' in text) and  # 文章らしい
                not ('copyright' in text.lower() or 'privacy' in text.lower())):  # ノイズ除去
                
                review = {
                    'review_text': text,
                    'source': 'broad_search',
                    'element_class': element.get('class', []),
                    'element_tag': element.name
                }
                
                # 同じ要素の兄弟から評価を探す
                parent = element.parent if element.parent else element
                rating_elem = parent.find(['div', 'span'], string=re.compile(r'[★⭐]{1,5}|[0-5]点|[0-5]/5'))
                if rating_elem:
                    rating_text = rating_elem.get_text(strip=True)
                    rating_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                    if rating_match:
                        review['rating'] = float(rating_match.group(1))
                    elif '★' in rating_text:
                        review['rating'] = len(re.findall(r'★', rating_text))
                
                reviews.append(review)
        
        # 長さでソートして重複除去
        reviews.sort(key=lambda x: len(x['review_text']), reverse=True)
        
        unique_reviews = []
        seen_texts = set()
        for review in reviews[:50]:  # 最大50件
            text = review['review_text'][:100]
            if text not in seen_texts:
                unique_reviews.append(review)
                seen_texts.add(text)
        
        return unique_reviews[:20]  # 最大20件返す

    def extract_single_review(self, element) -> Dict[str, Any]:
        """単一要素からレビューデータを抽出"""
        review = {}
        
        # テキスト抽出
        text = element.get_text(strip=True)
        if len(text) > 10:
            review['review_text'] = text
        
        # 評価抽出
        rating_patterns = [
            r'([★⭐]{1,5})',
            r'(\d+(?:\.\d+)?)点',
            r'(\d+(?:\.\d+)?)/5',
            r'評価[:：]\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, text)
            if match:
                if '★' in match.group(1):
                    review['rating'] = len(match.group(1))
                else:
                    try:
                        review['rating'] = float(match.group(1))
                    except:
                        pass
                break
        
        # クラス名やID情報を保存（デバッグ用）
        review['element_info'] = {
            'class': element.get('class', []),
            'id': element.get('id', ''),
            'tag': element.name
        }
        
        return review

    def find_reviews_in_data(self, data: Any) -> List[Dict[str, Any]]:
        """ネストされたデータからレビューを再帰的に検索"""
        reviews = []
        
        if isinstance(data, dict):
            # レビューっぽい単一オブジェクト
            if ('text' in data or 'review' in str(data).lower()) and len(str(data)) > 50:
                review = {
                    'review_text': data.get('text', data.get('review', data.get('comment', ''))),
                    'rating': data.get('rating', data.get('score', data.get('star', 0))),
                    'content_title': data.get('title', data.get('contentTitle', data.get('name', ''))),
                    'write_date': data.get('date', data.get('writeDate', data.get('created', ''))),
                    'content_id': data.get('contentId', data.get('id', '')),
                    'reviewer_id': data.get('reviewerId', data.get('userId', ''))
                }
                if review['review_text']:
                    reviews.append(review)
            
            # ネストされた構造を再帰検索
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    reviews.extend(self.find_reviews_in_data(value))
        
        elif isinstance(data, list):
            for item in data:
                reviews.extend(self.find_reviews_in_data(item))
        
        return reviews

    def scrape_multiple_pages(self, base_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
        """複数ページのスクレイピング"""
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            if 'page=' in base_url:
                url = re.sub(r'page=\d+', f'page={page}', base_url)
            elif '?' in base_url:
                url = f"{base_url}&page={page}"
            else:
                url = f"{base_url}?page={page}"
            
            print(f"\n=== ページ {page} の処理開始 ===")
            print(f"URL: {url}")
            
            reviews = self.extract_reviews_from_page(url)
            
            if not reviews:
                print(f"ページ {page} でレビューが見つかりませんでした。処理を終了します。")
                break
                
            all_reviews.extend(reviews)
            print(f"ページ {page}: {len(reviews)} 件のレビューを取得")
            
            # レート制限
            time.sleep(3)
        
        return all_reviews

    def save_to_json(self, reviews: List[Dict[str, Any]], filename: str = None):
        """JSON形式で保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dmm_reviews_cookie_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        
        print(f"レビューをJSONで保存: {filename} ({len(reviews)} 件)")

    def save_to_csv(self, reviews: List[Dict[str, Any]], filename: str = None):
        """CSV形式で保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dmm_reviews_cookie_{timestamp}.csv"
        
        if not reviews:
            print("保存するレビューがありません")
            return
        
        # すべてのキーを取得
        all_keys = set()
        for review in reviews:
            if isinstance(review, dict):
                all_keys.update(review.keys())
        
        # ネストされたオブジェクトは文字列化
        processed_reviews = []
        for review in reviews:
            processed_review = {}
            for key, value in review.items():
                if isinstance(value, (dict, list)):
                    processed_review[key] = json.dumps(value, ensure_ascii=False)
                else:
                    processed_review[key] = value
            processed_reviews.append(processed_review)
        
        fieldnames = sorted(list(all_keys))
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_reviews)
        
        print(f"レビューをCSVで保存: {filename} ({len(reviews)} 件)")

def main():
    scraper = CookieDMMScraper()
    
    print("=== DMM Cookie認証スクレイピング ===\n")
    
    # Cookie読み込みを試行
    cookie_loaded = False
    
    # 1. JSONファイルからの読み込み
    if os.path.exists("dmm_cookies.json"):
        cookie_loaded = scraper.load_cookies_from_json("dmm_cookies.json")
    
    # 2. 手動入力オプション
    if not cookie_loaded:
        print("Cookieファイルが見つかりません。")
        print("1. ブラウザでDMMにアクセスして年齢認証を完了してください")
        print("2. Developer Tools (F12) > Application > Cookies でCookie情報を確認してください")
        print("3. 重要なCookieをdmm_cookies.jsonファイルに保存してください")
        print("\n例: dmm_cookies.json")
        print('''{
  "ckcy": "1",
  "age_check_done": "1", 
  "session_id": "your_session_id"
}''')
        
        # 緊急用: 手動Cookie入力
        manual_cookies = input("\nCookie文字列を直接入力できます (name1=value1; name2=value2): ").strip()
        if manual_cookies:
            scraper.load_cookies_from_browser_format(manual_cookies)
            cookie_loaded = True
    
    if not cookie_loaded:
        print("Cookieが設定されていません。処理を終了します。")
        return
    
    # Cookie有効性テスト
    if not scraper.test_cookie_validity():
        print("Cookie認証に失敗しました。新しいCookieを取得してください。")
        return
    
    # スクレイピング実行
    target_url = "https://review.dmm.co.jp/review-front/reviewer/list/185585?page=1"
    print(f"\nターゲットURL: {target_url}")
    
    reviews = scraper.scrape_multiple_pages(target_url, max_pages=3)
    
    print(f"\n=== 結果 ===")
    print(f"取得したレビュー数: {len(reviews)}")
    
    if reviews:
        # ファイル保存
        scraper.save_to_json(reviews)
        scraper.save_to_csv(reviews)
        
        # サンプル表示
        print(f"\n=== サンプルレビュー ===")
        for i, review in enumerate(reviews[:3]):
            print(f"\n--- レビュー {i+1} ---")
            for key, value in review.items():
                if value and key != 'element_info':
                    print(f"{key}: {str(value)[:100]}...")
    else:
        print("レビューが取得できませんでした。")
        print("- Cookieが期限切れの可能性があります")
        print("- サイト構造が変更された可能性があります")
        print("- 年齢認証がまだ完了していない可能性があります")

if __name__ == "__main__":
    main()