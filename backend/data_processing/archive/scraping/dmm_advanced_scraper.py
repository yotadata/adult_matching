import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import time
from datetime import datetime
import csv
from urllib.parse import urljoin, urlparse, parse_qs

class DMMAdvancedScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def handle_age_verification(self, url: str) -> str:
        """Handle age verification if present and return final URL"""
        response = self.session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if this is an age verification page
        if 'age_check' in response.url or '年齢認証' in soup.get_text():
            print("Age verification detected, attempting to bypass...")
            
            # Look for age verification form or buttons
            forms = soup.find_all('form')
            for form in forms:
                action = form.get('action', '')
                if 'age' in action.lower() or 'verify' in action.lower():
                    # Submit the form to accept age verification
                    form_data = {}
                    for input_tag in form.find_all('input'):
                        name = input_tag.get('name')
                        value = input_tag.get('value', '')
                        if name:
                            form_data[name] = value
                    
                    # Add common age verification parameters
                    form_data.update({
                        'age_check': 'yes',
                        'age': 'over18',
                        'accept': '1'
                    })
                    
                    # Submit form
                    if action:
                        verify_url = urljoin(response.url, action)
                    else:
                        verify_url = response.url
                        
                    verify_response = self.session.post(verify_url, data=form_data)
                    if verify_response.status_code == 200:
                        print(f"Age verification submitted, redirected to: {verify_response.url}")
                        return verify_response.url
            
            # Try common age verification bypass patterns
            bypass_patterns = [
                '/enter',
                '?age_check=ok',
                '?agecheck=1',
                '?over18=1'
            ]
            
            for pattern in bypass_patterns:
                bypass_url = url + pattern
                try:
                    bypass_response = self.session.get(bypass_url)
                    if bypass_response.status_code == 200 and 'age_check' not in bypass_response.url:
                        print(f"Age verification bypassed with: {pattern}")
                        return bypass_response.url
                except:
                    continue
                    
            # If we couldn't bypass, try setting cookies
            self.session.cookies.update({
                'age_check': 'yes',
                'age_verification': 'confirmed',
                'adult_check': '1',
                'over18': 'true'
            })
            
        return url

    def extract_reviews_from_page(self, url: str) -> List[Dict[str, Any]]:
        """Extract reviews from a single page"""
        try:
            # Handle age verification first
            final_url = self.handle_age_verification(url)
            
            response = self.session.get(final_url)
            response.raise_for_status()
            
            print(f"Final URL: {final_url}")
            print(f"Response status: {response.status_code}")
            print(f"Content length: {len(response.content)}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if we still have age verification
            if 'age_check' in response.url or '年齢認証' in soup.get_text()[:500]:
                print("Still on age verification page, trying direct access...")
                # Try to construct the direct URL without age check
                if 'dmm.co.jp' in url and 'age_check' not in url:
                    # Set adult cookies and try again
                    self.session.cookies.update({
                        'ckcy': '1',  # DMM cookie
                        'age_check_done': '1',
                        'adult': '1'
                    })
                    response = self.session.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract reviews using multiple methods
            reviews = []
            
            # Method 1: Look for JSON data in script tags
            script_reviews = self.extract_from_scripts(soup)
            reviews.extend(script_reviews)
            
            # Method 2: Extract from HTML structure
            if not reviews:
                html_reviews = self.extract_from_html_structure(soup)
                reviews.extend(html_reviews)
            
            # Method 3: Try API endpoints
            if not reviews:
                api_reviews = self.try_api_endpoints(url)
                reviews.extend(api_reviews)
                
            return reviews
            
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []

    def extract_from_scripts(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract reviews from JavaScript data"""
        reviews = []
        script_tags = soup.find_all('script')
        
        for script in script_tags:
            if not script.string:
                continue
                
            script_content = script.string
            
            # Look for various JSON patterns
            json_patterns = [
                r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                r'window\.__NEXT_DATA__\s*=\s*({.*?});',
                r'reviewList\s*[:=]\s*(\[.*?\])',
                r'reviews\s*[:=]\s*(\[.*?\])',
                r'items\s*[:=]\s*(\[.*?\])',
                r'data\s*[:=]\s*({.*?reviewList.*?})',
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, script_content, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1)
                        data = json.loads(json_str)
                        
                        # Extract reviews from nested data
                        extracted_reviews = self.find_reviews_in_data(data)
                        reviews.extend(extracted_reviews)
                        
                    except json.JSONDecodeError:
                        continue
        
        return reviews

    def extract_from_html_structure(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract reviews from HTML structure"""
        reviews = []
        
        # Look for review containers with various selectors
        selectors = [
            'div[class*="review"]',
            'li[class*="review"]',
            'div[class*="comment"]',
            'article[class*="review"]',
            '.review-item',
            '.comment-item',
            '[data-review]',
            '[data-comment]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                print(f"Found {len(elements)} elements with selector: {selector}")
                
            for element in elements:
                review = self.extract_single_review(element)
                if review and review.get('review_text'):
                    reviews.append(review)
        
        return reviews

    def extract_single_review(self, element) -> Dict[str, Any]:
        """Extract review data from single HTML element"""
        review = {}
        
        # Extract review text
        text_selectors = [
            '.review-text',
            '.comment-text',
            '.content',
            'p',
            '[class*="text"]'
        ]
        
        for selector in text_selectors:
            text_elem = element.select_one(selector)
            if text_elem:
                text = text_elem.get_text(strip=True)
                if len(text) > 20:  # Reasonable minimum length
                    review['review_text'] = text
                    break
        
        # Extract rating
        rating_selectors = [
            '.rating',
            '.score',
            '[class*="star"]',
            '[class*="rating"]'
        ]
        
        for selector in rating_selectors:
            rating_elem = element.select_one(selector)
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                # Try to extract numeric rating
                rating_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if rating_match:
                    review['rating'] = float(rating_match.group(1))
                    break
                    
                # Count stars
                star_count = len(re.findall(r'[★⭐]', rating_text))
                if star_count > 0:
                    review['rating'] = star_count
                    break
        
        # Extract title
        title_selectors = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            '.title',
            '.name',
            'a[href*="item"]'
        ]
        
        for selector in title_selectors:
            title_elem = element.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title and len(title) < 200:  # Reasonable title length
                    review['content_title'] = title
                    break
        
        # Extract date
        date_selectors = [
            'time',
            '.date',
            '[class*="date"]',
            '[datetime]'
        ]
        
        for selector in date_selectors:
            date_elem = element.select_one(selector)
            if date_elem:
                date_text = date_elem.get('datetime') or date_elem.get_text(strip=True)
                if date_text:
                    review['write_date'] = date_text
                    break
        
        return review

    def try_api_endpoints(self, url: str) -> List[Dict[str, Any]]:
        """Try to find API endpoints that might contain review data"""
        reviews = []
        
        # Parse the original URL to extract parameters
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        query_params = parse_qs(parsed_url.query)
        
        # Try to find reviewer ID or product ID
        reviewer_id = None
        for part in path_parts:
            if part.isdigit():
                reviewer_id = part
                break
        
        if not reviewer_id:
            return reviews
        
        # Common API patterns for DMM
        api_endpoints = [
            f"https://api.dmm.co.jp/reviews/{reviewer_id}",
            f"https://review.dmm.co.jp/api/reviews/{reviewer_id}",
            f"https://dmm.co.jp/api/review-front/reviewer/{reviewer_id}",
        ]
        
        for endpoint in api_endpoints:
            try:
                api_response = self.session.get(endpoint)
                if api_response.status_code == 200:
                    try:
                        api_data = api_response.json()
                        api_reviews = self.find_reviews_in_data(api_data)
                        reviews.extend(api_reviews)
                        print(f"Found {len(api_reviews)} reviews from API: {endpoint}")
                    except json.JSONDecodeError:
                        continue
            except:
                continue
        
        return reviews

    def find_reviews_in_data(self, data: Any) -> List[Dict[str, Any]]:
        """Recursively find review data in nested structures"""
        reviews = []
        
        if isinstance(data, dict):
            # Check for review-like structures
            if all(key in data for key in ['text', 'rating']) or 'review' in str(data).lower():
                # This looks like a single review
                if 'text' in data:
                    review = {
                        'review_text': data.get('text', ''),
                        'rating': data.get('rating', data.get('score', 0)),
                        'content_title': data.get('title', data.get('contentTitle', '')),
                        'write_date': data.get('date', data.get('writeDate', '')),
                        'content_id': data.get('contentId', ''),
                        'reviewer_id': data.get('reviewerId', '')
                    }
                    reviews.append(review)
            
            # Look for arrays of reviews
            for key, value in data.items():
                if key.lower() in ['reviews', 'reviewlist', 'items', 'data', 'results']:
                    if isinstance(value, list):
                        for item in value:
                            nested_reviews = self.find_reviews_in_data(item)
                            reviews.extend(nested_reviews)
                elif isinstance(value, (dict, list)):
                    nested_reviews = self.find_reviews_in_data(value)
                    reviews.extend(nested_reviews)
        
        elif isinstance(data, list):
            for item in data:
                nested_reviews = self.find_reviews_in_data(item)
                reviews.extend(nested_reviews)
        
        return reviews

    def scrape_multiple_pages(self, base_url: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Scrape multiple pages of reviews"""
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            # Construct page URL
            if 'page=' in base_url:
                url = re.sub(r'page=\d+', f'page={page}', base_url)
            elif '?' in base_url:
                url = f"{base_url}&page={page}"
            else:
                url = f"{base_url}?page={page}"
            
            print(f"\nScraping page {page}: {url}")
            
            reviews = self.extract_reviews_from_page(url)
            if not reviews:
                print(f"No reviews found on page {page}, stopping...")
                break
                
            all_reviews.extend(reviews)
            print(f"Found {len(reviews)} reviews on page {page}")
            
            # Rate limiting
            time.sleep(2)
        
        return all_reviews

    def save_to_json(self, reviews: List[Dict[str, Any]], filename: str = None):
        """Save reviews to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dmm_reviews_advanced_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(reviews)} reviews to {filename}")

    def save_to_csv(self, reviews: List[Dict[str, Any]], filename: str = None):
        """Save reviews to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dmm_reviews_advanced_{timestamp}.csv"
        
        if not reviews:
            print("No reviews to save")
            return
        
        # Get all unique keys
        all_keys = set()
        for review in reviews:
            all_keys.update(review.keys())
        
        fieldnames = sorted(list(all_keys))
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reviews)
        
        print(f"Saved {len(reviews)} reviews to {filename}")

def main():
    scraper = DMMAdvancedScraper()
    
    # Target URL
    url = "https://review.dmm.co.jp/review-front/reviewer/list/185585?page=1"
    
    print("Starting advanced DMM review scraping...")
    print(f"Target URL: {url}")
    
    # Scrape multiple pages
    reviews = scraper.scrape_multiple_pages(url, max_pages=3)
    print(f"\nTotal reviews collected: {len(reviews)}")
    
    if reviews:
        # Save results
        scraper.save_to_json(reviews)
        scraper.save_to_csv(reviews)
        
        # Show sample
        print("\nSample review:")
        for key, value in reviews[0].items():
            if value:  # Only show non-empty values
                print(f"{key}: {value}")
    else:
        print("No reviews found. This might be due to:")
        print("1. Age verification blocking access")
        print("2. Site structure changes")
        print("3. Anti-scraping measures")
        print("4. JavaScript-heavy content loading")

if __name__ == "__main__":
    main()