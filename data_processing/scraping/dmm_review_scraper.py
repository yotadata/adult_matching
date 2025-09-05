import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import time
from urllib.parse import urljoin, urlparse
import csv
from datetime import datetime

class DMMReviewScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_reviews_from_page(self, url: str) -> List[Dict[str, Any]]:
        """Extract reviews from a single page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            print(f"Response status: {response.status_code}")
            print(f"Content length: {len(response.content)}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Save HTML to file for inspection
            with open('debug_page.html', 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            print("Saved page HTML to debug_page.html for inspection")
            
            # Find script tags containing review data
            reviews = []
            script_tags = soup.find_all('script')
            print(f"Found {len(script_tags)} script tags")
            
            for i, script in enumerate(script_tags):
                if script.string:
                    script_content = script.string
                    # Look for various patterns that might contain review data
                    patterns = [
                        r'reviewList["\']?\s*:\s*(\[.*?\])',
                        r'reviews["\']?\s*:\s*(\[.*?\])',  
                        r'data["\']?\s*:\s*({.*?reviewList.*?})',
                        r'window\.__INITIAL_STATE__\s*=\s*({.*?})',
                        r'__NEXT_DATA__\s*=\s*({.*?})'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, script_content, re.DOTALL)
                        if matches:
                            print(f"Found potential review data in script {i} with pattern: {pattern}")
                            for match in matches:
                                try:
                                    if match.startswith('['):
                                        review_data = json.loads(match)
                                    else:
                                        data = json.loads(match)
                                        # Look for review data in nested structure
                                        review_data = self.find_reviews_in_data(data)
                                    
                                    if isinstance(review_data, list):
                                        for review in review_data:
                                            processed_review = self.process_review(review)
                                            if processed_review:
                                                reviews.append(processed_review)
                                    
                                except json.JSONDecodeError as e:
                                    print(f"JSON decode error: {e}")
                                    continue
            
            # Fallback: try to extract from HTML elements directly
            if not reviews:
                print("No reviews found in scripts, trying HTML extraction...")
                reviews = self.extract_from_html_elements(soup)
                
            return reviews
            
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []
    
    def find_reviews_in_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursively find review data in nested structures"""
        reviews = []
        
        if isinstance(data, dict):
            # Check for common review list keys
            review_keys = ['reviewList', 'reviews', 'items', 'data', 'results']
            for key in review_keys:
                if key in data and isinstance(data[key], list):
                    reviews.extend(data[key])
            
            # Recursively search nested structures
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    nested_reviews = self.find_reviews_in_data(value)
                    reviews.extend(nested_reviews)
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    nested_reviews = self.find_reviews_in_data(item)
                    reviews.extend(nested_reviews)
        
        return reviews
    
    def extract_from_html_elements(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Fallback method to extract reviews from HTML elements"""
        reviews = []
        
        print("Attempting HTML extraction...")
        
        # Try multiple strategies to find review containers
        selectors = [
            ['div', 'section', 'article'],  # Common container elements
            ['li'],  # List items
            ['tr'],  # Table rows
        ]
        
        class_patterns = [
            r'review',
            r'item',  
            r'comment',
            r'card',
            r'post',
            r'content',
            r'css-.*'  # CSS modules
        ]
        
        # Try different combinations of selectors and class patterns
        for elements in selectors:
            for pattern in class_patterns:
                review_elements = soup.find_all(elements, class_=re.compile(pattern, re.I))
                print(f"Found {len(review_elements)} elements with {elements} and class pattern {pattern}")
                
                for element in review_elements:
                    review_data = self.extract_review_from_element(element)
                    if review_data and review_data.get('review_text'):
                        reviews.append(review_data)
        
        # Remove duplicates based on review text
        unique_reviews = []
        seen_texts = set()
        for review in reviews:
            text = review.get('review_text', '').strip()
            if text and text not in seen_texts:
                unique_reviews.append(review)
                seen_texts.add(text)
        
        print(f"HTML extraction found {len(unique_reviews)} unique reviews")
        return unique_reviews
    
    def extract_review_from_element(self, element) -> Dict[str, Any]:
        """Extract review data from a single HTML element"""
        review_data = {}
        
        # Extract review text - try multiple approaches
        text_selectors = [
            lambda el: el.find(['div', 'p', 'span'], class_=re.compile(r'text|content|review|comment', re.I)),
            lambda el: el.find(['div', 'p', 'span'], string=re.compile(r'.{20,}', re.DOTALL)),  # Long text
            lambda el: el.find('p'),  # Simple paragraph
            lambda el: next((child for child in el.find_all(['div', 'p', 'span']) if child.get_text(strip=True) and len(child.get_text(strip=True)) > 20), None)
        ]
        
        for selector in text_selectors:
            try:
                text_element = selector(element)
                if text_element:
                    text = text_element.get_text(strip=True)
                    if len(text) > 10:  # Minimum length for review text
                        review_data['review_text'] = text
                        break
            except:
                continue
        
        # Extract rating - look for star ratings, numeric scores
        rating_selectors = [
            lambda el: el.find(['div', 'span'], class_=re.compile(r'rating|star|score', re.I)),
            lambda el: el.find(['div', 'span'], string=re.compile(r'[★☆⭐]|[0-5]点|[0-5]/5')),
        ]
        
        for selector in rating_selectors:
            try:
                rating_element = selector(element)
                if rating_element:
                    rating_text = rating_element.get_text(strip=True)
                    # Look for various rating patterns
                    patterns = [
                        r'(\d+(?:\.\d+)?)/5',  # X/5 format
                        r'(\d+(?:\.\d+)?)点',   # X点 format
                        r'([★⭐]{1,5})',      # Star symbols
                        r'(\d+(?:\.\d+)?)',    # Simple number
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, rating_text)
                        if match:
                            try:
                                if '★' in match.group(1) or '⭐' in match.group(1):
                                    review_data['rating'] = len(match.group(1))
                                else:
                                    review_data['rating'] = float(match.group(1))
                                break
                            except:
                                continue
                    if 'rating' in review_data:
                        break
            except:
                continue
        
        # Extract title
        title_selectors = [
            lambda el: el.find(['h1', 'h2', 'h3', 'h4', 'a'], class_=re.compile(r'title|name', re.I)),
            lambda el: el.find(['h1', 'h2', 'h3', 'h4']),
        ]
        
        for selector in title_selectors:
            try:
                title_element = selector(element)
                if title_element:
                    title = title_element.get_text(strip=True)
                    if title:
                        review_data['content_title'] = title
                        break
            except:
                continue
        
        # Extract date
        date_selectors = [
            lambda el: el.find(['time', 'span', 'div'], class_=re.compile(r'date|time', re.I)),
            lambda el: el.find('time'),
            lambda el: el.find(string=re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}')),
        ]
        
        for selector in date_selectors:
            try:
                date_element = selector(element)
                if date_element:
                    if hasattr(date_element, 'get_text'):
                        date_text = date_element.get_text(strip=True)
                    else:
                        date_text = str(date_element).strip()
                    
                    if date_text:
                        review_data['write_date'] = date_text
                        break
            except:
                continue
        
        return review_data
    
    def process_review(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean review data"""
        processed = {
            'content_id': review_data.get('contentId', ''),
            'content_title': review_data.get('contentTitle', ''),
            'review_text': review_data.get('text', ''),
            'rating': self.extract_rating(review_data),
            'evaluate_count': review_data.get('evaluateCount', 0),
            'shop_name': review_data.get('displayShopName', ''),
            'write_date': review_data.get('writeDate', ''),
            'reviewer_id': review_data.get('reviewerId', ''),
            'review_id': review_data.get('reviewId', '')
        }
        
        # Clean text
        if processed['review_text']:
            processed['review_text'] = processed['review_text'].strip()
        
        return processed
    
    def extract_rating(self, review_data: Dict[str, Any]) -> float:
        """Extract rating from review data"""
        # Look for various rating fields
        rating_fields = ['rating', 'score', 'evaluate', 'star']
        
        for field in rating_fields:
            if field in review_data:
                try:
                    return float(review_data[field])
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def scrape_multiple_pages(self, base_url: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Scrape multiple pages of reviews"""
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            # Construct URL with page parameter
            if '?' in base_url:
                url = f"{base_url}&page={page}" if 'page=' not in base_url else base_url.replace(f'page={page-1}', f'page={page}')
            else:
                url = f"{base_url}?page={page}"
            
            print(f"Scraping page {page}: {url}")
            
            reviews = self.extract_reviews_from_page(url)
            if not reviews:
                print(f"No reviews found on page {page}, stopping...")
                break
                
            all_reviews.extend(reviews)
            print(f"Found {len(reviews)} reviews on page {page}")
            
            # Be respectful with rate limiting
            time.sleep(2)
        
        return all_reviews
    
    def save_to_csv(self, reviews: List[Dict[str, Any]], filename: str = None):
        """Save reviews to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dmm_reviews_{timestamp}.csv"
        
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
    
    def save_to_json(self, reviews: List[Dict[str, Any]], filename: str = None):
        """Save reviews to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dmm_reviews_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(reviews, jsonfile, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(reviews)} reviews to {filename}")

def main():
    scraper = DMMReviewScraper()
    
    # Target URL
    url = "https://review.dmm.co.jp/review-front/reviewer/list/185585?page=1"
    
    print("Starting DMM review scraping...")
    print(f"Target URL: {url}")
    
    # First, try single page
    reviews = scraper.extract_reviews_from_page(url)
    print(f"Found {len(reviews)} reviews on first page")
    
    # If successful, scrape multiple pages
    if reviews:
        print("Scraping multiple pages...")
        all_reviews = scraper.scrape_multiple_pages(url, max_pages=5)
        print(f"Total reviews collected: {len(all_reviews)}")
        
        # Save results
        scraper.save_to_json(all_reviews)
        scraper.save_to_csv(all_reviews)
        
        # Show sample
        if all_reviews:
            print("\nSample review:")
            sample = all_reviews[0]
            for key, value in sample.items():
                print(f"{key}: {value}")
    else:
        print("No reviews found. Please check the URL or site structure.")

if __name__ == "__main__":
    main()