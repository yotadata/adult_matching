#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scrape DMM Review top-100 reviewers' reviews and export CSV.

Outputs columns: product_url, reviewer_id, stars

Assumptions and notes:
- Ranking source: https://review.dmm.co.jp/review-front/ranking/<period>
- Each reviewer entry links to a reviewer review-list URL that includes the reviewer ID
    (e.g., contains reviewer_id=12345 or id=12345). We extract the numeric/slug ID from URL.
- Each review list page contains product links whose URL includes a CID (e.g., cid=abcd12345).
- Star ratings are extracted by common patterns (data-rate, active star icons, or text like "星5").
- The script follows pagination for each reviewer until no next page is found.

Usage:
    python scripts/scrape_dmm_reviews/scrape_dmm_reviews.py \
        --output ml/data/raw/reviews/dmm_reviews.csv \
        --max-reviewers 100 \
        --delay 1.2 \
        --timeout 20 \
        --verbose

Dependencies: requests, beautifulsoup4, lxml, tqdm
    pip install -r scripts/scrape_dmm_reviews/requirements.txt

Caveats:
- Site structure may change. This script uses heuristic selectors and regexes to be resilient,
    but you may need to tweak selectors if DMM updates markup.
- Respect site terms and robots. Add longer delays if needed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


RANKING_PERIOD_PATHS = {
    "week": "1week",
    "month": "1month",
    "3month": "3month",
    "halfyear": "6month",
    "year": "1year",
    "all": "total",
}
DEFAULT_RANKING_PERIODS = ["year"]
AGE_CHECK_COOKIE_NAME = "age_check_done"
AGE_CHECK_COOKIE_VALUE = "1"
MAX_MISSING_STAR_LOGS = 50
_missing_star_logs = 0


@dataclass
class ReviewerLink:
    reviewer_id: str
    url: str
    name: Optional[str] = None


def create_session(timeout: int) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
            # Some DMM endpoints are sensitive to Referer; set a sane default.
            "Referer": "https://www.dmm.co.jp/",
        }
    )

    # Attach a default timeout to the session via a wrapper
    original_request = s.request

    def request_with_timeout(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return original_request(method, url, **kwargs)

    s.request = request_with_timeout  # type: ignore
    return s


def ensure_age_check_cookie(session: requests.Session) -> None:
    """Pre-set the DMM age-check cookie so ranking/review pages render.

    DMM redirects to an age-check gate if this cookie is absent. Setting it on
    the parent domain ensures subdomains like review.dmm.co.jp receive it.
    """
    # Set on base domain and common subdomains to be safe.
    for domain in [".dmm.co.jp", "www.dmm.co.jp", "review.dmm.co.jp"]:
        session.cookies.set(AGE_CHECK_COOKIE_NAME, AGE_CHECK_COOKIE_VALUE, domain=domain, path="/")
    # Some pages also check cookie-consent; this helps avoid interstitials.
    session.cookies.set("ckcy", "1", domain=".dmm.co.jp", path="/")


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    logging.debug(f"GET {url}")
    resp = session.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def dump_html(output_dir: Optional[str], filename: str, soup_or_text) -> Optional[str]:
    if not output_dir:
        return None
    try:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        path = p / filename
        text = soup_or_text.prettify() if hasattr(soup_or_text, "prettify") else str(soup_or_text)
        path.write_text(text, encoding="utf-8")
        logging.debug(f"Dumped HTML to {path}")
        return str(path)
    except Exception as e:
        logging.debug(f"Failed to dump HTML {filename}: {e}")
        return None


def parse_total_reviews_from_reviewer_page(soup: BeautifulSoup) -> Optional[int]:
    """Try to extract the total review count for a reviewer from the page.

    Looks for common Japanese patterns like "全123件", "123件のレビュー", or elements with
    count-like classes or attributes.
    """
    text = soup.get_text(" ", strip=True)
    # Prefer explicit レビュー counts; avoid 参考になった系
    for pat in [
        r"レビュー[^0-9]{0,6}([0-9,]+)\s*件",
        r"([0-9,]+)\s*件のレビュー",
        r"レビュー[^0-9]*期間中[^0-9]*[0-9,]+[^0-9]*累計\s*([0-9,]+)",
    ]:
        m = re.search(pat, text)
        if m:
            try:
                val = int(m.group(1).replace(",", ""))
                if val <= 200000:
                    return val
            except Exception:
                pass

    # Attribute-based counts
    for el in soup.find_all(True):
        for attr in ("data-total", "data-count", "data-review-count", "aria-setsize"):
            v = el.get(attr)
            if v and re.fullmatch(r"\d+", v):
                try:
                    val = int(v)
                    if val <= 200000:
                        return val
                except Exception:
                    pass
    return None


def extract_reviewer_id_from_url(url: str) -> Optional[str]:
    # Try typical query params first
    m = re.search(r"(?:reviewer_id|user_id|uid|id)=([0-9A-Za-z_-]+)", url)
    if m:
        return m.group(1)
    # DMM often encodes params in path using /=/param=value/ style
    m = re.search(r"/=/(?:reviewer_id|user_id|uid|id)=([0-9A-Za-z_-]+)/?", url)
    if m:
        return m.group(1)
    # Known path shapes
    # - /review-front/user/123456
    m = re.search(r"/review-front/user/([0-9A-Za-z_-]{3,})/?", url)
    if m:
        return m.group(1)
    # - /reviewer/123456 or /reviewers/...
    m = re.search(r"/reviewer(?:s)?/(?:-/)?(?:.*/)?([0-9A-Za-z_-]{3,})/?", url)
    if m:
        return m.group(1)
    return None


def parse_ranking_for_reviewers(soup: BeautifulSoup) -> List[ReviewerLink]:
    links: List[ReviewerLink] = []
    seen: set[str] = set()
    # Collect candidate anchors on the ranking page
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Normalize to absolute when needed
        if href.startswith("/"):
            href_abs = f"https://review.dmm.co.jp{href}"
        elif href.startswith("http"):
            href_abs = href
        else:
            continue

        # Heuristics: reviewer-related links. Accept common shapes.
        if not (
            re.search(r"reviewer|/review-front/user/|/=/(?:reviewer_id|user_id|uid|id)=", href_abs)
        ):
            continue

        reviewer_id = extract_reviewer_id_from_url(href_abs)
        if not reviewer_id:
            continue

        if reviewer_id in seen:
            continue
        seen.add(reviewer_id)

        name = a.get_text(strip=True) or None
        links.append(ReviewerLink(reviewer_id=reviewer_id, url=href_abs, name=name))

    logging.info(f"Found {len(links)} reviewer links on ranking page")
    return links


def _abs_review_url(href: str) -> str:
    return href if href.startswith("http") else f"https://review.dmm.co.jp{href}"


def _parse_current_page_from_url(url: str) -> Optional[int]:
    m = re.search(r"[?&](?:page|p|pageNum|pagenum)=([0-9]+)", url)
    if m:
        return int(m.group(1))
    m = re.search(r"/=/(?:page|p|pageno|pagenum)=([0-9]+)/?", url)
    if m:
        return int(m.group(1))
    m = re.search(r"/(?:page|p)=([0-9]+)/?", url)
    if m:
        return int(m.group(1))
    return None


def _is_reviewer_list_url(url: str) -> bool:
    return bool(re.search(r"review-front/reviewer/list", url)) or bool(
        re.search(r"[?&/](?:page|p|pagenum|pageNum)=\d+", url)
    )


def find_next_page_url(soup: BeautifulSoup, current_url: Optional[str] = None) -> Optional[str]:
    # 1) rel=next
    a = soup.find("a", attrs={"rel": "next"})
    if a and a.has_attr("href") and a["href"] != "#":
        candidate = _abs_review_url(a["href"])
        if _is_reviewer_list_url(candidate):
            return candidate

    # 2) aria-label or text indicating next
    for a in soup.find_all("a", href=True):
        aria = a.get("aria-label", "")
        txt = (a.get_text() or "").strip()
        classes = " ".join(a.get("class", []))
        if (
            re.search(r"次(へ|のページ)?|Next|›|»|＞", aria or txt)
            or re.search(r"\b(next|pagination__next|pager-next|arrow-next)\b", classes)
        ):
            href = a["href"]
            if href and href != "#":
                candidate = _abs_review_url(href)
                if _is_reviewer_list_url(candidate):
                    return candidate

    # 3) Numeric pagination: try to infer next number relative to current
    current_num = _parse_current_page_from_url(current_url or "") if current_url else None

    # Prefer within a pagination container
    containers = soup.find_all(class_=re.compile(r"pagination|pager|pagenation|page"))
    containers = containers or [soup]  # fallback to whole document
    for cont in containers:
        page_nodes = cont.find_all(["a", "span", "li"], recursive=True)
        # Try to locate current index by an element with current-ish class
        current_idx = None
        for i, el in enumerate(page_nodes):
            classes = " ".join(el.get("class", []))
            if re.search(r"\b(current|is-active|active|is-current|on)\b", classes):
                current_idx = i
                break
        # If not found, use current page number from URL and find matching anchor
        if current_idx is None and current_num is not None:
            for i, el in enumerate(page_nodes):
                txt = (el.get_text() or "").strip()
                if txt.isdigit() and int(txt) == current_num:
                    current_idx = i
                    break
        # If we have a current, pick the next anchor after it
        if current_idx is not None:
            for el in page_nodes[current_idx + 1 :]:
                if el.name == "a" and el.has_attr("href"):
                    href = el["href"]
                    if href and href != "#":
                        candidate = _abs_review_url(href)
                        if _is_reviewer_list_url(candidate):
                            return candidate

    # 4) As a last resort: if we know current number, look for anchor with number+1
    if current_num is not None:
        target = str(current_num + 1)
        for a in soup.find_all("a", href=True):
            if (a.get_text() or "").strip() == target:
                candidate = _abs_review_url(a["href"])
                if _is_reviewer_list_url(candidate):
                    return candidate

    return None


def extract_cid_from_url(url: str) -> Optional[str]:
    # Typical DMM format: /- /detail/=/cid=abcd12345/
    m = re.search(r"[?&/]cid=([^/?&#]+)", url)
    if m:
        return m.group(1)
    # Video domain format: .../av/content/?id=abcd12345
    if "video.dmm.co.jp" in url or "/av/content/" in url:
        m = re.search(r"[?&]id=([^/?&#]+)", url)
        if m:
            return m.group(1)
    # Rare variants as fallback
    m = re.search(r"/(cid[^/]+)/?", url)
    if m:
        return m.group(1)
    return None


def is_target_review_url(url: str) -> bool:
    if not url:
        return False
    return (
        ("video.dmm.co.jp" in url and "/av/" in url)
        or ("dmm.co.jp" in url and "/digital/videoa/-/detail/=/cid=" in url)
    )


def find_video_av_url_in_block(block) -> Optional[str]:
    # Returns the first matching target review URL (video.dmm /av/ or www.dmm.co.jp/digital/videoa/-/detail)
    for a in block.find_all("a", href=True):
        href = a["href"]
        # Absolute-ize when necessary
        if href.startswith("/") and "/digital/videoa/-/detail/=/cid=" in href:
            href_abs = f"https://www.dmm.co.jp{href}"
        elif href.startswith("/") and "/av/" in href:
            href_abs = f"https://video.dmm.co.jp{href}"
        else:
            href_abs = href

        if is_target_review_url(href_abs):
            return href_abs
    return None


def build_video_av_url_from_cid(cid: str) -> str:
    return f"https://video.dmm.co.jp/av/content/?id={cid}"


def build_digital_videoa_detail_url_from_cid(cid: str) -> str:
    return f"https://www.dmm.co.jp/digital/videoa/-/detail/=/cid={cid}/"


def verify_av_url(session: requests.Session, url: str, timeout: int = 10) -> bool:
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return False
        # Accept if the final URL is still a target review URL
        return is_target_review_url(resp.url) or is_target_review_url(url)
    except Exception:
        return False


def parse_star_from_block(block) -> Optional[int]:
    # 1) data-* attributes
    for attr in ("data-rate", "data-rating", "data-score", "data-star"):
        val = block.get(attr)
        if val and re.fullmatch(r"[1-5]", str(val)):
            return int(val)

    # 1.1) itemprop / microdata
    meta = block.find(attrs={"itemprop": "ratingValue"})
    if meta:
        # Could be <meta itemprop="ratingValue" content="5">
        content = meta.get("content") or meta.get_text(strip=True)
        if content and re.fullmatch(r"[1-5](?:\.0+)?", content):
            return int(float(content))

    # 1.2) aria-label patterns e.g., "5点", "星4つ", "4/5"
    for el in [block] + block.find_all(True):
        aria = el.get("aria-label")
        if not aria:
            continue
        m = re.search(r"星\s*([1-5])|([1-5])\s*/\s*5|([1-5])\s*点", aria)
        if m:
            for g in m.groups():
                if g and re.fullmatch(r"[1-5]", g):
                    return int(g)

    # 1.3) class-derived scores e.g., is-5, rate-4, rating-3
    cls_text = " ".join(block.get("class", []))
    m = re.search(r"\b(?:is|rate|score|rating|stars?)-([1-5])\b", cls_text)
    if m:
        return int(m.group(1))

    # 1.4) width-based star bars e.g., style="width: 80%" => 4
    for el in block.find_all(True):
        style = el.get("style", "")
        if "width" in style and "%" in style and re.search(r"rating|rate|star", " ".join(el.get("class", []))):
            m = re.search(r"width\s*:\s*([0-9]{1,3})%", style)
            if m:
                pct = int(m.group(1))
                if 0 <= pct <= 100:
                    stars = round(pct / 20)
                    if 1 <= stars <= 5:
                        return int(stars)

    # 2) Count active star icons
    # Common patterns: class contains 'star' and 'on'/'active'/'is-active'/'full'
    star_container = None
    for cls_pat in ("rating", "rate", "star", "c-review__rating", "review__evaluate"):
        star_container = block.find(class_=re.compile(cls_pat))
        if star_container:
            break

    def count_active(el) -> int:
        cnt = 0
        for i in el.find_all(["i", "span", "em"], class_=True):
            cls = " ".join(i.get("class", []))
            if re.search(r"\b(on|is-active|active|full|filled)\b", cls):
                cnt += 1
        return cnt

    if star_container:
        c = count_active(star_container)
        if 1 <= c <= 5:
            return c

    # 3) Text patterns: 星5, 5点, ★★★★☆, 4/5 etc.
    text = block.get_text(" ", strip=True)
    # Full-width star
    m = re.search(r"星\s*([1-5])", text)
    if m:
        return int(m.group(1))
    # Japanese points
    m = re.search(r"([1-5])\s*点", text)
    if m:
        return int(m.group(1))
    # Star characters count (e.g., ★★★★☆)
    filled = text.count("★")
    if 1 <= filled <= 5:
        return filled
    m = re.search(r"([1-5])\s*/\s*5", text)
    if m:
        return int(m.group(1))
    return None


def iter_review_items(soup: BeautifulSoup):
    # Treat each div.css-1kosv36 as a single review block
    for c in soup.select("div.css-1kosv36"):
        # Use the first anchor as the product URL source (no AV verification)
        prod_a = c.find("a", href=True)
        yield c, prod_a


def find_review_detail_link(block) -> Optional[str]:
    # Look for a link that seems to be a review detail rather than a product.
    for a in block.find_all("a", href=True):
        href = a["href"]
        if re.search(r"review-front/.*/detail|review-front/review|/review/-/detail", href):
            return href
    return None


def crawl_reviewer_reviews(
    session: requests.Session,
    reviewer: ReviewerLink,
    delay: float = 1.0,
    max_detail_fetch_per_reviewer: Optional[int] = 50,
    dump_html_dir: Optional[str] = None,
    dump_on_missing_star: bool = False,
    dump_limit_per_reviewer: int = 1,
    verify_av: bool = True,
) -> Tuple[List[Tuple[str, str, int]], Optional[int]]:
    base_url = f"https://review.dmm.co.jp/review-front/reviewer/list/{reviewer.reviewer_id}"
    rows: List[Tuple[str, str, int]] = []
    # No dedupe: collect all reviews as-is
    expected_total: Optional[int] = None

    detail_fetch_count = 0
    page_index = 0
    missing_star_dumped = 0
    while True:
        page_index += 1
        url = f"{base_url}?page={page_index}"
        soup = fetch_soup(session, url)

        # Capture expected total from the first page if available
        if expected_total is None:
            expected_total = parse_total_reviews_from_reviewer_page(soup)
            if expected_total is not None:
                logging.debug(f"Reviewer {reviewer.reviewer_id}: expected total {expected_total}")

        # For each review block, extract cid and star
        # No per-page dedupe
        items_on_page = 0
        for block, link in iter_review_items(soup):
            items_on_page += 1
            cid = None
            if link is not None:
                href = link.get("href", "")
                if href.startswith("/"):
                    href = f"https://www.dmm.co.jp{href}"
                cid = extract_cid_from_url(href)
            # Determine product URL (no filtering/verification): use the link href
            product_url = None
            if link is not None:
                href0 = link.get("href", "")
                product_url = (
                    f"https://www.dmm.co.jp{href0}" if href0.startswith("/") else href0
                )
            if not product_url:
                # As a last resort, find any anchor in the block
                a2 = block.find("a", href=True)
                if a2:
                    href2 = a2.get("href", "")
                    product_url = (
                        f"https://www.dmm.co.jp{href2}" if href2.startswith("/") else href2
                    )
            if not product_url:
                logging.debug(f"Skip review item without product URL (reviewer {reviewer.reviewer_id})")
                continue

            # Extract stars: prioritize Material UI rating in span.MuiRating-root
            stars_val = None
            mui = block.select_one('span.MuiRating-root')
            if mui:
                aria = mui.get('aria-label') or ''
                m = re.search(r'([1-5])', aria)
                if m:
                    stars_val = int(m.group(1))
                if stars_val is None:
                    filled = len(mui.select('.MuiRating-iconFilled'))
                    if 1 <= filled <= 5:
                        stars_val = filled
            if stars_val is None:
                stars_val = parse_star_from_block(block)
            if stars_val is None:
                # If still missing, optionally try detail page if allowed
                if max_detail_fetch_per_reviewer is None or detail_fetch_count < max_detail_fetch_per_reviewer:
                    detail_href = find_review_detail_link(block)
                    if detail_href:
                        detail_url = detail_href if detail_href.startswith("http") else _abs_review_url(detail_href)
                        try:
                            detail_fetch_count += 1
                            detail_soup = fetch_soup(session, detail_url)
                            stars_val = parse_star_from_block(detail_soup)
                        except Exception:
                            pass
                        finally:
                            time.sleep(delay)
            if stars_val is None:
                logging.debug(f"Star not found (reviewer {reviewer.reviewer_id}) url={product_url}")
                continue

            rows.append((product_url, reviewer.reviewer_id, int(stars_val)))

        # Pagination termination conditions
        # If we already met the expected total, we can stop early
        if expected_total is not None and len(rows) >= expected_total:
            break
        # If this page produced fewer than 20 review items, likely the last page
        if items_on_page == 0 or items_on_page < 20:
            break
        time.sleep(delay)

    return rows, expected_total


def write_csv(rows: Iterable[Tuple[str, str, int]], output_path: str) -> int:
    n = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_url", "reviewer_id", "stars"])
        for product_url, reviewer_id, stars in rows:
            writer.writerow([product_url, reviewer_id, stars])
            n += 1
    return n


def write_csv_header(output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_url", "reviewer_id", "stars"])


def append_csv_rows(output_path: str, rows: Iterable[Tuple[str, str, int]]) -> int:
    n = 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for product_url, reviewer_id, stars in rows:
            writer.writerow([product_url, reviewer_id, stars])
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Scrape DMM Review top-100 reviewers' reviews → CSV")
    parser.add_argument("--output", required=True, help="Output CSV path (e.g., ml/data/raw/reviews/dmm_reviews.csv)")
    parser.add_argument(
        "--ranking-period",
        action="append",
        choices=sorted(RANKING_PERIOD_PATHS.keys()),
        help=(
            "Ranking period(s) to crawl (e.g., year, month, all). "
            "Can be specified multiple times; defaults to year only."
        ),
    )
    parser.add_argument("--max-reviewers", type=int, default=100, help="Max reviewers to process (default: 100)")
    parser.add_argument("--delay", type=float, default=1.2, help="Delay seconds between page requests (default: 1.2)")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP request timeout seconds (default: 20)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--only-reviewers",
        nargs="+",
        default=None,
        help=(
            "Process only these reviewer IDs (space- or comma-separated). "
            "When set, skips ranking fetch and crawls only specified reviewers."
        ),
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling reviewers (by default reviewers are shuffled)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional seed for reviewer shuffling (default: random)",
    )
    parser.add_argument(
        "--max-detail-fetch-per-reviewer",
        type=int,
        default=50,
        help="Max times to fetch review-detail pages per reviewer when star missing (default: 50). Set 0 to disable.",
    )
    parser.add_argument(
        "--dump-html-dir",
        default=None,
        help="Directory to dump HTML for debugging (e.g., ml/data/dumps).",
    )
    parser.add_argument(
        "--dump-on-missing-star",
        action="store_true",
        help="Dump the reviewer list page HTML when a missing star is encountered (limited per reviewer).",
    )
    parser.add_argument(
        "--dump-limit-per-reviewer",
        type=int,
        default=1,
        help="Max number of HTML dumps per reviewer (default: 1).",
    )
    parser.add_argument(
        "--dump-on-count-mismatch",
        action="store_true",
        help="After finishing a reviewer, if collected rows != expected count, dump the first reviewer page HTML.",
    )
    parser.add_argument(
        "--no-verify-av-url",
        action="store_true",
        help="Do not verify constructed /av/ URLs with a network request; rely on CID mapping only.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    session = create_session(timeout=args.timeout)
    ensure_age_check_cookie(session)

    # Build reviewer list
    reviewers: List[ReviewerLink] = []
    if args.only_reviewers:
        # Normalize IDs from args (accept comma-separated or space-separated)
        raw_ids: List[str] = []
        for token in args.only_reviewers:
            raw_ids.extend([t for t in token.split(",") if t])
        ids_set = []
        seen_id: set[str] = set()
        for rid in raw_ids:
            rid_norm = rid.strip()
            if not rid_norm or rid_norm in seen_id:
                continue
            seen_id.add(rid_norm)
            reviewers.append(
                ReviewerLink(
                    reviewer_id=rid_norm,
                    url=f"https://review.dmm.co.jp/review-front/reviewer/list/{rid_norm}",
                    name=None,
                )
            )
        if not reviewers:
            logging.error("No valid reviewer IDs provided to --only-reviewers")
            return
        logging.info(f"Processing only specified reviewer(s): {', '.join([r.reviewer_id for r in reviewers])}")
    else:
        period_keys = args.ranking_period or DEFAULT_RANKING_PERIODS
        reviewer_map: Dict[str, ReviewerLink] = {}
        for period in period_keys:
            period_path = RANKING_PERIOD_PATHS[period]
            ranking_url = f"https://review.dmm.co.jp/review-front/ranking/{period_path}"
            try:
                soup = fetch_soup(session, ranking_url)
                period_reviewers = parse_ranking_for_reviewers(soup)
                for reviewer in period_reviewers:
                    reviewer_map.setdefault(reviewer.reviewer_id, reviewer)
                logging.info(f"Fetched {len(period_reviewers)} reviewers from {ranking_url}")
            except Exception as exc:
                logging.error(f"Failed to fetch ranking for period '{period}' ({ranking_url}): {exc}")
        reviewers = list(reviewer_map.values())

        if not reviewers:
            logging.warning("No reviewers found on ranking page. The page structure may have changed.")

        # Shuffle reviewers to avoid always starting from the largest-volume reviewer
        if not args.no_shuffle:
            if args.shuffle_seed is not None:
                random.Random(args.shuffle_seed).shuffle(reviewers)
            else:
                random.shuffle(reviewers)
            logging.info("Shuffled reviewers before processing")

        reviewers = reviewers[: args.max_reviewers]
    logging.info(f"Processing {len(reviewers)} reviewer(s)")

    # 2) Crawl per reviewer and append rows to a single CSV incrementally
    write_csv_header(args.output)
    total_written = 0
    for idx, reviewer in enumerate(reviewers, start=1):
        logging.info(f"[{idx}/{len(reviewers)}] Reviewer {reviewer.reviewer_id} - {reviewer.name or ''}")
        try:
            mdf = None if args.max_detail_fetch_per_reviewer < 0 else args.max_detail_fetch_per_reviewer
            rows, expected = crawl_reviewer_reviews(
                session,
                reviewer,
                delay=args.delay,
                max_detail_fetch_per_reviewer=mdf,
                dump_html_dir=args.dump_html_dir,
                dump_on_missing_star=args.dump_on_missing_star,
                dump_limit_per_reviewer=args.dump_limit_per_reviewer,
                verify_av=not args.no_verify_av_url,
            )
            got = len(rows)
            if expected is not None and got != expected:
                # Page's total likely counts all categories, but we filter to /av/ only.
                logging.info(
                    f"Reviewer {reviewer.reviewer_id}: collected {got} AV reviews; page shows total {expected} across all content."
                )
                if args.dump_on_count_mismatch and args.dump_html_dir:
                    # Try dumping the canonical first page of this reviewer
                    first_url = f"https://review.dmm.co.jp/review-front/reviewer/list/{reviewer.reviewer_id}"
                    try:
                        first_soup = fetch_soup(session, first_url)
                        dump_html(args.dump_html_dir, f"reviewer_{reviewer.reviewer_id}_first_page_mismatch.html", first_soup)
                    except Exception:
                        pass
            written = append_csv_rows(args.output, rows)
            total_written += written
        except requests.HTTPError as e:
            logging.error(f"HTTP error for reviewer {reviewer.reviewer_id}: {e}")
        except Exception as e:
            logging.exception(f"Error for reviewer {reviewer.reviewer_id}: {e}")
        finally:
            time.sleep(args.delay)
    logging.info(f"Done. Wrote {total_written} rows to {args.output}")


if __name__ == "__main__":
    main()
