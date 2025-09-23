"""
DMM Sync Manager

DMM API統合データ同期管理システム
監視・エラーハンドリング・レート制限を備えた統一同期サービス
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import os

from backend.utils.logger import get_logger
from backend.utils.database import get_supabase_client

logger = get_logger(__name__)

@dataclass
class DMMSyncConfig:
    """DMM同期設定"""
    api_id: str = "W63Kd4A4ym2DaycFcXSU"
    affiliate_id: str = "yotadata2-990"
    base_url: str = "https://api.dmm.com/affiliate/v3/ItemList"
    site: str = "FANZA"
    service: str = "digital"
    floor: str = "videoa"
    output: str = "json"
    
    # レート制限設定
    rate_limit_delay: float = 1.0  # 秒
    max_retries: int = 3
    timeout: int = 30
    
    # バッチ処理設定
    batch_size: int = 50
    max_pages: int = 100
    
    # 監視設定
    log_progress_interval: int = 10
    error_threshold: int = 5


@dataclass
class SyncResult:
    """同期結果"""
    success: bool = False
    total_items: int = 0
    new_items: int = 0
    updated_items: int = 0
    skipped_items: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class DMMSyncManager:
    """DMM API統合同期管理システム"""
    
    def __init__(self, config: Optional[DMMSyncConfig] = None):
        self.config = config or DMMSyncConfig()
        self.supabase = get_supabase_client()
        self.session: Optional[aiohttp.ClientSession] = None
        self.error_count = 0
        
    async def __aenter__(self):
        """非同期コンテキストマネージャーの開始"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーの終了"""
        if self.session:
            await self.session.close()
    
    async def fetch_dmm_page(self, page: int = 1, sort: str = "date") -> Dict[str, Any]:
        """
        DMM APIから1ページのデータを取得
        
        Args:
            page: ページ番号
            sort: ソート方式 ("date", "rank", "price")
            
        Returns:
            APIレスポンスデータ
        """
        params = {
            "api_id": self.config.api_id,
            "affiliate_id": self.config.affiliate_id,
            "site": self.config.site,
            "service": self.config.service,
            "floor": self.config.floor,
            "hits": self.config.batch_size,
            "offset": ((page - 1) * self.config.batch_size) + 1,
            "sort": sort,
            "output": self.config.output
        }
        
        logger.debug(f"🔍 Fetching DMM API data: page {page}, sort {sort}")
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(self.config.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("result", {}).get("items"):
                            logger.debug(f"✅ Retrieved {len(data['result']['items'])} items from page {page}")
                            return data
                        else:
                            logger.warning(f"No items in response for page {page}")
                            return {"result": {"items": []}}
                    else:
                        logger.error(f"HTTP {response.status} error for page {page}")
                        
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for page {page}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        raise Exception(f"Failed to fetch page {page} after {self.config.max_retries} attempts")
    
    def transform_dmm_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        DMM APIアイテムをデータベース形式に変換
        
        Args:
            item: DMM APIアイテム
            
        Returns:
            データベース挿入用辞書
        """
        try:
            # 基本情報
            transformed = {
                "external_id": item.get("content_id", ""),
                "source": "dmm",
                "title": item.get("title", ""),
                "description": item.get("title", ""),  # DMM APIではdescriptionが別フィールド
                "thumbnail_url": "",
                "maker": "",
                "genre": "",
                "price": 0,
                "sample_video_url": "",
                "image_urls": [],
                "performers": [],
                "tags": [],
                "release_date": None,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # 画像URL（最初の画像をサムネイルとして使用）
            if "imageURL" in item and item["imageURL"]:
                image_url = item["imageURL"].get("large") or item["imageURL"].get("small", "")
                transformed["thumbnail_url"] = image_url
                transformed["image_urls"] = [image_url] if image_url else []
            
            # メーカー情報
            if "iteminfo" in item and "maker" in item["iteminfo"]:
                makers = item["iteminfo"]["maker"]
                if isinstance(makers, list) and makers:
                    transformed["maker"] = makers[0].get("name", "")
            
            # ジャンル情報
            if "iteminfo" in item and "genre" in item["iteminfo"]:
                genres = item["iteminfo"]["genre"]
                if isinstance(genres, list) and genres:
                    transformed["genre"] = genres[0].get("name", "")
                    transformed["tags"] = [g.get("name", "") for g in genres if g.get("name")]
            
            # 出演者情報
            if "iteminfo" in item and "actress" in item["iteminfo"]:
                actresses = item["iteminfo"]["actress"]
                if isinstance(actresses, list):
                    transformed["performers"] = [a.get("name", "") for a in actresses if a.get("name")]
            
            # 価格情報
            if "prices" in item and item["prices"]:
                # 最初の価格を使用
                price_info = item["prices"][0] if isinstance(item["prices"], list) else item["prices"]
                price_str = price_info.get("price", "0").replace("円", "").replace(",", "")
                try:
                    transformed["price"] = int(price_str)
                except (ValueError, TypeError):
                    transformed["price"] = 0
            
            # リリース日
            if "date" in item:
                try:
                    # YYYY-MM-DD HH:MM:SS 形式を想定
                    release_date = datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S")
                    transformed["release_date"] = release_date.isoformat()
                except (ValueError, TypeError):
                    pass
            
            # サンプル動画URL
            if "sampleImageURL" in item and "sample_s" in item["sampleImageURL"]:
                sample_url = item["sampleImageURL"]["sample_s"].get("image", "")
                if sample_url:
                    transformed["sample_video_url"] = sample_url
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming DMM item {item.get('content_id', 'unknown')}: {e}")
            return None
    
    async def store_items(self, items: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """
        アイテムをデータベースに保存
        
        Args:
            items: 変換済みアイテムリスト
            
        Returns:
            (新規作成数, 更新数, スキップ数)
        """
        new_count = 0
        updated_count = 0
        skipped_count = 0
        
        for item in items:
            try:
                # 既存アイテムチェック
                existing = self.supabase.table("videos").select("id").eq("external_id", item["external_id"]).eq("source", "dmm").execute()
                
                if existing.data:
                    # 更新
                    result = self.supabase.table("videos").update(item).eq("external_id", item["external_id"]).eq("source", "dmm").execute()
                    updated_count += 1
                    logger.debug(f"Updated item: {item['external_id']}")
                else:
                    # 新規作成
                    result = self.supabase.table("videos").insert(item).execute()
                    new_count += 1
                    logger.debug(f"Created new item: {item['external_id']}")
                    
            except Exception as e:
                logger.error(f"Error storing item {item.get('external_id', 'unknown')}: {e}")
                skipped_count += 1
                
        return new_count, updated_count, skipped_count
    
    async def sync_dmm_data(
        self, 
        max_pages: Optional[int] = None,
        sort: str = "date",
        progress_callback: Optional[callable] = None
    ) -> SyncResult:
        """
        DMM データの完全同期
        
        Args:
            max_pages: 最大ページ数（Noneで設定値使用）
            sort: ソート方式
            progress_callback: 進捗コールバック関数
            
        Returns:
            同期結果
        """
        result = SyncResult()
        result.start_time = datetime.utcnow()
        
        max_pages = max_pages or self.config.max_pages
        
        try:
            logger.info(f"🚀 Starting DMM sync: max_pages={max_pages}, sort={sort}")
            
            for page in range(1, max_pages + 1):
                try:
                    # レート制限
                    if page > 1:
                        await asyncio.sleep(self.config.rate_limit_delay)
                    
                    # データ取得
                    api_data = await self.fetch_dmm_page(page, sort)
                    items = api_data.get("result", {}).get("items", [])
                    
                    if not items:
                        logger.info(f"No more items at page {page}, stopping sync")
                        break
                    
                    # データ変換
                    transformed_items = []
                    for item in items:
                        transformed = self.transform_dmm_item(item)
                        if transformed:
                            transformed_items.append(transformed)
                    
                    # データ保存
                    new, updated, skipped = await self.store_items(transformed_items)
                    
                    # 結果更新
                    result.total_items += len(items)
                    result.new_items += new
                    result.updated_items += updated
                    result.skipped_items += skipped
                    
                    # 進捗報告
                    if page % self.config.log_progress_interval == 0:
                        logger.info(f"📊 Progress: page {page}/{max_pages}, total: {result.total_items}, new: {result.new_items}, updated: {result.updated_items}")
                    
                    if progress_callback:
                        progress_callback(page, max_pages, result)
                    
                except Exception as e:
                    error_msg = f"Error processing page {page}: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    self.error_count += 1
                    
                    if self.error_count >= self.config.error_threshold:
                        logger.error(f"Too many errors ({self.error_count}), stopping sync")
                        break
            
            result.success = self.error_count < self.config.error_threshold
            
        except Exception as e:
            error_msg = f"Fatal sync error: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.success = False
        
        finally:
            result.end_time = datetime.utcnow()
            
            # 同期サマリー
            logger.info(f"🏁 DMM sync completed:")
            logger.info(f"   Success: {result.success}")
            logger.info(f"   Duration: {result.duration}")
            logger.info(f"   Total items: {result.total_items}")
            logger.info(f"   New: {result.new_items}, Updated: {result.updated_items}, Skipped: {result.skipped_items}")
            logger.info(f"   Errors: {len(result.errors)}")
        
        return result


# 便利関数
async def run_dmm_sync(config: Optional[DMMSyncConfig] = None, **kwargs) -> SyncResult:
    """
    DMM同期の実行（コンテキストマネージャー付き）
    
    Args:
        config: 同期設定
        **kwargs: sync_dmm_dataに渡す追加引数
        
    Returns:
        同期結果
    """
    async with DMMSyncManager(config) as sync_manager:
        return await sync_manager.sync_dmm_data(**kwargs)


if __name__ == "__main__":
    # テスト実行
    async def main():
        config = DMMSyncConfig(max_pages=2, batch_size=10)  # テスト用の小さな設定
        result = await run_dmm_sync(config)
        print(f"Sync result: {result}")
    
    asyncio.run(main())