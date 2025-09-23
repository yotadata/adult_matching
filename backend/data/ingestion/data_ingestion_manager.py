"""
Data Ingestion Manager

統一データ取り込み管理システム
- API データ取り込み (DMM/FANZA)
- ウェブスクレイピングデータ統合
- ファイルベースデータインポート
- リアルタイムデータストリーミング
"""

import asyncio
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from enum import Enum

from backend.ml.utils.logger import get_ml_logger

logger = get_ml_logger(__name__)

class DataSourceType(Enum):
    """データソースタイプ"""
    API = "api"
    SCRAPING = "scraping"
    FILE = "file"
    DATABASE = "database"
    STREAMING = "streaming"

class IngestionStatus(Enum):
    """取り込み状態"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class IngestionTask:
    """取り込みタスク"""
    task_id: str
    source_type: DataSourceType
    source_config: Dict[str, Any]
    target_path: Path
    status: IngestionStatus = IngestionStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    records_processed: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class IngestionResult:
    """取り込み結果"""
    task_id: str
    success: bool
    records_ingested: int
    output_files: List[Path]
    processing_time_seconds: float
    data_quality_score: float
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = None

class DataIngestionManager:
    """データ取り込み管理システム"""
    
    def __init__(self, sources: List[str] = None, output_dir: Path = None):
        self.sources = sources or ["api", "scraping", "file"]
        self.output_dir = output_dir or Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_tasks: Dict[str, IngestionTask] = {}
        self.completed_tasks: Dict[str, IngestionTask] = {}
        
        # データ品質管理
        self.quality_thresholds = {
            "min_records": 10,
            "max_error_rate": 0.05,
            "required_fields": ["id", "title"],
            "max_processing_time_hours": 24
        }
        
    async def ingest_dmm_api_data(
        self,
        api_config: Dict[str, Any],
        batch_size: int = 100,
        max_pages: int = None
    ) -> IngestionResult:
        """DMM APIデータの取り込み"""
        task_id = f"dmm_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = IngestionTask(
            task_id=task_id,
            source_type=DataSourceType.API,
            source_config=api_config,
            target_path=self.output_dir / f"{task_id}.json",
            metadata={
                "batch_size": batch_size,
                "max_pages": max_pages,
                "api_endpoint": "dmm_api"
            }
        )
        
        return await self._execute_ingestion_task(task)
    
    async def ingest_scraping_data(
        self,
        scraping_config: Dict[str, Any],
        data_source_path: Path
    ) -> IngestionResult:
        """スクレイピングデータの取り込み"""
        task_id = f"scraping_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = IngestionTask(
            task_id=task_id,
            source_type=DataSourceType.SCRAPING,
            source_config=scraping_config,
            target_path=self.output_dir / f"{task_id}.json",
            metadata={
                "source_path": str(data_source_path),
                "scraping_type": scraping_config.get("type", "reviews")
            }
        )
        
        return await self._execute_ingestion_task(task)
    
    async def ingest_file_data(
        self,
        file_path: Path,
        file_format: str = "auto",
        processing_options: Dict[str, Any] = None
    ) -> IngestionResult:
        """ファイルデータの取り込み"""
        task_id = f"file_{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = IngestionTask(
            task_id=task_id,
            source_type=DataSourceType.FILE,
            source_config={
                "file_path": str(file_path),
                "format": file_format,
                "options": processing_options or {}
            },
            target_path=self.output_dir / f"{task_id}.json",
            metadata={
                "original_file": file_path.name,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
            }
        )
        
        return await self._execute_ingestion_task(task)
    
    async def _execute_ingestion_task(self, task: IngestionTask) -> IngestionResult:
        """取り込みタスクの実行"""
        logger.info(f"Starting ingestion task: {task.task_id}")
        
        start_time = datetime.now()
        task.started_at = start_time
        task.status = IngestionStatus.IN_PROGRESS
        self.active_tasks[task.task_id] = task
        
        try:
            # データソースタイプ別の処理
            if task.source_type == DataSourceType.API:
                result_data = await self._process_api_data(task)
            elif task.source_type == DataSourceType.SCRAPING:
                result_data = await self._process_scraping_data(task)
            elif task.source_type == DataSourceType.FILE:
                result_data = await self._process_file_data(task)
            else:
                raise ValueError(f"Unsupported source type: {task.source_type}")
            
            # データ品質検証
            quality_score = self._assess_data_quality(result_data)
            
            # 結果の保存
            output_files = await self._save_ingested_data(task, result_data)
            
            # タスク完了
            end_time = datetime.now()
            task.completed_at = end_time
            task.status = IngestionStatus.COMPLETED
            task.records_processed = len(result_data) if isinstance(result_data, list) else 1
            
            # アクティブタスクから完了タスクに移動
            del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Ingestion task {task.task_id} completed successfully")
            
            return IngestionResult(
                task_id=task.task_id,
                success=True,
                records_ingested=task.records_processed,
                output_files=output_files,
                processing_time_seconds=processing_time,
                data_quality_score=quality_score,
                metadata={
                    "source_type": task.source_type.value,
                    "started_at": start_time.isoformat(),
                    "completed_at": end_time.isoformat()
                }
            )
            
        except Exception as error:
            logger.error(f"Ingestion task {task.task_id} failed: {error}")
            
            task.status = IngestionStatus.FAILED
            task.error_message = str(error)
            task.completed_at = datetime.now()
            
            # 失敗タスクも完了タスクに移動
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            return IngestionResult(
                task_id=task.task_id,
                success=False,
                records_ingested=0,
                output_files=[],
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                data_quality_score=0.0,
                error_details=str(error)
            )
    
    async def _process_api_data(self, task: IngestionTask) -> List[Dict[str, Any]]:
        """APIデータの処理"""
        config = task.source_config
        logger.info(f"Processing API data with config: {config}")
        
        # DMM API統合（既存のスクリプトを活用）
        try:
            # 実際のAPI呼び出しはここで実装
            # 現在は模擬データを返す
            mock_data = [
                {
                    "id": f"api_item_{i}",
                    "title": f"API Item {i}",
                    "genre": "API Genre",
                    "maker": "API Maker",
                    "price": 1000 + i * 100,
                    "description": f"API Description for item {i}",
                    "ingested_at": datetime.now().isoformat()
                }
                for i in range(1, 51)  # 50件のサンプルデータ
            ]
            
            logger.info(f"Retrieved {len(mock_data)} records from API")
            return mock_data
            
        except Exception as error:
            logger.error(f"API data processing failed: {error}")
            raise
    
    async def _process_scraping_data(self, task: IngestionTask) -> List[Dict[str, Any]]:
        """スクレイピングデータの処理"""
        config = task.source_config
        source_path = Path(config.get("source_path", ""))
        
        logger.info(f"Processing scraping data from: {source_path}")
        
        if not source_path.exists():
            raise FileNotFoundError(f"Scraping data source not found: {source_path}")
        
        try:
            # 既存のスクレイピングデータの読み込み
            with open(source_path, 'r', encoding='utf-8') as f:
                scraping_data = json.load(f)
            
            # データの標準化
            standardized_data = []
            for item in scraping_data:
                standardized_item = {
                    "id": item.get("content_id", f"scraped_{len(standardized_data)}"),
                    "title": item.get("title", "Unknown Title"),
                    "genre": item.get("genre", "Unknown Genre"),
                    "maker": item.get("maker", "Unknown Maker"),
                    "rating": item.get("rating", 0),
                    "review_count": item.get("review_count", 0),
                    "original_data": item,
                    "ingested_at": datetime.now().isoformat(),
                    "source": "scraping"
                }
                standardized_data.append(standardized_item)
            
            logger.info(f"Processed {len(standardized_data)} scraping records")
            return standardized_data
            
        except Exception as error:
            logger.error(f"Scraping data processing failed: {error}")
            raise
    
    async def _process_file_data(self, task: IngestionTask) -> List[Dict[str, Any]]:
        """ファイルデータの処理"""
        config = task.source_config
        file_path = Path(config["file_path"])
        file_format = config.get("format", "auto")
        
        logger.info(f"Processing file data: {file_path} (format: {file_format})")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # ファイル形式の自動検出
            if file_format == "auto":
                file_format = file_path.suffix.lower().lstrip('.')
            
            # ファイル形式別の読み込み
            if file_format in ["json", "jsonl"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_format == "jsonl":
                        file_data = [json.loads(line) for line in f]
                    else:
                        file_data = json.load(f)
            
            elif file_format == "csv":
                df = pd.read_csv(file_path)
                file_data = df.to_dict('records')
            
            elif file_format in ["parquet", "feather"]:
                if file_format == "parquet":
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_feather(file_path)
                file_data = df.to_dict('records')
            
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # データの標準化
            standardized_data = []
            for i, item in enumerate(file_data):
                if isinstance(item, dict):
                    standardized_item = {
                        **item,
                        "ingested_at": datetime.now().isoformat(),
                        "source": "file",
                        "source_file": file_path.name,
                        "record_index": i
                    }
                else:
                    standardized_item = {
                        "data": item,
                        "ingested_at": datetime.now().isoformat(),
                        "source": "file",
                        "source_file": file_path.name,
                        "record_index": i
                    }
                
                standardized_data.append(standardized_item)
            
            logger.info(f"Processed {len(standardized_data)} file records")
            return standardized_data
            
        except Exception as error:
            logger.error(f"File data processing failed: {error}")
            raise
    
    def _assess_data_quality(self, data: List[Dict[str, Any]]) -> float:
        """データ品質の評価"""
        if not data:
            return 0.0
        
        score_components = []
        
        # レコード数チェック
        records_score = min(len(data) / self.quality_thresholds["min_records"], 1.0)
        score_components.append(records_score)
        
        # 必須フィールドチェック
        required_fields = self.quality_thresholds["required_fields"]
        field_scores = []
        
        for record in data:
            record_score = sum(1 for field in required_fields if field in record and record[field]) / len(required_fields)
            field_scores.append(record_score)
        
        avg_field_score = sum(field_scores) / len(field_scores) if field_scores else 0.0
        score_components.append(avg_field_score)
        
        # 全体品質スコア
        overall_score = sum(score_components) / len(score_components)
        
        logger.info(f"Data quality assessment: {overall_score:.2f}")
        return overall_score
    
    async def _save_ingested_data(self, task: IngestionTask, data: List[Dict[str, Any]]) -> List[Path]:
        """取り込みデータの保存"""
        output_files = []
        
        # JSONファイルとして保存
        json_file = task.target_path
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        output_files.append(json_file)
        
        # メタデータファイルの保存
        metadata_file = json_file.with_suffix('.metadata.json')
        metadata = {
            "task": asdict(task),
            "record_count": len(data),
            "file_size_bytes": json_file.stat().st_size,
            "created_at": datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        output_files.append(metadata_file)
        
        logger.info(f"Saved ingested data to {len(output_files)} files")
        return output_files
    
    def get_task_status(self, task_id: str) -> Optional[IngestionTask]:
        """タスク状態の取得"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    def list_active_tasks(self) -> List[IngestionTask]:
        """アクティブタスクの一覧"""
        return list(self.active_tasks.values())
    
    def list_completed_tasks(self, limit: int = 50) -> List[IngestionTask]:
        """完了タスクの一覧"""
        tasks = list(self.completed_tasks.values())
        # 完了時間でソート（新しい順）
        tasks.sort(key=lambda t: t.completed_at or datetime.min, reverse=True)
        return tasks[:limit]
    
    async def cancel_task(self, task_id: str) -> bool:
        """タスクのキャンセル"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = IngestionStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # キャンセルタスクを完了タスクに移動
            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def get_ingestion_summary(self) -> Dict[str, Any]:
        """取り込み概要の取得"""
        active_count = len(self.active_tasks)
        completed_count = len(self.completed_tasks)
        
        # 完了タスクの統計
        successful_tasks = [t for t in self.completed_tasks.values() if t.status == IngestionStatus.COMPLETED]
        failed_tasks = [t for t in self.completed_tasks.values() if t.status == IngestionStatus.FAILED]
        
        total_records = sum(t.records_processed for t in successful_tasks)
        
        return {
            "active_tasks": active_count,
            "completed_tasks": completed_count,
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "total_records_ingested": total_records,
            "success_rate": len(successful_tasks) / completed_count if completed_count > 0 else 0.0,
            "data_sources": list(set(t.source_type.value for t in self.completed_tasks.values())),
            "last_ingestion": max(t.completed_at for t in self.completed_tasks.values() if t.completed_at) if successful_tasks else None
        }