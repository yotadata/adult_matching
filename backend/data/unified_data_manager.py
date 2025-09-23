"""
Unified Data Manager

データパッケージ統合管理システム
- 全データコンポーネントの統合インターフェース
- パフォーマンス監視とシステム状態管理
- エンドツーエンドデータフロー制御
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

from backend.ml.utils.logger import get_ml_logger
from backend.data.ingestion import DataIngestionManager
from backend.data.processing import UnifiedDataProcessor  
from backend.data.validation import DataValidator, QualityAssessor
from backend.data.export import ExportManager
from backend.data.sync.dmm import DMMSyncManager
from backend.data.quality import DataQualityMonitor
from backend.data.pipelines.pipeline_manager import PipelineManager, PipelineConfig, PipelineStage
from backend.data.schemas import SchemaManager

logger = get_ml_logger(__name__)

class SystemStatus(Enum):
    """システム状態"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class SystemMetrics:
    """システムメトリクス"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_processes: int
    pending_jobs: int
    error_rate: float
    throughput_per_hour: float
    quality_score: float
    uptime_hours: float

@dataclass
class DataFlowStatus:
    """データフロー状態"""
    ingestion_status: str
    processing_status: str
    validation_status: str
    export_status: str
    sync_status: str
    total_records_today: int
    failed_records_today: int
    last_successful_run: Optional[datetime]
    next_scheduled_run: Optional[datetime]

class UnifiedDataManager:
    """統合データ管理システム"""
    
    def __init__(self, config: Dict[str, Any] = None, base_dir: Path = None):
        self.config = config or {}
        self.base_dir = base_dir or Path("backend/data")
        self.startup_time = datetime.now()
        
        # ディレクトリ設定
        self.dirs = {
            "storage": self.base_dir / "storage",
            "logs": self.base_dir / "logs",
            "configs": self.base_dir / "configs",
            "exports": self.base_dir / "storage" / "exported",
            "temp": self.base_dir / "storage" / "temp"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # コンポーネントの初期化
        self._initialize_components()
        
        # システム状態管理
        self.system_status = SystemStatus.HEALTHY
        self.metrics_history: List[SystemMetrics] = []
        self.active_jobs: Dict[str, Any] = {}
        
        logger.info("Unified Data Manager initialized successfully")
    
    def _initialize_components(self):
        """全データコンポーネントの初期化"""
        try:
            # 基本コンポーネント
            self.ingestion_manager = DataIngestionManager(
                sources=self.config.get("data_sources", ["api", "scraping", "file"]),
                output_dir=self.dirs["storage"] / "raw"
            )
            
            self.data_processor = UnifiedDataProcessor(
                input_dir=self.dirs["storage"] / "raw",
                output_dir=self.dirs["storage"] / "processed"
            )
            
            self.data_validator = DataValidator(
                input_dir=self.dirs["storage"] / "processed",
                output_dir=self.dirs["storage"] / "validated"
            )
            
            self.quality_assessor = QualityAssessor(
                input_dir=self.dirs["storage"] / "validated"
            )
            
            self.export_manager = ExportManager(
                input_dir=self.dirs["storage"] / "validated",
                output_dir=self.dirs["exports"]
            )
            
            # 特化コンポーネント
            self.dmm_sync_manager = DMMSyncManager(
                api_config=self.config.get("dmm_api", {}),
                output_dir=self.dirs["storage"] / "raw"
            )
            
            self.quality_monitor = DataQualityMonitor(
                storage_dir=self.dirs["storage"]
            )
            
            self.schema_manager = SchemaManager(
                schema_dir=self.base_dir / "schemas"
            )
            
            # パイプライン管理
            self.pipeline_manager = PipelineManager(
                config=self.config.get("pipeline", {}),
                base_dir=self.base_dir
            )
            
            # デフォルトパイプラインの登録
            self._register_default_pipelines()
            
            logger.info("All data components initialized successfully")
            
        except Exception as error:
            logger.error(f"Component initialization failed: {error}")
            self.system_status = SystemStatus.ERROR
            raise
    
    def _register_default_pipelines(self):
        """デフォルトパイプラインの登録"""
        
        # 完全データ処理パイプライン
        full_pipeline = PipelineConfig(
            name="full_data_pipeline",
            stages=[
                PipelineStage.INGESTION,
                PipelineStage.PROCESSING,
                PipelineStage.VALIDATION,
                PipelineStage.EXPORT
            ],
            schedule="daily",
            data_sources=["api", "scraping"],
            output_format="json,csv",
            quality_threshold=0.8
        )
        self.pipeline_manager.register_pipeline(full_pipeline)
        
        # API同期パイプライン
        api_sync_pipeline = PipelineConfig(
            name="api_sync_pipeline",
            stages=[
                PipelineStage.INGESTION,
                PipelineStage.PROCESSING,
                PipelineStage.VALIDATION
            ],
            schedule="hourly",
            data_sources=["api"],
            output_format="json",
            quality_threshold=0.9
        )
        self.pipeline_manager.register_pipeline(api_sync_pipeline)
        
        # ML学習データ準備パイプライン
        ml_pipeline = PipelineConfig(
            name="ml_training_pipeline",
            stages=[
                PipelineStage.PROCESSING,
                PipelineStage.VALIDATION,
                PipelineStage.EXPORT,
                PipelineStage.ML_TRAINING
            ],
            data_sources=["file"],
            output_format="parquet",
            quality_threshold=0.95
        )
        self.pipeline_manager.register_pipeline(ml_pipeline)
        
        logger.info("Default pipelines registered")
    
    async def run_full_data_flow(
        self,
        sources: List[str] = None,
        target_format: str = "json",
        quality_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """完全データフロー実行"""
        
        job_id = f"full_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting full data flow: {job_id}")
        
        start_time = time.time()
        self.active_jobs[job_id] = {
            "type": "full_data_flow",
            "started_at": datetime.now(),
            "status": "running"
        }
        
        try:
            results = {}
            
            # 1. データ取り込み
            logger.info("Step 1: Data Ingestion")
            ingestion_results = await self._run_ingestion(sources or ["api", "scraping"])
            results["ingestion"] = ingestion_results
            
            if not any(r.get("success", False) for r in ingestion_results.values()):
                raise Exception("No successful ingestion results")
            
            # 2. データ処理
            logger.info("Step 2: Data Processing")
            processing_result = await self._run_processing(job_id)
            results["processing"] = processing_result
            
            if not processing_result.get("success", False):
                raise Exception("Data processing failed")
            
            # 3. データ検証・品質チェック
            logger.info("Step 3: Data Validation & Quality Check")
            validation_result = await self._run_validation(job_id, quality_threshold)
            results["validation"] = validation_result
            
            if not validation_result.get("success", False):
                raise Exception("Data validation failed")
            
            # 4. データエクスポート
            logger.info("Step 4: Data Export")
            export_result = await self._run_export(job_id, target_format)
            results["export"] = export_result
            
            # 最終結果
            execution_time = time.time() - start_time
            
            final_result = {
                "job_id": job_id,
                "success": True,
                "execution_time_seconds": execution_time,
                "stages_completed": len(results),
                "results": results,
                "summary": {
                    "total_records_processed": processing_result.get("output_records", 0),
                    "quality_score": validation_result.get("quality_score", 0),
                    "exported_files": export_result.get("exported_files", [])
                }
            }
            
            self.active_jobs[job_id]["status"] = "completed"
            self.active_jobs[job_id]["result"] = final_result
            
            logger.info(f"Full data flow completed: {job_id}")
            return final_result
            
        except Exception as error:
            logger.error(f"Full data flow failed: {error}")
            
            error_result = {
                "job_id": job_id,
                "success": False,
                "error": str(error),
                "execution_time_seconds": time.time() - start_time
            }
            
            self.active_jobs[job_id]["status"] = "failed"
            self.active_jobs[job_id]["error"] = str(error)
            
            return error_result
    
    async def _run_ingestion(self, sources: List[str]) -> Dict[str, Any]:
        """データ取り込み実行"""
        results = {}
        
        for source in sources:
            try:
                if source == "api":
                    result = await self.ingestion_manager.ingest_dmm_api_data({
                        "api_type": "dmm",
                        "batch_size": 100
                    })
                    results[source] = {
                        "success": result.success,
                        "records_ingested": result.records_ingested,
                        "output_files": result.output_files
                    }
                
                elif source == "scraping":
                    result = await self.ingestion_manager.ingest_scraping_data({
                        "type": "reviews"
                    })
                    results[source] = {
                        "success": result.success,
                        "records_ingested": result.records_ingested,
                        "output_files": result.output_files
                    }
                
            except Exception as error:
                results[source] = {
                    "success": False,
                    "error": str(error)
                }
        
        return results
    
    async def _run_processing(self, job_id: str) -> Dict[str, Any]:
        """データ処理実行"""
        # 最新の取り込みファイルを検索
        raw_files = list((self.dirs["storage"] / "raw").glob("*.json"))
        
        if not raw_files:
            return {"success": False, "error": "No input files found"}
        
        latest_file = max(raw_files, key=lambda f: f.stat().st_mtime)
        
        # データ処理実行
        result = await self.data_processor.process_data(latest_file, job_id=job_id)
        
        return {
            "success": result.success,
            "output_records": result.output_records,
            "quality_score": result.quality_score,
            "processing_time": result.processing_time_seconds,
            "output_files": result.output_files
        }
    
    async def _run_validation(self, job_id: str, quality_threshold: float) -> Dict[str, Any]:
        """データ検証実行"""
        # 処理済みファイルを検索
        processed_files = list((self.dirs["storage"] / "processed").glob(f"{job_id}*.json"))
        
        if not processed_files:
            return {"success": False, "error": "No processed files found"}
        
        latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
        
        # データ検証実行
        validation_result = await self.data_validator.validate_data(latest_file)
        
        if not validation_result.success:
            return {
                "success": False,
                "error": validation_result.error_details
            }
        
        # 品質評価実行
        quality_result = await self.quality_assessor.assess_quality(
            validation_result.output_files[0]
        )
        
        quality_score = quality_result.overall_score if quality_result.success else 0
        
        return {
            "success": quality_score >= quality_threshold,
            "quality_score": quality_score,
            "validation_records": validation_result.validated_records,
            "quality_details": quality_result.details if quality_result.success else None
        }
    
    async def _run_export(self, job_id: str, target_format: str) -> Dict[str, Any]:
        """データエクスポート実行"""
        # 検証済みファイルを検索
        validated_files = list((self.dirs["storage"] / "validated").glob(f"{job_id}*.json"))
        
        if not validated_files:
            return {"success": False, "error": "No validated files found"}
        
        latest_file = max(validated_files, key=lambda f: f.stat().st_mtime)
        
        # エクスポート実行
        export_result = await self.export_manager.export_data(
            latest_file,
            target_format,
            output_name=job_id
        )
        
        return {
            "success": export_result.success,
            "exported_files": export_result.output_files,
            "export_records": export_result.exported_records
        }
    
    async def get_system_metrics(self) -> SystemMetrics:
        """システムメトリクス取得"""
        import psutil
        
        # システムリソース
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.base_dir))
        
        # アプリケーション固有メトリクス
        active_processes = len(self.active_jobs)
        uptime = (datetime.now() - self.startup_time).total_seconds() / 3600
        
        # 品質スコア（最近の実行結果から）
        recent_quality_scores = []
        for job in list(self.active_jobs.values())[-10:]:
            if "result" in job and "results" in job["result"]:
                validation = job["result"]["results"].get("validation", {})
                if "quality_score" in validation:
                    recent_quality_scores.append(validation["quality_score"])
        
        avg_quality = sum(recent_quality_scores) / len(recent_quality_scores) if recent_quality_scores else 0
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            active_processes=active_processes,
            pending_jobs=len([j for j in self.active_jobs.values() if j.get("status") == "running"]),
            error_rate=0,  # TODO: 計算ロジック実装
            throughput_per_hour=0,  # TODO: 計算ロジック実装
            quality_score=avg_quality,
            uptime_hours=uptime
        )
        
        self.metrics_history.append(metrics)
        
        # 履歴を最新100件に制限
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    async def get_data_flow_status(self) -> DataFlowStatus:
        """データフロー状態取得"""
        
        # 各コンポーネントの状態確認
        ingestion_status = "idle"
        processing_status = "idle"
        validation_status = "idle"
        export_status = "idle"
        sync_status = "idle"
        
        # アクティブジョブから状態判定
        for job in self.active_jobs.values():
            if job.get("status") == "running":
                job_type = job.get("type", "")
                if "ingestion" in job_type:
                    ingestion_status = "running"
                elif "processing" in job_type:
                    processing_status = "running"
                elif "validation" in job_type:
                    validation_status = "running"
                elif "export" in job_type:
                    export_status = "running"
                elif "sync" in job_type:
                    sync_status = "running"
        
        # 今日の統計
        today = datetime.now().date()
        today_jobs = [
            job for job in self.active_jobs.values()
            if job.get("started_at", datetime.min).date() == today
        ]
        
        total_records_today = sum(
            job.get("result", {}).get("summary", {}).get("total_records_processed", 0)
            for job in today_jobs
            if job.get("status") == "completed"
        )
        
        failed_records_today = len([
            job for job in today_jobs
            if job.get("status") == "failed"
        ])
        
        # 最後の成功実行
        successful_jobs = [
            job for job in self.active_jobs.values()
            if job.get("status") == "completed"
        ]
        
        last_successful_run = None
        if successful_jobs:
            last_job = max(successful_jobs, key=lambda j: j.get("started_at", datetime.min))
            last_successful_run = last_job.get("started_at")
        
        return DataFlowStatus(
            ingestion_status=ingestion_status,
            processing_status=processing_status,
            validation_status=validation_status,
            export_status=export_status,
            sync_status=sync_status,
            total_records_today=total_records_today,
            failed_records_today=failed_records_today,
            last_successful_run=last_successful_run,
            next_scheduled_run=None  # TODO: スケジューラーから取得
        )
    
    def get_component_status(self) -> Dict[str, Any]:
        """全コンポーネント状態取得"""
        
        components = {
            "ingestion_manager": {"status": "healthy", "last_activity": None},
            "data_processor": {"status": "healthy", "last_activity": None},
            "data_validator": {"status": "healthy", "last_activity": None},
            "quality_assessor": {"status": "healthy", "last_activity": None},
            "export_manager": {"status": "healthy", "last_activity": None},
            "dmm_sync_manager": {"status": "healthy", "last_activity": None},
            "quality_monitor": {"status": "healthy", "last_activity": None},
            "schema_manager": {"status": "healthy", "last_activity": None},
            "pipeline_manager": {"status": "healthy", "last_activity": None}
        }
        
        # パイプライン状態追加
        try:
            pipeline_status = self.pipeline_manager.get_system_status()
            components["pipeline_manager"]["details"] = pipeline_status
        except Exception as error:
            components["pipeline_manager"]["status"] = "error"
            components["pipeline_manager"]["error"] = str(error)
        
        return components
    
    async def run_health_check(self) -> Dict[str, Any]:
        """システムヘルスチェック"""
        logger.info("Running system health check")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "metrics": {},
            "recommendations": []
        }
        
        try:
            # システムメトリクス取得
            metrics = await self.get_system_metrics()
            health_report["metrics"] = asdict(metrics)
            
            # コンポーネント状態確認
            components = self.get_component_status()
            health_report["components"] = components
            
            # データフロー状態
            data_flow = await self.get_data_flow_status()
            health_report["data_flow"] = asdict(data_flow)
            
            # 全体状態判定
            if metrics.cpu_usage > 90:
                health_report["overall_status"] = "warning"
                health_report["recommendations"].append("High CPU usage detected")
            
            if metrics.memory_usage > 90:
                health_report["overall_status"] = "warning"
                health_report["recommendations"].append("High memory usage detected")
            
            if metrics.disk_usage > 90:
                health_report["overall_status"] = "error"
                health_report["recommendations"].append("Disk space critically low")
            
            # エラー率チェック
            failed_jobs = len([j for j in self.active_jobs.values() if j.get("status") == "failed"])
            total_jobs = len(self.active_jobs)
            
            if total_jobs > 0 and failed_jobs / total_jobs > 0.1:
                health_report["overall_status"] = "warning"
                health_report["recommendations"].append("High job failure rate detected")
            
        except Exception as error:
            logger.error(f"Health check failed: {error}")
            health_report["overall_status"] = "error"
            health_report["error"] = str(error)
        
        return health_report
    
    async def cleanup_old_data(self, retention_days: int = 30) -> Dict[str, Any]:
        """古いデータのクリーンアップ"""
        logger.info(f"Starting data cleanup (retention: {retention_days} days)")
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleanup_report = {
            "cutoff_date": cutoff_date.isoformat(),
            "directories_cleaned": {},
            "total_files_removed": 0,
            "total_space_freed_mb": 0
        }
        
        for dir_name, dir_path in self.dirs.items():
            if dir_name == "configs":  # 設定ファイルはスキップ
                continue
            
            files_removed = 0
            space_freed = 0
            
            try:
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if file_time < cutoff_date:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            space_freed += file_size
                
                cleanup_report["directories_cleaned"][dir_name] = {
                    "files_removed": files_removed,
                    "space_freed_mb": space_freed / (1024 * 1024)
                }
                
                cleanup_report["total_files_removed"] += files_removed
                cleanup_report["total_space_freed_mb"] += space_freed / (1024 * 1024)
                
            except Exception as error:
                logger.warning(f"Cleanup failed for {dir_name}: {error}")
                cleanup_report["directories_cleaned"][dir_name] = {
                    "error": str(error)
                }
        
        logger.info(f"Cleanup completed: {cleanup_report['total_files_removed']} files removed")
        return cleanup_report
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボード用データ取得"""
        
        metrics = await self.get_system_metrics()
        data_flow = await self.get_data_flow_status()
        components = self.get_component_status()
        
        # 最近のジョブ統計
        recent_jobs = list(self.active_jobs.values())[-20:]
        job_stats = {
            "total": len(recent_jobs),
            "completed": len([j for j in recent_jobs if j.get("status") == "completed"]),
            "failed": len([j for j in recent_jobs if j.get("status") == "failed"]),
            "running": len([j for j in recent_jobs if j.get("status") == "running"])
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": asdict(metrics),
            "data_flow_status": asdict(data_flow),
            "component_status": components,
            "job_statistics": job_stats,
            "pipeline_status": self.pipeline_manager.get_system_status(),
            "storage_info": {
                dir_name: {
                    "file_count": len(list(dir_path.glob("*"))),
                    "size_mb": sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file()) / (1024 * 1024)
                }
                for dir_name, dir_path in self.dirs.items()
            }
        }

# Factory function for easy instantiation
def create_unified_data_manager(config: Dict[str, Any] = None) -> UnifiedDataManager:
    """統合データ管理システムの作成"""
    
    default_config = {
        "data_sources": ["api", "scraping", "file"],
        "dmm_api": {
            "api_id": "W63Kd4A4ym2DaycFcXSU",
            "affiliate_id": "yotadata2-990",
            "rate_limit": 1.0
        },
        "pipeline": {
            "auto_retry": True,
            "max_retries": 3,
            "quality_threshold": 0.8
        }
    }
    
    if config:
        default_config.update(config)
    
    return UnifiedDataManager(config=default_config)