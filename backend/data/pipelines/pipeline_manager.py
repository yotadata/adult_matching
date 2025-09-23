"""
Pipeline Manager

データパイプライン管理システム
- 取り込み→処理→検証→エクスポートの統合パイプライン
- スケジュール実行・監視・エラーハンドリング
- MLパイプラインとの統合
"""

import asyncio
import json
import schedule
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

from backend.ml.utils.logger import get_ml_logger
from backend.data.ingestion import DataIngestionManager, IngestionResult
from backend.data.processing import UnifiedDataProcessor, ProcessingResult

logger = get_ml_logger(__name__)

class PipelineStatus(Enum):
    """パイプライン状態"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PipelineStage(Enum):
    """パイプラインステージ"""
    INGESTION = "ingestion"
    PROCESSING = "processing"
    VALIDATION = "validation"
    EXPORT = "export"
    ML_TRAINING = "ml_training"

@dataclass
class PipelineConfig:
    """パイプライン設定"""
    name: str
    stages: List[PipelineStage]
    schedule: Optional[str] = None  # cron形式
    auto_retry: bool = True
    max_retries: int = 3
    notification_email: Optional[str] = None
    data_sources: List[str] = None
    output_format: str = "json"
    quality_threshold: float = 0.8
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = ["api", "scraping", "file"]

@dataclass
class PipelineRun:
    """パイプライン実行記録"""
    run_id: str
    pipeline_name: str
    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    stages_completed: List[PipelineStage] = None
    current_stage: Optional[PipelineStage] = None
    error_message: Optional[str] = None
    input_records: int = 0
    output_records: int = 0
    processing_time_seconds: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.stages_completed is None:
            self.stages_completed = []
        if self.metadata is None:
            self.metadata = {}

class PipelineManager:
    """パイプライン管理システム"""
    
    def __init__(self, config: Dict[str, Any] = None, base_dir: Path = None):
        self.config = config or {}
        self.base_dir = base_dir or Path("data")
        
        # サブディレクトリの作成
        self.dirs = {
            "raw": self.base_dir / "storage" / "raw",
            "processed": self.base_dir / "storage" / "processed",
            "validated": self.base_dir / "storage" / "validated",
            "exported": self.base_dir / "storage" / "exported",
            "logs": self.base_dir / "logs"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # コンポーネントの初期化
        self.ingestion_manager = DataIngestionManager(
            sources=self.config.get("data_sources", ["api", "scraping", "file"]),
            output_dir=self.dirs["raw"]
        )
        self.data_processor = UnifiedDataProcessor(
            input_dir=self.dirs["raw"],
            output_dir=self.dirs["processed"]
        )
        
        # パイプライン管理
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.active_runs: Dict[str, PipelineRun] = {}
        self.completed_runs: Dict[str, PipelineRun] = {}
        
        # スケジューラー
        self.scheduler_running = False
        
    def register_pipeline(self, config: PipelineConfig) -> str:
        """パイプラインの登録"""
        logger.info(f"Registering pipeline: {config.name}")
        
        self.pipelines[config.name] = config
        
        # スケジュール設定があれば登録
        if config.schedule:
            self._schedule_pipeline(config.name, config.schedule)
        
        logger.info(f"Pipeline {config.name} registered successfully")
        return config.name
    
    async def run_pipeline(
        self,
        pipeline_name: str,
        input_data: Optional[Union[Path, Dict[str, Any]]] = None,
        run_id: Optional[str] = None
    ) -> PipelineRun:
        """パイプラインの実行"""
        
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_name}")
        
        config = self.pipelines[pipeline_name]
        
        if run_id is None:
            run_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting pipeline run: {run_id}")
        
        # パイプライン実行記録の作成
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=pipeline_name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
            metadata={"config": asdict(config)}
        )
        
        self.active_runs[run_id] = run
        
        try:
            # ステージごとの実行
            for stage in config.stages:
                run.current_stage = stage
                logger.info(f"Executing stage: {stage.value}")
                
                if stage == PipelineStage.INGESTION:
                    await self._execute_ingestion_stage(run, input_data)
                elif stage == PipelineStage.PROCESSING:
                    await self._execute_processing_stage(run)
                elif stage == PipelineStage.VALIDATION:
                    await self._execute_validation_stage(run)
                elif stage == PipelineStage.EXPORT:
                    await self._execute_export_stage(run)
                elif stage == PipelineStage.ML_TRAINING:
                    await self._execute_ml_training_stage(run)
                
                run.stages_completed.append(stage)
                logger.info(f"Stage {stage.value} completed")
            
            # パイプライン完了
            run.status = PipelineStatus.COMPLETED
            run.completed_at = datetime.now()
            run.processing_time_seconds = (run.completed_at - run.started_at).total_seconds()
            
            # アクティブから完了に移動
            del self.active_runs[run_id]
            self.completed_runs[run_id] = run
            
            logger.info(f"Pipeline run {run_id} completed successfully")
            
        except Exception as error:
            logger.error(f"Pipeline run {run_id} failed: {error}")
            
            run.status = PipelineStatus.FAILED
            run.error_message = str(error)
            run.completed_at = datetime.now()
            
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            self.completed_runs[run_id] = run
            
            # 自動リトライ
            if config.auto_retry and run.metadata.get("retry_count", 0) < config.max_retries:
                retry_count = run.metadata.get("retry_count", 0) + 1
                run.metadata["retry_count"] = retry_count
                
                logger.info(f"Retrying pipeline run {run_id} (attempt {retry_count})")
                await asyncio.sleep(60)  # 1分待機
                return await self.run_pipeline(pipeline_name, input_data, f"{run_id}_retry_{retry_count}")
        
        return run
    
    async def _execute_ingestion_stage(self, run: PipelineRun, input_data: Any):
        """取り込みステージの実行"""
        logger.info("Executing ingestion stage")
        
        ingestion_results = []
        
        config = self.pipelines[run.pipeline_name]
        
        # データソース別の取り込み
        for source in config.data_sources:
            try:
                if source == "api":
                    result = await self.ingestion_manager.ingest_dmm_api_data({
                        "api_type": "dmm",
                        "batch_size": 100
                    })
                    ingestion_results.append(result)
                
                elif source == "scraping" and input_data:
                    if isinstance(input_data, Path):
                        result = await self.ingestion_manager.ingest_scraping_data({
                            "type": "reviews"
                        }, input_data)
                        ingestion_results.append(result)
                
                elif source == "file" and input_data:
                    if isinstance(input_data, Path):
                        result = await self.ingestion_manager.ingest_file_data(input_data)
                        ingestion_results.append(result)
                
            except Exception as error:
                logger.warning(f"Ingestion from {source} failed: {error}")
                continue
        
        # 取り込み結果の集計
        if ingestion_results:
            total_records = sum(r.records_ingested for r in ingestion_results if r.success)
            run.input_records = total_records
            run.metadata["ingestion_results"] = [asdict(r) for r in ingestion_results]
            
            logger.info(f"Ingestion completed: {total_records} records")
        else:
            raise Exception("No successful ingestion results")
    
    async def _execute_processing_stage(self, run: PipelineRun):
        """処理ステージの実行"""
        logger.info("Executing processing stage")
        
        # 取り込みステージの出力ファイルを検索
        raw_files = list(self.dirs["raw"].glob("*.json"))
        
        if not raw_files:
            raise Exception("No input files found for processing")
        
        # 最新のファイルを処理
        latest_file = max(raw_files, key=lambda f: f.stat().st_mtime)
        
        # データ処理の実行
        processing_result = await self.data_processor.process_data(
            latest_file,
            job_id=f"{run.run_id}_processing"
        )
        
        if not processing_result.success:
            raise Exception(f"Processing failed: {processing_result.error_details}")
        
        run.output_records = processing_result.output_records
        run.quality_score = processing_result.quality_score
        run.metadata["processing_result"] = asdict(processing_result)
        
        logger.info(f"Processing completed: {processing_result.output_records} records")
    
    async def _execute_validation_stage(self, run: PipelineRun):
        """検証ステージの実行"""
        logger.info("Executing validation stage")
        
        config = self.pipelines[run.pipeline_name]
        
        # 品質しきい値チェック
        if run.quality_score < config.quality_threshold:
            raise Exception(f"Quality score {run.quality_score} below threshold {config.quality_threshold}")
        
        # 処理済みファイルの検証
        processed_files = list(self.dirs["processed"].glob(f"{run.run_id}_processing*.json"))
        
        if not processed_files:
            raise Exception("No processed files found for validation")
        
        # 検証済みディレクトリにコピー
        for file_path in processed_files:
            validated_path = self.dirs["validated"] / f"{run.run_id}_validated.json"
            
            # ファイル内容の読み込みと検証
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                raise Exception("Validated data is empty")
            
            # 検証済みファイルとして保存
            with open(validated_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            run.metadata["validated_file"] = str(validated_path)
        
        logger.info("Validation stage completed")
    
    async def _execute_export_stage(self, run: PipelineRun):
        """エクスポートステージの実行"""
        logger.info("Executing export stage")
        
        config = self.pipelines[run.pipeline_name]
        
        # 検証済みファイルを検索
        validated_file = run.metadata.get("validated_file")
        
        if not validated_file or not Path(validated_file).exists():
            raise Exception("No validated file found for export")
        
        # エクスポート形式別の処理
        export_formats = config.output_format.split(",") if isinstance(config.output_format, str) else [config.output_format]
        
        exported_files = []
        
        for format_type in export_formats:
            format_type = format_type.strip()
            
            try:
                with open(validated_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if format_type == "json":
                    # 既にJSON形式なのでコピー
                    export_path = self.dirs["exported"] / f"{run.run_id}.json"
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                    exported_files.append(str(export_path))
                
                elif format_type == "csv":
                    # CSV形式でエクスポート
                    import pandas as pd
                    df = pd.DataFrame(data)
                    export_path = self.dirs["exported"] / f"{run.run_id}.csv"
                    df.to_csv(export_path, index=False, encoding='utf-8')
                    exported_files.append(str(export_path))
                
                elif format_type == "parquet":
                    # Parquet形式でエクスポート
                    import pandas as pd
                    df = pd.DataFrame(data)
                    export_path = self.dirs["exported"] / f"{run.run_id}.parquet"
                    df.to_parquet(export_path, index=False)
                    exported_files.append(str(export_path))
                
            except Exception as error:
                logger.warning(f"Export to {format_type} failed: {error}")
                continue
        
        if not exported_files:
            raise Exception("All export formats failed")
        
        run.metadata["exported_files"] = exported_files
        logger.info(f"Export completed: {len(exported_files)} files")
    
    async def _execute_ml_training_stage(self, run: PipelineRun):
        """ML学習ステージの実行"""
        logger.info("Executing ML training stage")
        
        # エクスポートされたデータでMLトレーニングを実行
        exported_files = run.metadata.get("exported_files", [])
        
        if not exported_files:
            raise Exception("No exported files found for ML training")
        
        # ML学習の統合（実装は別モジュール）
        # ここでは模擬的な処理
        training_result = {
            "model_trained": True,
            "training_records": run.output_records,
            "model_accuracy": 0.85,
            "training_time_seconds": 300
        }
        
        run.metadata["ml_training_result"] = training_result
        logger.info("ML training stage completed")
    
    def _schedule_pipeline(self, pipeline_name: str, schedule_expr: str):
        """パイプラインのスケジュール設定"""
        logger.info(f"Scheduling pipeline {pipeline_name} with expression: {schedule_expr}")
        
        # 簡単なスケジュール設定（実際のプロダクションではより高度なスケジューラーを使用）
        if schedule_expr == "daily":
            schedule.every().day.at("02:00").do(self._run_scheduled_pipeline, pipeline_name)
        elif schedule_expr == "hourly":
            schedule.every().hour.do(self._run_scheduled_pipeline, pipeline_name)
        elif schedule_expr.startswith("cron:"):
            # cron形式の解析は今後実装
            logger.warning("Cron expressions not yet implemented")
    
    def _run_scheduled_pipeline(self, pipeline_name: str):
        """スケジュール実行（同期ラッパー）"""
        logger.info(f"Running scheduled pipeline: {pipeline_name}")
        
        # 非同期実行のため、別スレッドで実行
        import threading
        
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_pipeline(pipeline_name))
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async)
        thread.start()
    
    async def start_scheduler(self):
        """スケジューラーの開始"""
        if self.scheduler_running:
            logger.warning("Scheduler is already running")
            return
        
        self.scheduler_running = True
        logger.info("Starting pipeline scheduler")
        
        while self.scheduler_running:
            schedule.run_pending()
            await asyncio.sleep(60)  # 1分間隔でチェック
    
    def stop_scheduler(self):
        """スケジューラーの停止"""
        self.scheduler_running = False
        logger.info("Pipeline scheduler stopped")
    
    def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """パイプライン状態の取得"""
        if pipeline_name not in self.pipelines:
            return {"error": "Pipeline not found"}
        
        config = self.pipelines[pipeline_name]
        active_runs = [run for run in self.active_runs.values() if run.pipeline_name == pipeline_name]
        recent_runs = [run for run in self.completed_runs.values() if run.pipeline_name == pipeline_name]
        recent_runs.sort(key=lambda r: r.completed_at or datetime.min, reverse=True)
        
        return {
            "name": pipeline_name,
            "config": asdict(config),
            "active_runs": len(active_runs),
            "recent_runs": [asdict(run) for run in recent_runs[:5]],
            "last_success": next(
                (run.completed_at.isoformat() for run in recent_runs 
                 if run.status == PipelineStatus.COMPLETED), None
            )
        }
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """パイプライン一覧の取得"""
        return [
            {
                "name": name,
                "stages": [stage.value for stage in config.stages],
                "schedule": config.schedule,
                "status": "active" if any(run.pipeline_name == name for run in self.active_runs.values()) else "idle"
            }
            for name, config in self.pipelines.items()
        ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム全体の状態"""
        return {
            "registered_pipelines": len(self.pipelines),
            "active_runs": len(self.active_runs),
            "completed_runs": len(self.completed_runs),
            "scheduler_running": self.scheduler_running,
            "storage_usage": {
                dir_name: len(list(dir_path.glob("*"))) 
                for dir_name, dir_path in self.dirs.items()
            }
        }