#!/usr/bin/env python3
"""
Data Export Manager
統合データエクスポートシステム - 複数形式対応・スケジューラー統合
"""

from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import csv
import logging
import asyncio
import aiofiles
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """エクスポート形式"""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"
    NDJSON = "ndjson"  # Newline Delimited JSON
    ARROW = "arrow"

class ExportType(Enum):
    """エクスポートタイプ"""
    FULL = "full"
    INCREMENTAL = "incremental"
    FILTERED = "filtered"
    AGGREGATED = "aggregated"
    CUSTOM = "custom"

@dataclass
class ExportFilter:
    """エクスポートフィルタ定義"""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, like, ilike
    value: Any
    logical_operator: str = "AND"  # AND, OR

@dataclass
class ExportConfig:
    """エクスポート設定"""
    name: str
    data_source: str
    export_format: ExportFormat
    export_type: ExportType
    output_path: Path
    filters: List[ExportFilter] = field(default_factory=list)
    columns: Optional[List[str]] = None
    aggregations: Optional[Dict[str, str]] = None
    chunk_size: int = 10000
    compression: Optional[str] = None
    schedule: Optional[str] = None  # cron expression
    enabled: bool = True
    retention_days: int = 30

@dataclass
class ExportResult:
    """エクスポート結果"""
    export_name: str
    export_id: str
    status: str  # success, failed, partial
    start_time: datetime
    end_time: Optional[datetime] = None
    records_exported: int = 0
    file_size_bytes: int = 0
    output_files: List[Path] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

class BaseExporter(ABC):
    """エクスポータベースクラス"""
    
    @abstractmethod
    async def export_data(
        self, 
        data: pd.DataFrame, 
        output_path: Path, 
        config: ExportConfig
    ) -> ExportResult:
        """データエクスポート実行"""
        pass

class JSONExporter(BaseExporter):
    """JSON形式エクスポータ"""
    
    async def export_data(
        self, 
        data: pd.DataFrame, 
        output_path: Path, 
        config: ExportConfig
    ) -> ExportResult:
        start_time = datetime.now()
        export_id = f"json_{config.name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSONに変換
            json_data = data.to_json(
                orient='records',
                date_format='iso',
                force_ascii=False
            )
            
            # 圧縮オプション
            if config.compression == "gzip":
                import gzip
                with gzip.open(f"{output_path}.gz", 'wt', encoding='utf-8') as f:
                    f.write(json_data)
                final_path = Path(f"{output_path}.gz")
            else:
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(json_data)
                final_path = output_path
            
            file_size = final_path.stat().st_size
            
            return ExportResult(
                export_name=config.name,
                export_id=export_id,
                status="success",
                start_time=start_time,
                end_time=datetime.now(),
                records_exported=len(data),
                file_size_bytes=file_size,
                output_files=[final_path]
            )
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return ExportResult(
                export_name=config.name,
                export_id=export_id,
                status="failed",
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )

class CSVExporter(BaseExporter):
    """CSV形式エクスポータ"""
    
    async def export_data(
        self, 
        data: pd.DataFrame, 
        output_path: Path, 
        config: ExportConfig
    ) -> ExportResult:
        start_time = datetime.now()
        export_id = f"csv_{config.name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # チャンク処理で大きなデータセットに対応
            if len(data) > config.chunk_size:
                output_files = []
                for i, chunk in enumerate(np.array_split(data, len(data) // config.chunk_size + 1)):
                    chunk_path = output_path.parent / f"{output_path.stem}_part_{i+1:03d}.csv"
                    chunk.to_csv(chunk_path, index=False, encoding='utf-8')
                    output_files.append(chunk_path)
            else:
                data.to_csv(output_path, index=False, encoding='utf-8')
                output_files = [output_path]
            
            total_size = sum(f.stat().st_size for f in output_files)
            
            return ExportResult(
                export_name=config.name,
                export_id=export_id,
                status="success",
                start_time=start_time,
                end_time=datetime.now(),
                records_exported=len(data),
                file_size_bytes=total_size,
                output_files=output_files
            )
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return ExportResult(
                export_name=config.name,
                export_id=export_id,
                status="failed",
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )

class ParquetExporter(BaseExporter):
    """Parquet形式エクスポータ"""
    
    async def export_data(
        self, 
        data: pd.DataFrame, 
        output_path: Path, 
        config: ExportConfig
    ) -> ExportResult:
        start_time = datetime.now()
        export_id = f"parquet_{config.name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Parquet形式で保存
            compression = config.compression or "snappy"
            data.to_parquet(
                output_path,
                compression=compression,
                index=False
            )
            
            file_size = output_path.stat().st_size
            
            return ExportResult(
                export_name=config.name,
                export_id=export_id,
                status="success",
                start_time=start_time,
                end_time=datetime.now(),
                records_exported=len(data),
                file_size_bytes=file_size,
                output_files=[output_path]
            )
            
        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            return ExportResult(
                export_name=config.name,
                export_id=export_id,
                status="failed",
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )

class ExportManager:
    """統合エクスポート管理システム"""
    
    def __init__(self, base_export_path: Optional[Path] = None):
        self.base_export_path = base_export_path or Path.cwd() / "exports"
        self.export_configs: Dict[str, ExportConfig] = {}
        self.export_history: List[ExportResult] = []
        self.exporters: Dict[ExportFormat, BaseExporter] = {
            ExportFormat.JSON: JSONExporter(),
            ExportFormat.CSV: CSVExporter(),
            ExportFormat.PARQUET: ParquetExporter()
        }
        
        # エクスポートディレクトリ作成
        self.base_export_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Export manager initialized with base path: {self.base_export_path}")
    
    def register_export_config(self, config: ExportConfig):
        """エクスポート設定登録"""
        self.export_configs[config.name] = config
        logger.info(f"Export config registered: {config.name}")
    
    async def execute_export(
        self, 
        config_name: str, 
        data_source: Optional[pd.DataFrame] = None,
        custom_filters: Optional[List[ExportFilter]] = None
    ) -> ExportResult:
        """エクスポート実行"""
        if config_name not in self.export_configs:
            raise ValueError(f"Export config '{config_name}' not found")
        
        config = self.export_configs[config_name]
        
        if not config.enabled:
            logger.warning(f"Export config '{config_name}' is disabled")
            return ExportResult(
                export_name=config_name,
                export_id=f"disabled_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                status="failed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message="Export configuration is disabled"
            )
        
        try:
            # データ取得
            if data_source is not None:
                data = data_source.copy()
            else:
                data = await self._load_data_source(config.data_source)
            
            # フィルタ適用
            filters_to_apply = config.filters.copy()
            if custom_filters:
                filters_to_apply.extend(custom_filters)
            
            if filters_to_apply:
                data = self._apply_filters(data, filters_to_apply)
            
            # カラム選択
            if config.columns:
                available_columns = [col for col in config.columns if col in data.columns]
                data = data[available_columns]
            
            # 集約処理
            if config.aggregations:
                data = self._apply_aggregations(data, config.aggregations)
            
            # 出力パス生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{config.name}_{timestamp}.{config.export_format.value}"
            output_path = self.base_export_path / config.name / output_filename
            
            # エクスポート実行
            exporter = self.exporters.get(config.export_format)
            if not exporter:
                raise ValueError(f"Unsupported export format: {config.export_format}")
            
            result = await exporter.export_data(data, output_path, config)
            
            # 履歴に追加
            self.export_history.append(result)
            
            # 古いファイルのクリーンアップ
            if config.retention_days > 0:
                await self._cleanup_old_exports(config)
            
            logger.info(f"Export completed: {config_name} -> {result.status}")
            return result
            
        except Exception as e:
            logger.error(f"Export failed for {config_name}: {e}")
            error_result = ExportResult(
                export_name=config_name,
                export_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                status="failed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )
            self.export_history.append(error_result)
            return error_result
    
    async def _load_data_source(self, data_source: str) -> pd.DataFrame:
        """データソース読み込み"""
        # データソースが存在する場合の読み込みロジック
        # 実際の実装では、データベース、ファイル、APIなどから読み込み
        logger.warning(f"Data source loading not implemented: {data_source}")
        return pd.DataFrame()
    
    def _apply_filters(self, data: pd.DataFrame, filters: List[ExportFilter]) -> pd.DataFrame:
        """フィルタ適用"""
        if not filters:
            return data
        
        mask = pd.Series([True] * len(data))
        current_logical_op = "AND"
        
        for filter_config in filters:
            if filter_config.field not in data.columns:
                logger.warning(f"Filter field '{filter_config.field}' not found in data")
                continue
            
            field_data = data[filter_config.field]
            
            # フィルタ条件適用
            if filter_config.operator == "eq":
                condition = field_data == filter_config.value
            elif filter_config.operator == "ne":
                condition = field_data != filter_config.value
            elif filter_config.operator == "gt":
                condition = field_data > filter_config.value
            elif filter_config.operator == "lt":
                condition = field_data < filter_config.value
            elif filter_config.operator == "gte":
                condition = field_data >= filter_config.value
            elif filter_config.operator == "lte":
                condition = field_data <= filter_config.value
            elif filter_config.operator == "in":
                condition = field_data.isin(filter_config.value)
            elif filter_config.operator == "not_in":
                condition = ~field_data.isin(filter_config.value)
            elif filter_config.operator == "like":
                condition = field_data.astype(str).str.contains(filter_config.value, na=False)
            else:
                logger.warning(f"Unknown filter operator: {filter_config.operator}")
                continue
            
            # 論理演算子適用
            if current_logical_op == "AND":
                mask = mask & condition
            elif current_logical_op == "OR":
                mask = mask | condition
            
            current_logical_op = filter_config.logical_operator
        
        return data[mask]
    
    def _apply_aggregations(self, data: pd.DataFrame, aggregations: Dict[str, str]) -> pd.DataFrame:
        """集約処理適用"""
        try:
            agg_funcs = {}
            for field, func in aggregations.items():
                if field in data.columns:
                    agg_funcs[field] = func
            
            if agg_funcs:
                return data.agg(agg_funcs).to_frame().T
            else:
                return data
                
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return data
    
    async def _cleanup_old_exports(self, config: ExportConfig):
        """古いエクスポートファイルのクリーンアップ"""
        try:
            config_dir = self.base_export_path / config.name
            if not config_dir.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=config.retention_days)
            
            for file_path in config_dir.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Cleaned up old export file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Cleanup failed for {config.name}: {e}")
    
    def get_export_history(
        self, 
        config_name: Optional[str] = None, 
        limit: int = 50
    ) -> List[ExportResult]:
        """エクスポート履歴取得"""
        history = self.export_history
        
        if config_name:
            history = [h for h in history if h.export_name == config_name]
        
        # 最新順でソート
        history.sort(key=lambda x: x.start_time, reverse=True)
        
        return history[:limit]
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """エクスポート統計情報取得"""
        total_exports = len(self.export_history)
        successful_exports = len([h for h in self.export_history if h.status == "success"])
        failed_exports = len([h for h in self.export_history if h.status == "failed"])
        
        total_records = sum(h.records_exported for h in self.export_history if h.status == "success")
        total_size = sum(h.file_size_bytes for h in self.export_history if h.status == "success")
        
        return {
            "total_exports": total_exports,
            "successful_exports": successful_exports,
            "failed_exports": failed_exports,
            "success_rate": successful_exports / total_exports if total_exports > 0 else 0.0,
            "total_records_exported": total_records,
            "total_size_bytes": total_size,
            "registered_configs": len(self.export_configs),
            "enabled_configs": len([c for c in self.export_configs.values() if c.enabled])
        }
    
    async def batch_export(self, config_names: List[str]) -> List[ExportResult]:
        """バッチエクスポート実行"""
        results = []
        
        for config_name in config_names:
            try:
                result = await self.execute_export(config_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch export failed for {config_name}: {e}")
                error_result = ExportResult(
                    export_name=config_name,
                    export_id=f"batch_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    status="failed",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results