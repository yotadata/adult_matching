"""
Unified data export system supporting multiple formats and destinations.
Handles batch exports, streaming exports, and format conversions.
"""

from typing import Dict, List, Any, Optional, Union, Iterator, AsyncIterator
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import asyncio
from enum import Enum
import gzip
import pickle
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"
    NUMPY = "numpy"


class CompressionType(Enum):
    """Supported compression types"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bz2"
    XZ = "xz"


@dataclass
class ExportConfig:
    """Configuration for data export operations"""
    format: ExportFormat
    output_path: Path
    compression: CompressionType = CompressionType.NONE
    batch_size: int = 10000
    include_metadata: bool = True
    overwrite: bool = False
    streaming: bool = False
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportResult:
    """Result of an export operation"""
    export_id: str
    config: ExportConfig
    records_exported: int
    file_size_bytes: int
    export_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    output_files: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseExporter(ABC):
    """Base class for data exporters"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @abstractmethod
    async def export_dataframe(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> ExportResult:
        """Export a pandas DataFrame"""
        pass
    
    @abstractmethod
    async def export_streaming(
        self, 
        data_iterator: AsyncIterator[pd.DataFrame], 
        metadata: Optional[Dict] = None
    ) -> ExportResult:
        """Export data from an async iterator"""
        pass
    
    def _prepare_output_path(self) -> Path:
        """Prepare the output path with appropriate extensions"""
        output_path = self.config.output_path
        
        # Add format extension if not present
        format_extensions = {
            ExportFormat.JSON: ".json",
            ExportFormat.JSONL: ".jsonl",
            ExportFormat.CSV: ".csv",
            ExportFormat.PARQUET: ".parquet",
            ExportFormat.PICKLE: ".pkl",
            ExportFormat.NUMPY: ".npz"
        }
        
        format_ext = format_extensions.get(self.config.format)
        if format_ext and not output_path.suffix:
            output_path = output_path.with_suffix(format_ext)
        
        # Add compression extension
        compression_extensions = {
            CompressionType.GZIP: ".gz",
            CompressionType.BZIP2: ".bz2",
            CompressionType.XZ: ".xz"
        }
        
        comp_ext = compression_extensions.get(self.config.compression)
        if comp_ext and self.config.compression != CompressionType.NONE:
            output_path = Path(str(output_path) + comp_ext)
        
        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check overwrite policy
        if output_path.exists() and not self.config.overwrite:
            raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")
        
        return output_path
    
    def _add_metadata(self, data: Dict[str, Any], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add metadata to exported data"""
        if not self.config.include_metadata:
            return data
        
        export_metadata = {
            "export_id": self.export_id,
            "export_timestamp": datetime.now().isoformat(),
            "export_config": {
                "format": self.config.format.value,
                "compression": self.config.compression.value,
                "batch_size": self.config.batch_size
            }
        }
        
        if metadata:
            export_metadata["source_metadata"] = metadata
        
        data["_metadata"] = export_metadata
        return data


class JSONExporter(BaseExporter):
    """JSON format exporter"""
    
    async def export_dataframe(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> ExportResult:
        start_time = datetime.now()
        output_path = self._prepare_output_path()
        
        try:
            # Convert DataFrame to dictionary
            data = {
                "data": df.to_dict(orient="records"),
                "schema": {
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "shape": df.shape
                }
            }
            
            # Add metadata
            data = self._add_metadata(data, metadata)
            
            # Write to file with compression if specified
            if self.config.compression == CompressionType.GZIP:
                with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # Calculate result metrics
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = output_path.stat().st_size
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=len(df),
                file_size_bytes=file_size,
                export_time_seconds=export_time,
                success=True,
                output_files=[output_path],
                metadata={"original_shape": df.shape}
            )
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            export_time = (datetime.now() - start_time).total_seconds()
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=0,
                file_size_bytes=0,
                export_time_seconds=export_time,
                success=False,
                error_message=str(e)
            )
    
    async def export_streaming(
        self, 
        data_iterator: AsyncIterator[pd.DataFrame], 
        metadata: Optional[Dict] = None
    ) -> ExportResult:
        start_time = datetime.now()
        output_path = self._prepare_output_path()
        
        try:
            total_records = 0
            all_data = []
            
            async for batch_df in data_iterator:
                batch_data = batch_df.to_dict(orient="records")
                all_data.extend(batch_data)
                total_records += len(batch_df)
            
            # Create final data structure
            data = {
                "data": all_data,
                "metadata": {"total_batches_processed": len(all_data) // self.config.batch_size + 1}
            }
            
            data = self._add_metadata(data, metadata)
            
            # Write to file
            if self.config.compression == CompressionType.GZIP:
                with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = output_path.stat().st_size
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=total_records,
                file_size_bytes=file_size,
                export_time_seconds=export_time,
                success=True,
                output_files=[output_path]
            )
            
        except Exception as e:
            logger.error(f"Streaming JSON export failed: {e}")
            export_time = (datetime.now() - start_time).total_seconds()
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=0,
                file_size_bytes=0,
                export_time_seconds=export_time,
                success=False,
                error_message=str(e)
            )


class CSVExporter(BaseExporter):
    """CSV format exporter"""
    
    async def export_dataframe(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> ExportResult:
        start_time = datetime.now()
        output_path = self._prepare_output_path()
        
        try:
            # CSV-specific options
            csv_options = {
                "index": False,
                "encoding": "utf-8",
                **self.config.custom_options
            }
            
            # Handle compression
            if self.config.compression == CompressionType.GZIP:
                csv_options["compression"] = "gzip"
            elif self.config.compression == CompressionType.BZIP2:
                csv_options["compression"] = "bz2"
            elif self.config.compression == CompressionType.XZ:
                csv_options["compression"] = "xz"
            
            # Export DataFrame
            df.to_csv(output_path, **csv_options)
            
            # Write metadata file if requested
            metadata_path = None
            if self.config.include_metadata:
                metadata_path = output_path.with_suffix(".metadata.json")
                metadata_info = {
                    "schema": {
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "shape": df.shape
                    }
                }
                metadata_info = self._add_metadata(metadata_info, metadata)
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_info, f, indent=2, default=str)
            
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = output_path.stat().st_size
            output_files = [output_path]
            if metadata_path:
                output_files.append(metadata_path)
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=len(df),
                file_size_bytes=file_size,
                export_time_seconds=export_time,
                success=True,
                output_files=output_files,
                metadata={"original_shape": df.shape}
            )
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            export_time = (datetime.now() - start_time).total_seconds()
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=0,
                file_size_bytes=0,
                export_time_seconds=export_time,
                success=False,
                error_message=str(e)
            )
    
    async def export_streaming(
        self, 
        data_iterator: AsyncIterator[pd.DataFrame], 
        metadata: Optional[Dict] = None
    ) -> ExportResult:
        start_time = datetime.now()
        output_path = self._prepare_output_path()
        
        try:
            total_records = 0
            first_batch = True
            
            # CSV options
            csv_options = {
                "index": False,
                "encoding": "utf-8",
                "mode": "a",  # Append mode for streaming
                **self.config.custom_options
            }
            
            async for batch_df in data_iterator:
                # Write header only for first batch
                if first_batch:
                    csv_options["header"] = True
                    csv_options["mode"] = "w"  # Write mode for first batch
                    first_batch = False
                else:
                    csv_options["header"] = False
                    csv_options["mode"] = "a"  # Append mode for subsequent batches
                
                batch_df.to_csv(output_path, **csv_options)
                total_records += len(batch_df)
            
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = output_path.stat().st_size if output_path.exists() else 0
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=total_records,
                file_size_bytes=file_size,
                export_time_seconds=export_time,
                success=True,
                output_files=[output_path]
            )
            
        except Exception as e:
            logger.error(f"Streaming CSV export failed: {e}")
            export_time = (datetime.now() - start_time).total_seconds()
            
            return ExportResult(
                export_id=self.export_id,
                config=self.config,
                records_exported=0,
                file_size_bytes=0,
                export_time_seconds=export_time,
                success=False,
                error_message=str(e)
            )


class DataExporter:
    """
    Unified data export manager supporting multiple formats and destinations.
    Provides a simple interface for exporting data in various formats.
    """
    
    def __init__(self):
        self.exporters = {
            ExportFormat.JSON: JSONExporter,
            ExportFormat.JSONL: JSONExporter,  # Uses same exporter with different options
            ExportFormat.CSV: CSVExporter,
        }
    
    async def export_data(
        self, 
        data: Union[pd.DataFrame, AsyncIterator[pd.DataFrame]],
        config: ExportConfig,
        metadata: Optional[Dict] = None
    ) -> ExportResult:
        """
        Export data using the specified configuration
        
        Args:
            data: DataFrame or async iterator of DataFrames to export
            config: Export configuration
            metadata: Optional metadata to include with export
            
        Returns:
            ExportResult with operation details
        """
        
        if config.format not in self.exporters:
            raise ValueError(f"Unsupported export format: {config.format}")
        
        # Create appropriate exporter
        exporter_class = self.exporters[config.format]
        exporter = exporter_class(config)
        
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            return await exporter.export_dataframe(data, metadata)
        else:
            # Assume it's an async iterator
            return await exporter.export_streaming(data, metadata)
    
    async def export_multiple_formats(
        self,
        data: pd.DataFrame,
        output_dir: Path,
        formats: List[ExportFormat],
        base_filename: str = "export",
        metadata: Optional[Dict] = None
    ) -> List[ExportResult]:
        """
        Export the same data to multiple formats
        
        Args:
            data: DataFrame to export
            output_dir: Output directory
            formats: List of formats to export to
            base_filename: Base filename for exports
            metadata: Optional metadata
            
        Returns:
            List of ExportResults for each format
        """
        results = []
        
        for format_type in formats:
            try:
                config = ExportConfig(
                    format=format_type,
                    output_path=output_dir / f"{base_filename}_{format_type.value}",
                    include_metadata=True
                )
                
                result = await self.export_data(data, config, metadata)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to export to {format_type.value}: {e}")
                results.append(ExportResult(
                    export_id=f"failed_{format_type.value}",
                    config=config,
                    records_exported=0,
                    file_size_bytes=0,
                    export_time_seconds=0,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def create_data_package(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_dir: Path,
        package_name: str = "data_package",
        formats: Optional[List[ExportFormat]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, List[ExportResult]]:
        """
        Create a complete data package with multiple datasets and formats
        
        Args:
            datasets: Dictionary of dataset name to DataFrame
            output_dir: Output directory for the package
            package_name: Name of the data package
            formats: Formats to export (default: JSON, CSV)
            metadata: Package-level metadata
            
        Returns:
            Dictionary mapping dataset names to export results
        """
        if formats is None:
            formats = [ExportFormat.JSON, ExportFormat.CSV]
        
        package_dir = output_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Export each dataset
        for dataset_name, df in datasets.items():
            dataset_results = await self.export_multiple_formats(
                data=df,
                output_dir=package_dir,
                formats=formats,
                base_filename=dataset_name,
                metadata={
                    "dataset_name": dataset_name,
                    "package_name": package_name,
                    **(metadata or {})
                }
            )
            all_results[dataset_name] = dataset_results
        
        # Create package manifest
        manifest = {
            "package_name": package_name,
            "created_at": datetime.now().isoformat(),
            "datasets": {
                name: {
                    "record_count": len(df),
                    "columns": list(df.columns),
                    "formats": [f.value for f in formats]
                }
                for name, df in datasets.items()
            },
            "metadata": metadata
        }
        
        manifest_path = package_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        logger.info(f"Data package '{package_name}' created at {package_dir}")
        
        return all_results