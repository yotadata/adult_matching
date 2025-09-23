"""
Data Components Unit Tests

データ処理コンポーネントのユニットテスト
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Add backend to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.data.ingestion.data_ingestion_manager import (
    DataIngestionManager, 
    IngestionTask, 
    IngestionResult,
    DataSourceType,
    IngestionStatus
)
from backend.data.processing.unified_data_processor import (
    UnifiedDataProcessor,
    ProcessingConfig,
    ProcessingResult,
    ProcessingMode,
    ProcessingStage
)
from backend.data.validation.data_validator import (
    DataValidator,
    ValidationRule,
    ValidationResult,
    ValidationSeverity,
    RequiredFieldsRule,
    NullValueRule
)
from backend.data.export.data_exporter import (
    DataExporter,
    ExportConfig,
    ExportFormat,
    ExportResult,
    ExportStatus
)


@pytest.mark.unit
class TestDataIngestionManager:
    """DataIngestionManager ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        manager = DataIngestionManager()
        assert manager is not None
        assert hasattr(manager, 'ingest_data')
    
    @pytest.mark.asyncio
    async def test_ingest_api_data(self, sample_api_response):
        """API データ取り込みテスト"""
        manager = DataIngestionManager()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock API response
            mock_response = AsyncMock()
            mock_response.json.return_value = sample_api_response
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            task = IngestionTask(
                task_id="test_api_ingest",
                source_type=DataSourceType.API,
                source_config={
                    "api_url": "https://api.example.com/videos",
                    "api_key": "test_key"
                },
                output_path="test_output.json"
            )
            
            result = await manager.ingest_data(task)
            
            assert isinstance(result, IngestionResult)
            assert result.status == IngestionStatus.COMPLETED
            assert result.records_processed > 0
    
    @pytest.mark.asyncio
    async def test_ingest_file_data(self, temp_directory):
        """ファイルデータ取り込みテスト"""
        manager = DataIngestionManager()
        
        # Create test CSV file
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['Video 1', 'Video 2', 'Video 3'],
            'genre': ['Action', 'Comedy', 'Drama']
        })
        test_file = temp_directory / "test_data.csv"
        test_data.to_csv(test_file, index=False)
        
        task = IngestionTask(
            task_id="test_file_ingest",
            source_type=DataSourceType.FILE,
            source_config={
                "file_path": str(test_file),
                "file_format": "csv"
            }
        )
        
        result = await manager.ingest_data(task)
        
        assert result.status == IngestionStatus.COMPLETED
        assert result.records_processed == 3
    
    def test_validate_task(self):
        """タスク検証テスト"""
        manager = DataIngestionManager()
        
        # Valid task
        valid_task = IngestionTask(
            task_id="valid_task",
            source_type=DataSourceType.API,
            source_config={"api_url": "https://api.example.com"}
        )
        
        assert manager.validate_task(valid_task) is True
        
        # Invalid task (missing required config)
        invalid_task = IngestionTask(
            task_id="invalid_task",
            source_type=DataSourceType.API,
            source_config={}  # Missing required fields
        )
        
        assert manager.validate_task(invalid_task) is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """エラーハンドリングテスト"""
        manager = DataIngestionManager()
        
        # Invalid API URL
        task = IngestionTask(
            task_id="error_test",
            source_type=DataSourceType.API,
            source_config={
                "api_url": "invalid_url",
                "api_key": "test_key"
            }
        )
        
        result = await manager.ingest_data(task)
        
        assert result.status == IngestionStatus.FAILED
        assert result.error_message is not None


@pytest.mark.unit
class TestUnifiedDataProcessor:
    """UnifiedDataProcessor ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        processor = UnifiedDataProcessor()
        assert processor is not None
        assert hasattr(processor, 'process_data')
    
    @pytest.mark.asyncio
    async def test_process_data_batch_mode(self, sample_video_data, temp_directory):
        """バッチモードデータ処理テスト"""
        processor = UnifiedDataProcessor()
        
        # Save sample data to temp file
        input_file = temp_directory / "input_data.csv"
        sample_video_data.to_csv(input_file, index=False)
        
        config = ProcessingConfig(
            mode=ProcessingMode.BATCH,
            input_format="csv",
            output_format="json",
            stages=[ProcessingStage.CLEAN, ProcessingStage.TRANSFORM]
        )
        
        result = await processor.process_data(str(input_file), config=config)
        
        assert isinstance(result, ProcessingResult)
        assert result.status == "completed"
        assert result.records_processed == len(sample_video_data)
    
    def test_clean_data(self, sample_video_data):
        """データクリーニングテスト"""
        processor = UnifiedDataProcessor()
        
        # Add some dirty data
        dirty_data = sample_video_data.copy()
        dirty_data.loc[0, 'title'] = ''  # Empty title
        dirty_data.loc[1, 'price'] = -100  # Invalid price
        dirty_data.loc[2, 'duration'] = None  # Null duration
        
        cleaned_data = processor.clean_data(dirty_data)
        
        # Check cleaning results
        assert len(cleaned_data) <= len(dirty_data)  # May remove invalid records
        assert not cleaned_data['title'].str.strip().eq('').any()  # No empty titles
        assert (cleaned_data['price'] >= 0).all()  # No negative prices
    
    def test_transform_data(self, sample_video_data):
        """データ変換テスト"""
        processor = UnifiedDataProcessor()
        
        transformed_data = processor.transform_data(sample_video_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(sample_video_data)
        
        # Check if new columns were added
        original_columns = set(sample_video_data.columns)
        transformed_columns = set(transformed_data.columns)
        assert len(transformed_columns) >= len(original_columns)
    
    def test_enrich_data(self, sample_video_data):
        """データエンリッチメントテスト"""
        processor = UnifiedDataProcessor()
        
        enriched_data = processor.enrich_data(sample_video_data)
        
        assert isinstance(enriched_data, pd.DataFrame)
        assert len(enriched_data) == len(sample_video_data)
        
        # Check if enrichment added meaningful data
        assert 'popularity_score' in enriched_data.columns
        assert 'genre_encoded' in enriched_data.columns
    
    @pytest.mark.asyncio
    async def test_streaming_mode(self, sample_video_data):
        """ストリーミングモード処理テスト"""
        processor = UnifiedDataProcessor()
        
        # Mock streaming data
        async def mock_data_stream():
            chunk_size = 10
            for i in range(0, len(sample_video_data), chunk_size):
                yield sample_video_data.iloc[i:i+chunk_size]
        
        config = ProcessingConfig(
            mode=ProcessingMode.STREAMING,
            batch_size=10,
            stages=[ProcessingStage.CLEAN]
        )
        
        results = []
        async for result_chunk in processor.process_streaming(mock_data_stream(), config):
            results.append(result_chunk)
        
        assert len(results) > 0
        total_processed = sum(len(chunk) for chunk in results)
        assert total_processed <= len(sample_video_data)


@pytest.mark.unit
class TestDataValidator:
    """DataValidator ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        validator = DataValidator()
        assert validator is not None
        assert len(validator.rules) == 0
    
    def test_add_required_fields_rule(self):
        """必須フィールドルール追加テスト"""
        validator = DataValidator()
        
        validator.add_required_fields_rule(['title', 'external_id'])
        
        assert len(validator.rules) == 1
        assert isinstance(validator.rules[0], RequiredFieldsRule)
    
    def test_add_null_check_rule(self):
        """NULL チェックルール追加テスト"""
        validator = DataValidator()
        
        validator.add_null_check_rule(['title', 'price'], max_null_ratio=0.1)
        
        assert len(validator.rules) == 1
        assert isinstance(validator.rules[0], NullValueRule)
    
    @pytest.mark.asyncio
    async def test_validate_data_success(self, sample_video_data):
        """データ検証成功テスト"""
        validator = DataValidator()
        
        # Add validation rules
        validator.add_required_fields_rule(['external_id', 'title'])
        validator.add_null_check_rule(['title'], max_null_ratio=0.0)
        
        report = await validator.validate(sample_video_data, data_source="test")
        
        assert report.passed is True
        assert len(report.results) == 2
        assert all(result.passed for result in report.results)
    
    @pytest.mark.asyncio
    async def test_validate_data_failure(self):
        """データ検証失敗テスト"""
        validator = DataValidator()
        
        # Create invalid data
        invalid_data = pd.DataFrame({
            'title': ['Valid Title', None, ''],  # Null and empty values
            'price': [1000, -500, 2000]  # Negative price
        })
        
        validator.add_required_fields_rule(['external_id'])  # Missing field
        validator.add_null_check_rule(['title'], max_null_ratio=0.0)
        
        report = await validator.validate(invalid_data, data_source="test")
        
        assert report.passed is False
        assert len(report.results) > 0
        assert any(not result.passed for result in report.results)
    
    def test_create_video_validator(self):
        """動画データ用バリデータ作成テスト"""
        validator = DataValidator()
        video_validator = validator.create_video_validator()
        
        assert isinstance(video_validator, DataValidator)
        assert len(video_validator.rules) > 0


@pytest.mark.unit  
class TestDataExporter:
    """DataExporter ユニットテスト"""
    
    def test_init(self):
        """初期化テスト"""
        exporter = DataExporter()
        assert exporter is not None
        assert hasattr(exporter, 'export_data')
    
    @pytest.mark.asyncio
    async def test_export_csv(self, sample_video_data, temp_directory):
        """CSV エクスポートテスト"""
        exporter = DataExporter()
        
        output_file = temp_directory / "export_test.csv"
        config = ExportConfig(
            format=ExportFormat.CSV,
            output_path=output_file,
            include_index=False
        )
        
        result = await exporter.export_data(sample_video_data, config)
        
        assert isinstance(result, ExportResult)
        assert result.status == ExportStatus.COMPLETED
        assert result.exported_records == len(sample_video_data)
        assert output_file.exists()
        
        # Verify exported data
        exported_data = pd.read_csv(output_file)
        assert len(exported_data) == len(sample_video_data)
    
    @pytest.mark.asyncio
    async def test_export_json(self, sample_video_data, temp_directory):
        """JSON エクスポートテスト"""
        exporter = DataExporter()
        
        output_file = temp_directory / "export_test.json"
        config = ExportConfig(
            format=ExportFormat.JSON,
            output_path=output_file
        )
        
        result = await exporter.export_data(sample_video_data, config)
        
        assert result.status == ExportStatus.COMPLETED
        assert output_file.exists()
    
    @pytest.mark.asyncio 
    async def test_export_with_filters(self, sample_video_data, temp_directory):
        """フィルタ付きエクスポートテスト"""
        exporter = DataExporter()
        
        output_file = temp_directory / "filtered_export.json"
        config = ExportConfig(
            format=ExportFormat.JSON,
            output_path=output_file,
            columns=['external_id', 'title', 'price'],
            query_filter="price > 1000"
        )
        
        result = await exporter.export_data(sample_video_data, config)
        
        assert result.status == ExportStatus.COMPLETED
        assert result.exported_records <= len(sample_video_data)
    
    def test_get_export_status(self):
        """エクスポート状況取得テスト"""
        exporter = DataExporter()
        
        # No export yet
        status = exporter.get_export_status("non_existent_id")
        assert status is None
    
    def test_cancel_export(self):
        """エクスポートキャンセルテスト"""
        exporter = DataExporter()
        
        # No running export to cancel
        result = exporter.cancel_export("non_existent_id")
        assert result is False


@pytest.mark.integration
class TestDataComponentsIntegration:
    """データコンポーネント統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_data_pipeline(self, temp_directory):
        """完全データパイプライン統合テスト"""
        # Setup components
        ingestion_manager = DataIngestionManager()
        processor = UnifiedDataProcessor()
        validator = DataValidator()
        exporter = DataExporter()
        
        # Create test data file
        test_data = pd.DataFrame({
            'external_id': ['vid_1', 'vid_2', 'vid_3'],
            'title': ['Video 1', 'Video 2', 'Video 3'],
            'price': [1000, 1500, 2000],
            'genre': ['Action', 'Comedy', 'Drama'],
            'source': ['dmm', 'dmm', 'dmm']
        })
        input_file = temp_directory / "input.csv"
        test_data.to_csv(input_file, index=False)
        
        # Step 1: Ingestion
        ingest_task = IngestionTask(
            task_id="integration_test",
            source_type=DataSourceType.FILE,
            source_config={
                "file_path": str(input_file),
                "file_format": "csv"
            }
        )
        
        ingest_result = await ingestion_manager.ingest_data(ingest_task)
        assert ingest_result.status == IngestionStatus.COMPLETED
        
        # Step 2: Processing
        process_config = ProcessingConfig(
            mode=ProcessingMode.BATCH,
            stages=[ProcessingStage.CLEAN, ProcessingStage.TRANSFORM]
        )
        
        process_result = await processor.process_data(str(input_file), config=process_config)
        assert process_result.status == "completed"
        
        # Step 3: Validation
        validator.add_required_fields_rule(['external_id', 'title'])
        validator.add_null_check_rule(['title'], max_null_ratio=0.0)
        
        validation_report = await validator.validate(test_data, data_source="integration_test")
        assert validation_report.passed is True
        
        # Step 4: Export
        output_file = temp_directory / "final_output.json"
        export_config = ExportConfig(
            format=ExportFormat.JSON,
            output_path=output_file
        )
        
        export_result = await exporter.export_data(test_data, export_config)
        assert export_result.status == ExportStatus.COMPLETED
        assert output_file.exists()
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self):
        """大規模データセット処理テスト"""
        processor = UnifiedDataProcessor()
        
        # Generate large dataset
        large_data = pd.DataFrame({
            'external_id': [f'video_{i}' for i in range(10000)],
            'title': [f'Video Title {i}' for i in range(10000)],
            'price': np.random.randint(100, 5000, 10000),
            'genre': np.random.choice(['Action', 'Comedy', 'Drama'], 10000),
            'source': ['dmm'] * 10000
        })
        
        config = ProcessingConfig(
            mode=ProcessingMode.BATCH,
            batch_size=1000,
            stages=[ProcessingStage.CLEAN, ProcessingStage.TRANSFORM]
        )
        
        # This should complete within reasonable time
        import time
        start_time = time.time()
        
        result = await processor.process_data(large_data, config=config)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert result.status == "completed" 
        assert result.records_processed == 10000
        assert processing_time < 60  # Should complete within 1 minute