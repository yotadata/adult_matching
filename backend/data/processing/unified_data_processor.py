"""
Unified Data Processor

統合データ処理システム
- データクリーニング・変換・正規化
- 特徴量エンジニアリング統合
- バッチ・ストリーミング処理対応
- パイプライン統合インターface
"""

import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

from backend.ml.utils.logger import get_ml_logger
from backend.ml.preprocessing import FeatureProcessor, FeatureConfig

logger = get_ml_logger(__name__)

class ProcessingMode(Enum):
    """処理モード"""
    BATCH = "batch"
    STREAMING = "streaming" 
    HYBRID = "hybrid"

class ProcessingStage(Enum):
    """処理ステージ"""
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    VALIDATION = "validation"
    FEATURE_ENGINEERING = "feature_engineering"

@dataclass
class ProcessingConfig:
    """データ処理設定"""
    mode: ProcessingMode = ProcessingMode.BATCH
    batch_size: int = 1000
    parallel_workers: int = 4
    enable_caching: bool = True
    quality_threshold: float = 0.8
    stages: List[ProcessingStage] = None
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = [
                ProcessingStage.CLEANING,
                ProcessingStage.TRANSFORMATION,
                ProcessingStage.ENRICHMENT,
                ProcessingStage.VALIDATION
            ]

@dataclass
class ProcessingResult:
    """処理結果"""
    job_id: str
    success: bool
    input_records: int
    output_records: int
    processing_time_seconds: float
    quality_score: float
    stages_completed: List[ProcessingStage]
    output_files: List[Path]
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = None

class UnifiedDataProcessor:
    """統合データ処理システム"""
    
    def __init__(
        self,
        processors: List[str] = None,
        input_dir: Path = None,
        output_dir: Path = None,
        config: ProcessingConfig = None
    ):
        self.processors = processors or ["clean", "transform", "enrich"]
        self.input_dir = input_dir or Path("data/raw")
        self.output_dir = output_dir or Path("data/processed")
        self.config = config or ProcessingConfig()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 特徴量処理器の初期化
        feature_config = FeatureConfig(target_dimension=768)
        self.feature_processor = FeatureProcessor(feature_config)
        
        # 処理ステージ関数のマッピング
        self.stage_processors = {
            ProcessingStage.CLEANING: self._clean_data,
            ProcessingStage.TRANSFORMATION: self._transform_data,
            ProcessingStage.ENRICHMENT: self._enrich_data,
            ProcessingStage.VALIDATION: self._validate_data,
            ProcessingStage.FEATURE_ENGINEERING: self._engineer_features
        }
        
        # 処理履歴
        self.processing_history: Dict[str, ProcessingResult] = {}
    
    async def process_data(
        self,
        input_path: Union[Path, str],
        job_id: Optional[str] = None,
        custom_config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """データ処理メイン関数"""
        
        if job_id is None:
            job_id = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = custom_config or self.config
        input_path = Path(input_path)
        
        logger.info(f"Starting data processing job: {job_id}")
        logger.info(f"Input: {input_path}, Mode: {config.mode.value}")
        
        start_time = datetime.now()
        
        try:
            # 入力データの読み込み
            input_data = await self._load_input_data(input_path)
            input_records = len(input_data)
            
            logger.info(f"Loaded {input_records} records for processing")
            
            # 処理モード別の実行
            if config.mode == ProcessingMode.BATCH:
                processed_data, stages_completed = await self._process_batch(input_data, config)
            elif config.mode == ProcessingMode.STREAMING:
                processed_data, stages_completed = await self._process_streaming(input_data, config)
            else:  # HYBRID
                processed_data, stages_completed = await self._process_hybrid(input_data, config)
            
            # 結果の保存
            output_files = await self._save_processed_data(processed_data, job_id)
            
            # 品質評価
            quality_score = self._assess_processing_quality(processed_data, input_data)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 結果の作成
            result = ProcessingResult(
                job_id=job_id,
                success=True,
                input_records=input_records,
                output_records=len(processed_data),
                processing_time_seconds=processing_time,
                quality_score=quality_score,
                stages_completed=stages_completed,
                output_files=output_files,
                metadata={
                    "config": asdict(config),
                    "started_at": start_time.isoformat(),
                    "completed_at": end_time.isoformat()
                }
            )
            
            self.processing_history[job_id] = result
            
            logger.info(f"Processing job {job_id} completed successfully")
            logger.info(f"Quality score: {quality_score:.2f}, Time: {processing_time:.2f}s")
            
            return result
            
        except Exception as error:
            logger.error(f"Processing job {job_id} failed: {error}")
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = ProcessingResult(
                job_id=job_id,
                success=False,
                input_records=len(input_data) if 'input_data' in locals() else 0,
                output_records=0,
                processing_time_seconds=processing_time,
                quality_score=0.0,
                stages_completed=[],
                output_files=[],
                error_details=str(error)
            )
            
            self.processing_history[job_id] = result
            return result
    
    async def _load_input_data(self, input_path: Path) -> List[Dict[str, Any]]:
        """入力データの読み込み"""
        logger.info(f"Loading input data from: {input_path}")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        try:
            if input_path.suffix.lower() == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif input_path.suffix.lower() == '.csv':
                df = pd.read_csv(input_path)
                data = df.to_dict('records')
            elif input_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(input_path)
                data = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")
            
            # データが辞書のリストでない場合は変換
            if not isinstance(data, list):
                if isinstance(data, dict):
                    data = [data]
                else:
                    data = [{"data": item} for item in data]
            
            logger.info(f"Successfully loaded {len(data)} records")
            return data
            
        except Exception as error:
            logger.error(f"Failed to load input data: {error}")
            raise
    
    async def _process_batch(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> tuple[List[Dict[str, Any]], List[ProcessingStage]]:
        """バッチ処理"""
        logger.info(f"Processing {len(data)} records in batch mode")
        
        processed_data = data.copy()
        stages_completed = []
        
        # 各ステージの順次実行
        for stage in config.stages:
            logger.info(f"Executing stage: {stage.value}")
            
            processor_func = self.stage_processors[stage]
            processed_data = await processor_func(processed_data, config)
            stages_completed.append(stage)
            
            logger.info(f"Stage {stage.value} completed, {len(processed_data)} records remaining")
        
        return processed_data, stages_completed
    
    async def _process_streaming(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> tuple[List[Dict[str, Any]], List[ProcessingStage]]:
        """ストリーミング処理"""
        logger.info(f"Processing {len(data)} records in streaming mode")
        
        processed_data = []
        stages_completed = []
        batch_size = config.batch_size
        
        # バッチごとのストリーミング処理
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
            
            batch_processed = batch.copy()
            batch_stages = []
            
            for stage in config.stages:
                if stage not in stages_completed:
                    processor_func = self.stage_processors[stage]
                    batch_processed = await processor_func(batch_processed, config)
                    if stage not in batch_stages:
                        batch_stages.append(stage)
            
            processed_data.extend(batch_processed)
            stages_completed = list(set(stages_completed + batch_stages))
        
        return processed_data, stages_completed
    
    async def _process_hybrid(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> tuple[List[Dict[str, Any]], List[ProcessingStage]]:
        """ハイブリッド処理"""
        logger.info(f"Processing {len(data)} records in hybrid mode")
        
        # 小さなデータセットはバッチ、大きなデータセットはストリーミング
        threshold = config.batch_size * 10
        
        if len(data) <= threshold:
            return await self._process_batch(data, config)
        else:
            return await self._process_streaming(data, config)
    
    async def _clean_data(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> List[Dict[str, Any]]:
        """データクリーニング"""
        logger.info("Starting data cleaning stage")
        
        cleaned_data = []
        
        for record in data:
            cleaned_record = record.copy()
            
            # 空の値やNoneの処理
            for key, value in cleaned_record.items():
                if value is None or value == "":
                    if key in ["id", "title"]:
                        # 必須フィールドが空の場合はスキップ
                        break
                    else:
                        cleaned_record[key] = "Unknown" if isinstance(value, str) else 0
            else:
                # 全てのフィールドが有効な場合のみ追加
                # 重複除去
                if not any(existing.get("id") == cleaned_record.get("id") for existing in cleaned_data):
                    cleaned_data.append(cleaned_record)
        
        logger.info(f"Data cleaning completed: {len(data)} -> {len(cleaned_data)} records")
        return cleaned_data
    
    async def _transform_data(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> List[Dict[str, Any]]:
        """データ変換"""
        logger.info("Starting data transformation stage")
        
        transformed_data = []
        
        for record in data:
            transformed_record = record.copy()
            
            # 日付フィールドの標準化
            for key, value in transformed_record.items():
                if "date" in key.lower() or "time" in key.lower():
                    if isinstance(value, str) and value != "Unknown":
                        try:
                            # ISO形式に変換
                            dt = pd.to_datetime(value)
                            transformed_record[key] = dt.isoformat()
                        except:
                            pass
                
                # 数値フィールドの変換
                elif key in ["price", "rating", "duration"]:
                    try:
                        transformed_record[key] = float(value) if value not in [None, "", "Unknown"] else 0.0
                    except (ValueError, TypeError):
                        transformed_record[key] = 0.0
            
            # メタデータの追加
            transformed_record["processed_at"] = datetime.now().isoformat()
            transformed_record["processing_version"] = "2.0.0"
            
            transformed_data.append(transformed_record)
        
        logger.info(f"Data transformation completed: {len(transformed_data)} records")
        return transformed_data
    
    async def _enrich_data(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> List[Dict[str, Any]]:
        """データエンリッチメント"""
        logger.info("Starting data enrichment stage")
        
        enriched_data = []
        
        # ジャンル・メーカー統計の計算
        genre_counts = {}
        maker_counts = {}
        
        for record in data:
            genre = record.get("genre", "Unknown")
            maker = record.get("maker", "Unknown")
            
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            maker_counts[maker] = maker_counts.get(maker, 0) + 1
        
        for record in data:
            enriched_record = record.copy()
            
            # 統計情報の追加
            genre = enriched_record.get("genre", "Unknown")
            maker = enriched_record.get("maker", "Unknown")
            
            enriched_record["genre_popularity"] = genre_counts.get(genre, 0)
            enriched_record["maker_popularity"] = maker_counts.get(maker, 0)
            
            # 品質スコアの計算
            quality_score = 1.0
            if enriched_record.get("title", "") == "Unknown":
                quality_score -= 0.3
            if enriched_record.get("description", "") in ["", "Unknown"]:
                quality_score -= 0.2
            if enriched_record.get("rating", 0) == 0:
                quality_score -= 0.1
            
            enriched_record["quality_score"] = max(0.0, quality_score)
            
            enriched_data.append(enriched_record)
        
        logger.info(f"Data enrichment completed: {len(enriched_data)} records")
        return enriched_data
    
    async def _validate_data(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> List[Dict[str, Any]]:
        """データ検証"""
        logger.info("Starting data validation stage")
        
        validated_data = []
        validation_errors = []
        
        required_fields = ["id", "title"]
        
        for i, record in enumerate(data):
            is_valid = True
            errors = []
            
            # 必須フィールドのチェック
            for field in required_fields:
                if field not in record or not record[field]:
                    is_valid = False
                    errors.append(f"Missing required field: {field}")
            
            # データ型の検証
            if "rating" in record:
                try:
                    rating = float(record["rating"])
                    if not (0 <= rating <= 5):
                        errors.append("Rating out of range (0-5)")
                except (ValueError, TypeError):
                    errors.append("Invalid rating format")
            
            # 品質しきい値チェック
            quality_score = record.get("quality_score", 0)
            if quality_score < config.quality_threshold:
                is_valid = False
                errors.append(f"Quality score below threshold: {quality_score}")
            
            if is_valid:
                validated_data.append(record)
            else:
                validation_errors.append({
                    "record_index": i,
                    "errors": errors,
                    "record": record
                })
        
        if validation_errors:
            logger.warning(f"Validation found {len(validation_errors)} invalid records")
            # エラー詳細をログに記録（デバッグ用）
            for error in validation_errors[:5]:  # 最初の5件のみログ出力
                logger.debug(f"Validation error: {error}")
        
        logger.info(f"Data validation completed: {len(data)} -> {len(validated_data)} records")
        return validated_data
    
    async def _engineer_features(
        self,
        data: List[Dict[str, Any]],
        config: ProcessingConfig
    ) -> List[Dict[str, Any]]:
        """特徴量エンジニアリング"""
        logger.info("Starting feature engineering stage")
        
        try:
            # DataFrameに変換
            df = pd.DataFrame(data)
            
            # 特徴量処理器によるエンジニアリング
            if len(df) > 0:
                # フィット・トランスフォーム
                self.feature_processor.fit(df)
                engineered_features = self.feature_processor.transform(df)
                
                # 元のデータに特徴量を追加
                for i, record in enumerate(data):
                    if i < len(engineered_features):
                        record["engineered_features"] = engineered_features[i].tolist()
                        record["feature_dimension"] = len(engineered_features[i])
            
            logger.info(f"Feature engineering completed: {len(data)} records")
            return data
            
        except Exception as error:
            logger.warning(f"Feature engineering failed: {error}, skipping stage")
            return data
    
    def _assess_processing_quality(
        self,
        processed_data: List[Dict[str, Any]],
        original_data: List[Dict[str, Any]]
    ) -> float:
        """処理品質の評価"""
        if not processed_data or not original_data:
            return 0.0
        
        # データ保持率
        retention_rate = len(processed_data) / len(original_data)
        
        # フィールド完成度
        total_fields = sum(len(record) for record in processed_data)
        completed_fields = sum(
            sum(1 for v in record.values() if v not in [None, "", "Unknown"])
            for record in processed_data
        )
        completion_rate = completed_fields / total_fields if total_fields > 0 else 0
        
        # 品質スコア平均
        quality_scores = [record.get("quality_score", 0.5) for record in processed_data]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # 総合品質スコア
        overall_quality = (retention_rate * 0.3 + completion_rate * 0.4 + avg_quality * 0.3)
        
        logger.info(f"Processing quality assessment - Retention: {retention_rate:.2f}, "
                   f"Completion: {completion_rate:.2f}, Quality: {avg_quality:.2f}")
        
        return overall_quality
    
    async def _save_processed_data(
        self,
        data: List[Dict[str, Any]],
        job_id: str
    ) -> List[Path]:
        """処理済みデータの保存"""
        output_files = []
        
        # JSONファイルとして保存
        json_file = self.output_dir / f"{job_id}_processed.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        output_files.append(json_file)
        
        # CSVファイルとしても保存（特徴量分析用）
        try:
            df = pd.DataFrame(data)
            csv_file = self.output_dir / f"{job_id}_processed.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            output_files.append(csv_file)
        except Exception as error:
            logger.warning(f"Failed to save CSV file: {error}")
        
        logger.info(f"Saved processed data to {len(output_files)} files")
        return output_files
    
    def get_processing_history(self, limit: int = 20) -> List[ProcessingResult]:
        """処理履歴の取得"""
        results = list(self.processing_history.values())
        results.sort(key=lambda r: r.metadata.get("completed_at", ""), reverse=True)
        return results[:limit]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計の取得"""
        results = list(self.processing_history.values())
        successful_results = [r for r in results if r.success]
        
        if not results:
            return {"status": "no_processing_history"}
        
        return {
            "total_jobs": len(results),
            "successful_jobs": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "total_records_processed": sum(r.input_records for r in successful_results),
            "total_records_output": sum(r.output_records for r in successful_results),
            "avg_processing_time": sum(r.processing_time_seconds for r in successful_results) / len(successful_results) if successful_results else 0,
            "avg_quality_score": sum(r.quality_score for r in successful_results) / len(successful_results) if successful_results else 0,
            "last_processing": max(r.metadata.get("completed_at", "") for r in results if r.metadata)
        }