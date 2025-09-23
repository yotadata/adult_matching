"""
Data Quality Assessor

データ品質評価システム - 完全性、正確性、一貫性、適時性評価
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class QualityDimension(Enum):
    """データ品質次元"""
    COMPLETENESS = "completeness"      # 完全性
    ACCURACY = "accuracy"              # 正確性
    CONSISTENCY = "consistency"        # 一貫性
    TIMELINESS = "timeliness"          # 適時性
    VALIDITY = "validity"              # 妥当性
    UNIQUENESS = "uniqueness"          # 一意性


@dataclass
class QualityMetrics:
    """品質メトリクス"""
    dimension: QualityDimension
    score: float  # 0.0-1.0
    details: Dict[str, Any]
    measurement_time: datetime = field(default_factory=datetime.now)


@dataclass
class QualityThresholds:
    """品質しきい値設定"""
    completeness_min: float = 0.95
    accuracy_min: float = 0.90
    consistency_min: float = 0.85
    timeliness_max_days: int = 7
    validity_min: float = 0.90
    uniqueness_min: float = 0.99


@dataclass
class QualityReport:
    """品質評価レポート"""
    dataset_name: str
    assessment_time: datetime
    total_records: int
    metrics: List[QualityMetrics]
    overall_score: float
    thresholds: QualityThresholds
    passed: bool
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def dimension_scores(self) -> Dict[str, float]:
        """次元別スコア"""
        return {metric.dimension.value: metric.score for metric in self.metrics}
    
    @property
    def failed_dimensions(self) -> List[str]:
        """失敗した品質次元"""
        failed = []
        for metric in self.metrics:
            if metric.dimension == QualityDimension.COMPLETENESS and metric.score < self.thresholds.completeness_min:
                failed.append(metric.dimension.value)
            elif metric.dimension == QualityDimension.ACCURACY and metric.score < self.thresholds.accuracy_min:
                failed.append(metric.dimension.value)
            elif metric.dimension == QualityDimension.CONSISTENCY and metric.score < self.thresholds.consistency_min:
                failed.append(metric.dimension.value)
            elif metric.dimension == QualityDimension.VALIDITY and metric.score < self.thresholds.validity_min:
                failed.append(metric.dimension.value)
            elif metric.dimension == QualityDimension.UNIQUENESS and metric.score < self.thresholds.uniqueness_min:
                failed.append(metric.dimension.value)
        return failed


class DataQualityAssessor:
    """データ品質評価システム"""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.logger = logging.getLogger(__name__)
    
    async def assess_quality(self, data: pd.DataFrame, dataset_name: str = "unknown", 
                           key_fields: Optional[List[str]] = None,
                           timestamp_field: Optional[str] = None) -> QualityReport:
        """包括的品質評価"""
        metrics = []
        
        # 各次元評価
        metrics.append(await self._assess_completeness(data))
        metrics.append(await self._assess_accuracy(data, key_fields))
        metrics.append(await self._assess_consistency(data))
        metrics.append(await self._assess_validity(data))
        metrics.append(await self._assess_uniqueness(data, key_fields))
        
        if timestamp_field:
            metrics.append(await self._assess_timeliness(data, timestamp_field))
        
        # 総合スコア計算
        overall_score = np.mean([metric.score for metric in metrics])
        
        # 合格判定
        passed = self._determine_pass_status(metrics)
        
        # 推奨事項生成
        recommendations = self._generate_recommendations(metrics)
        
        return QualityReport(
            dataset_name=dataset_name,
            assessment_time=datetime.now(),
            total_records=len(data),
            metrics=metrics,
            overall_score=overall_score,
            thresholds=self.thresholds,
            passed=passed,
            recommendations=recommendations
        )
    
    async def _assess_completeness(self, data: pd.DataFrame) -> QualityMetrics:
        """完全性評価"""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness_score = 1.0 - (missing_cells / total_cells)
        
        # フィールド別完全性
        field_completeness = {}
        for col in data.columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            field_completeness[col] = 1.0 - missing_ratio
        
        details = {
            "total_cells": total_cells,
            "missing_cells": int(missing_cells),
            "field_completeness": field_completeness,
            "worst_fields": sorted(field_completeness.items(), key=lambda x: x[1])[:5]
        }
        
        return QualityMetrics(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            details=details
        )
    
    async def _assess_accuracy(self, data: pd.DataFrame, key_fields: Optional[List[str]] = None) -> QualityMetrics:
        """正確性評価"""
        accuracy_checks = {}
        
        # 数値フィールドの統計的異常値検出
        for col in data.select_dtypes(include=[np.number]).columns:
            values = data[col].dropna()
            if len(values) > 0:
                z_scores = np.abs(stats.zscore(values))
                outlier_ratio = (z_scores > 3).sum() / len(values)
                accuracy_checks[f"{col}_outliers"] = 1.0 - outlier_ratio
        
        # 文字列フィールドの一貫性チェック
        for col in data.select_dtypes(include=['object']).columns:
            values = data[col].dropna().astype(str)
            if len(values) > 0:
                # 異常に短い/長い値の検出
                lengths = values.str.len()
                q1, q3 = lengths.quantile([0.25, 0.75])
                iqr = q3 - q1
                abnormal_length = ((lengths < q1 - 1.5 * iqr) | (lengths > q3 + 1.5 * iqr)).sum()
                accuracy_checks[f"{col}_length_anomalies"] = 1.0 - (abnormal_length / len(values))
        
        # キーフィールド形式チェック
        if key_fields:
            for field in key_fields:
                if field in data.columns:
                    # 一般的な形式チェック（簡略化）
                    values = data[field].dropna().astype(str)
                    valid_format = values.str.match(r'^[a-zA-Z0-9_-]+$').sum()
                    accuracy_checks[f"{field}_format"] = valid_format / len(values) if len(values) > 0 else 1.0
        
        overall_accuracy = np.mean(list(accuracy_checks.values())) if accuracy_checks else 1.0
        
        return QualityMetrics(
            dimension=QualityDimension.ACCURACY,
            score=overall_accuracy,
            details={"checks": accuracy_checks}
        )
    
    async def _assess_consistency(self, data: pd.DataFrame) -> QualityMetrics:
        """一貫性評価"""
        consistency_checks = {}
        
        # データ型一貫性
        type_consistency = {}
        for col in data.columns:
            # 同じ列内での型の統一性
            non_null_values = data[col].dropna()
            if len(non_null_values) > 0:
                type_counts = non_null_values.apply(type).value_counts()
                dominant_type_ratio = type_counts.max() / len(non_null_values)
                type_consistency[col] = dominant_type_ratio
        
        consistency_checks["type_consistency"] = np.mean(list(type_consistency.values()))
        
        # 値の分布一貫性（カテゴリカルデータ）
        categorical_consistency = {}
        for col in data.select_dtypes(include=['object']).columns:
            values = data[col].dropna()
            if len(values) > 10:  # 十分なデータがある場合
                value_counts = values.value_counts()
                # エントロピーベースの一貫性評価
                probabilities = value_counts / len(values)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                max_entropy = np.log2(len(value_counts))
                consistency_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
                categorical_consistency[col] = consistency_score
        
        if categorical_consistency:
            consistency_checks["categorical_consistency"] = np.mean(list(categorical_consistency.values()))
        
        overall_consistency = np.mean(list(consistency_checks.values())) if consistency_checks else 1.0
        
        return QualityMetrics(
            dimension=QualityDimension.CONSISTENCY,
            score=overall_consistency,
            details={
                "checks": consistency_checks,
                "type_consistency": type_consistency,
                "categorical_consistency": categorical_consistency
            }
        )
    
    async def _assess_validity(self, data: pd.DataFrame) -> QualityMetrics:
        """妥当性評価"""
        validity_checks = {}
        
        # 数値フィールドの範囲妥当性
        for col in data.select_dtypes(include=[np.number]).columns:
            values = data[col].dropna()
            if len(values) > 0:
                # 負の値が不適切なフィールドの推定
                if col.lower() in ['price', 'duration', 'rating', 'count', 'age']:
                    negative_ratio = (values < 0).sum() / len(values)
                    validity_checks[f"{col}_non_negative"] = 1.0 - negative_ratio
                
                # 異常に大きな値の検出
                q99 = values.quantile(0.99)
                extreme_values = (values > q99 * 100).sum()
                validity_checks[f"{col}_reasonable_range"] = 1.0 - (extreme_values / len(values))
        
        # 文字列フィールドの妥当性
        for col in data.select_dtypes(include=['object']).columns:
            values = data[col].dropna().astype(str)
            if len(values) > 0:
                # 空文字列やスペースのみの値
                empty_or_space = (values.str.strip().str.len() == 0).sum()
                validity_checks[f"{col}_non_empty"] = 1.0 - (empty_or_space / len(values))
                
                # 異常に長い文字列
                very_long = (values.str.len() > 1000).sum()
                validity_checks[f"{col}_reasonable_length"] = 1.0 - (very_long / len(values))
        
        overall_validity = np.mean(list(validity_checks.values())) if validity_checks else 1.0
        
        return QualityMetrics(
            dimension=QualityDimension.VALIDITY,
            score=overall_validity,
            details={"checks": validity_checks}
        )
    
    async def _assess_uniqueness(self, data: pd.DataFrame, key_fields: Optional[List[str]] = None) -> QualityMetrics:
        """一意性評価"""
        uniqueness_checks = {}
        
        # 全レコードの一意性
        total_duplicates = data.duplicated().sum()
        overall_uniqueness = 1.0 - (total_duplicates / len(data))
        uniqueness_checks["overall"] = overall_uniqueness
        
        # キーフィールドの一意性
        if key_fields:
            for field in key_fields:
                if field in data.columns:
                    field_duplicates = data[field].duplicated().sum()
                    field_uniqueness = 1.0 - (field_duplicates / len(data))
                    uniqueness_checks[field] = field_uniqueness
        
        # 個別フィールドの多様性
        diversity_scores = {}
        for col in data.columns:
            unique_values = data[col].nunique()
            total_values = len(data[col].dropna())
            diversity = unique_values / total_values if total_values > 0 else 0
            diversity_scores[col] = diversity
        
        overall_uniqueness = np.mean(list(uniqueness_checks.values())) if uniqueness_checks else 1.0
        
        return QualityMetrics(
            dimension=QualityDimension.UNIQUENESS,
            score=overall_uniqueness,
            details={
                "checks": uniqueness_checks,
                "diversity_scores": diversity_scores
            }
        )
    
    async def _assess_timeliness(self, data: pd.DataFrame, timestamp_field: str) -> QualityMetrics:
        """適時性評価"""
        if timestamp_field not in data.columns:
            return QualityMetrics(
                dimension=QualityDimension.TIMELINESS,
                score=0.0,
                details={"error": "Timestamp field not found"}
            )
        
        try:
            timestamps = pd.to_datetime(data[timestamp_field], errors='coerce')
            current_time = datetime.now()
            
            # 有効なタイムスタンプの割合
            valid_timestamps = timestamps.notna().sum()
            validity_ratio = valid_timestamps / len(data)
            
            if valid_timestamps == 0:
                return QualityMetrics(
                    dimension=QualityDimension.TIMELINESS,
                    score=0.0,
                    details={"error": "No valid timestamps found"}
                )
            
            # 最新性評価
            latest_timestamp = timestamps.max()
            days_old = (current_time - latest_timestamp).days
            recency_score = max(0, 1.0 - (days_old / (self.thresholds.timeliness_max_days * 2)))
            
            # データ鮮度分布
            valid_timestamps_series = timestamps.dropna()
            age_distribution = {}
            for days in [1, 7, 30, 90]:
                cutoff_date = current_time - timedelta(days=days)
                recent_count = (valid_timestamps_series >= cutoff_date).sum()
                age_distribution[f"within_{days}_days"] = recent_count / len(valid_timestamps_series)
            
            overall_timeliness = validity_ratio * recency_score
            
            return QualityMetrics(
                dimension=QualityDimension.TIMELINESS,
                score=overall_timeliness,
                details={
                    "validity_ratio": validity_ratio,
                    "recency_score": recency_score,
                    "days_old": days_old,
                    "latest_timestamp": latest_timestamp.isoformat() if latest_timestamp else None,
                    "age_distribution": age_distribution
                }
            )
            
        except Exception as e:
            return QualityMetrics(
                dimension=QualityDimension.TIMELINESS,
                score=0.0,
                details={"error": f"Timestamp parsing error: {str(e)}"}
            )
    
    def _determine_pass_status(self, metrics: List[QualityMetrics]) -> bool:
        """品質合格判定"""
        for metric in metrics:
            if metric.dimension == QualityDimension.COMPLETENESS:
                if metric.score < self.thresholds.completeness_min:
                    return False
            elif metric.dimension == QualityDimension.ACCURACY:
                if metric.score < self.thresholds.accuracy_min:
                    return False
            elif metric.dimension == QualityDimension.CONSISTENCY:
                if metric.score < self.thresholds.consistency_min:
                    return False
            elif metric.dimension == QualityDimension.VALIDITY:
                if metric.score < self.thresholds.validity_min:
                    return False
            elif metric.dimension == QualityDimension.UNIQUENESS:
                if metric.score < self.thresholds.uniqueness_min:
                    return False
        return True
    
    def _generate_recommendations(self, metrics: List[QualityMetrics]) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []
        
        for metric in metrics:
            if metric.dimension == QualityDimension.COMPLETENESS and metric.score < self.thresholds.completeness_min:
                worst_fields = metric.details.get("worst_fields", [])[:3]
                recommendations.append(f"完全性向上: {', '.join([f[0] for f in worst_fields])}フィールドの欠損値補完")
            
            elif metric.dimension == QualityDimension.ACCURACY and metric.score < self.thresholds.accuracy_min:
                recommendations.append("正確性向上: 外れ値の検証と修正、データ入力プロセスの見直し")
            
            elif metric.dimension == QualityDimension.CONSISTENCY and metric.score < self.thresholds.consistency_min:
                recommendations.append("一貫性向上: データ型の統一、カテゴリ値の標準化")
            
            elif metric.dimension == QualityDimension.VALIDITY and metric.score < self.thresholds.validity_min:
                recommendations.append("妥当性向上: 入力値検証の強化、ビジネスルールの適用")
            
            elif metric.dimension == QualityDimension.UNIQUENESS and metric.score < self.thresholds.uniqueness_min:
                recommendations.append("一意性向上: 重複データの除去、プライマリキー制約の強化")
            
            elif metric.dimension == QualityDimension.TIMELINESS and metric.score < 0.7:
                recommendations.append("適時性向上: データ更新頻度の増加、リアルタイム取り込みの検討")
        
        return recommendations
    
    async def assess_video_data_quality(self, data: pd.DataFrame) -> QualityReport:
        """動画データ特化品質評価"""
        return await self.assess_quality(
            data=data,
            dataset_name="video_data",
            key_fields=['external_id'],
            timestamp_field='created_at'
        )
    
    async def assess_user_data_quality(self, data: pd.DataFrame) -> QualityReport:
        """ユーザーデータ特化品質評価"""
        return await self.assess_quality(
            data=data,
            dataset_name="user_data", 
            key_fields=['user_id'],
            timestamp_field='created_at'
        )