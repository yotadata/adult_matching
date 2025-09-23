"""
Data Quality Monitor

データ品質監視システム
データの完全性、一貫性、正確性を継続的に監視・評価
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from backend.utils.logger import get_logger
from backend.utils.database import get_supabase_client

logger = get_logger(__name__)

@dataclass
class QualityRule:
    """データ品質ルール"""
    name: str
    description: str
    rule_type: str  # "completeness", "consistency", "accuracy", "validity", "uniqueness"
    column: Optional[str] = None
    condition: Optional[str] = None
    threshold: Optional[float] = None
    severity: str = "warning"  # "info", "warning", "error", "critical"
    active: bool = True

@dataclass 
class QualityMetric:
    """品質メトリクス"""
    metric_name: str
    value: float
    threshold: Optional[float] = None
    status: str = "unknown"  # "pass", "fail", "warning"
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityReport:
    """品質レポート"""
    table_name: str
    check_timestamp: datetime
    overall_score: float = 0.0
    total_records: int = 0
    metrics: List[QualityMetric] = field(default_factory=list)
    rule_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "table_name": self.table_name,
            "check_timestamp": self.check_timestamp.isoformat(),
            "overall_score": self.overall_score,
            "total_records": self.total_records,
            "metrics": [
                {
                    "name": m.metric_name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status,
                    "details": m.details
                }
                for m in self.metrics
            ],
            "rule_results": self.rule_results,
            "recommendations": self.recommendations
        }


class DataQualityMonitor:
    """データ品質監視システム"""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.supabase = get_supabase_client()
        self.storage_dir = storage_dir or Path("backend/data/storage/quality")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # デフォルト品質ルール
        self.default_rules = self._get_default_rules()
        
    def _get_default_rules(self) -> List[QualityRule]:
        """デフォルト品質ルール定義"""
        return [
            # 完全性ルール
            QualityRule(
                name="title_completeness",
                description="Title field completeness",
                rule_type="completeness",
                column="title",
                threshold=0.95,
                severity="error"
            ),
            QualityRule(
                name="external_id_completeness", 
                description="External ID completeness",
                rule_type="completeness",
                column="external_id",
                threshold=1.0,
                severity="critical"
            ),
            
            # ユニーク性ルール
            QualityRule(
                name="external_id_uniqueness",
                description="External ID uniqueness within source",
                rule_type="uniqueness",
                column="external_id",
                threshold=1.0,
                severity="critical"
            ),
            
            # 妥当性ルール
            QualityRule(
                name="price_validity",
                description="Price must be non-negative",
                rule_type="validity",
                column="price",
                condition="price >= 0",
                threshold=0.98,
                severity="warning"
            ),
            QualityRule(
                name="url_validity",
                description="Thumbnail URL format validation",
                rule_type="validity", 
                column="thumbnail_url",
                condition="thumbnail_url LIKE 'http%'",
                threshold=0.90,
                severity="warning"
            ),
            
            # 一貫性ルール
            QualityRule(
                name="source_consistency",
                description="Source field consistency",
                rule_type="consistency",
                column="source",
                condition="source IN ('dmm', 'fanza', 'manual')",
                threshold=1.0,
                severity="error"
            )
        ]
    
    async def check_table_quality(
        self,
        table_name: str,
        custom_rules: Optional[List[QualityRule]] = None,
        sample_size: Optional[int] = None
    ) -> QualityReport:
        """
        テーブル品質チェック実行
        
        Args:
            table_name: チェック対象テーブル名
            custom_rules: カスタム品質ルール
            sample_size: サンプルサイズ（Noneで全件）
            
        Returns:
            品質レポート
        """
        logger.info(f"🔍 Starting quality check for table: {table_name}")
        
        # データ取得
        try:
            query = self.supabase.table(table_name).select("*")
            if sample_size:
                query = query.limit(sample_size)
            
            response = query.execute()
            data = response.data
            
            if not data:
                logger.warning(f"No data found in table {table_name}")
                return QualityReport(
                    table_name=table_name,
                    check_timestamp=datetime.utcnow(),
                    recommendations=["Table is empty - no quality checks possible"]
                )
                
        except Exception as e:
            logger.error(f"Failed to fetch data from {table_name}: {e}")
            return QualityReport(
                table_name=table_name,
                check_timestamp=datetime.utcnow(),
                recommendations=[f"Failed to access table: {str(e)}"]
            )
        
        # データフレーム変換
        df = pd.DataFrame(data)
        logger.info(f"📊 Analyzing {len(df)} records from {table_name}")
        
        # 品質チェック実行
        report = QualityReport(
            table_name=table_name,
            check_timestamp=datetime.utcnow(),
            total_records=len(df)
        )
        
        # 使用するルール決定
        rules_to_apply = custom_rules or self.default_rules
        active_rules = [r for r in rules_to_apply if r.active]
        
        # ルール別チェック実行
        for rule in active_rules:
            try:
                metric = await self._apply_quality_rule(df, rule)
                if metric:
                    report.metrics.append(metric)
                    
            except Exception as e:
                logger.error(f"Failed to apply rule {rule.name}: {e}")
                report.rule_results[rule.name] = {"error": str(e)}
        
        # 基本統計メトリクス
        basic_metrics = self._calculate_basic_metrics(df)
        report.metrics.extend(basic_metrics)
        
        # 全体スコア計算
        report.overall_score = self._calculate_overall_score(report.metrics)
        
        # 推奨事項生成
        report.recommendations = self._generate_recommendations(df, report.metrics)
        
        # レポート保存
        await self._save_quality_report(report)
        
        logger.info(f"✅ Quality check completed. Overall score: {report.overall_score:.2f}")
        return report
    
    async def _apply_quality_rule(self, df: pd.DataFrame, rule: QualityRule) -> Optional[QualityMetric]:
        """品質ルール適用"""
        try:
            if rule.rule_type == "completeness":
                return self._check_completeness(df, rule)
            elif rule.rule_type == "uniqueness":
                return self._check_uniqueness(df, rule)
            elif rule.rule_type == "validity":
                return self._check_validity(df, rule)
            elif rule.rule_type == "consistency":
                return self._check_consistency(df, rule)
            elif rule.rule_type == "accuracy":
                return self._check_accuracy(df, rule)
            else:
                logger.warning(f"Unknown rule type: {rule.rule_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying rule {rule.name}: {e}")
            return None
    
    def _check_completeness(self, df: pd.DataFrame, rule: QualityRule) -> QualityMetric:
        """完全性チェック"""
        if rule.column not in df.columns:
            return QualityMetric(
                metric_name=f"{rule.name}_completeness",
                value=0.0,
                threshold=rule.threshold,
                status="fail",
                details={"error": f"Column {rule.column} not found"}
            )
        
        non_null_count = df[rule.column].notna().sum()
        completeness_ratio = non_null_count / len(df) if len(df) > 0 else 0.0
        
        status = "pass" if completeness_ratio >= (rule.threshold or 1.0) else "fail"
        
        return QualityMetric(
            metric_name=f"{rule.name}_completeness",
            value=completeness_ratio,
            threshold=rule.threshold,
            status=status,
            details={
                "non_null_count": int(non_null_count),
                "total_count": len(df),
                "null_count": len(df) - int(non_null_count)
            }
        )
    
    def _check_uniqueness(self, df: pd.DataFrame, rule: QualityRule) -> QualityMetric:
        """一意性チェック"""
        if rule.column not in df.columns:
            return QualityMetric(
                metric_name=f"{rule.name}_uniqueness",
                value=0.0,
                threshold=rule.threshold,
                status="fail",
                details={"error": f"Column {rule.column} not found"}
            )
        
        # グループごとの一意性をチェック（sourceカラムがある場合）
        if "source" in df.columns:
            unique_counts = []
            for source in df["source"].unique():
                source_df = df[df["source"] == source]
                if len(source_df) > 0:
                    unique_ratio = source_df[rule.column].nunique() / len(source_df)
                    unique_counts.append(unique_ratio)
            
            avg_uniqueness = np.mean(unique_counts) if unique_counts else 0.0
        else:
            unique_count = df[rule.column].nunique()
            avg_uniqueness = unique_count / len(df) if len(df) > 0 else 0.0
        
        status = "pass" if avg_uniqueness >= (rule.threshold or 1.0) else "fail"
        
        return QualityMetric(
            metric_name=f"{rule.name}_uniqueness",
            value=avg_uniqueness,
            threshold=rule.threshold,
            status=status,
            details={
                "unique_values": int(df[rule.column].nunique()),
                "total_records": len(df),
                "duplicate_count": len(df) - int(df[rule.column].nunique())
            }
        )
    
    def _check_validity(self, df: pd.DataFrame, rule: QualityRule) -> QualityMetric:
        """妥当性チェック"""
        if rule.column not in df.columns:
            return QualityMetric(
                metric_name=f"{rule.name}_validity",
                value=0.0,
                threshold=rule.threshold,
                status="fail",
                details={"error": f"Column {rule.column} not found"}
            )
        
        try:
            # 条件チェック（簡易版）
            if rule.condition:
                if "price >= 0" in rule.condition:
                    valid_count = (df[rule.column] >= 0).sum()
                elif "LIKE 'http%" in rule.condition:
                    valid_count = df[rule.column].astype(str).str.startswith('http').sum()
                else:
                    # より複雑な条件は別途実装が必要
                    valid_count = len(df)
            else:
                valid_count = len(df)
            
            validity_ratio = valid_count / len(df) if len(df) > 0 else 0.0
            status = "pass" if validity_ratio >= (rule.threshold or 1.0) else "fail"
            
            return QualityMetric(
                metric_name=f"{rule.name}_validity",
                value=validity_ratio,
                threshold=rule.threshold,
                status=status,
                details={
                    "valid_count": int(valid_count),
                    "invalid_count": len(df) - int(valid_count),
                    "condition": rule.condition
                }
            )
        except Exception as e:
            return QualityMetric(
                metric_name=f"{rule.name}_validity",
                value=0.0,
                threshold=rule.threshold,
                status="fail",
                details={"error": f"Validation error: {str(e)}"}
            )
    
    def _check_consistency(self, df: pd.DataFrame, rule: QualityRule) -> QualityMetric:
        """一貫性チェック"""
        if rule.column not in df.columns:
            return QualityMetric(
                metric_name=f"{rule.name}_consistency",
                value=0.0,
                threshold=rule.threshold,
                status="fail",
                details={"error": f"Column {rule.column} not found"}
            )
        
        try:
            # 許可された値のチェック
            if "IN (" in rule.condition:
                # 例: "source IN ('dmm', 'fanza', 'manual')"
                import re
                values_match = re.search(r"IN \(([^)]+)\)", rule.condition)
                if values_match:
                    allowed_values = [v.strip().strip("'\"") for v in values_match.group(1).split(",")]
                    consistent_count = df[rule.column].isin(allowed_values).sum()
                else:
                    consistent_count = len(df)
            else:
                consistent_count = len(df)
            
            consistency_ratio = consistent_count / len(df) if len(df) > 0 else 0.0
            status = "pass" if consistency_ratio >= (rule.threshold or 1.0) else "fail"
            
            return QualityMetric(
                metric_name=f"{rule.name}_consistency",
                value=consistency_ratio,
                threshold=rule.threshold,
                status=status,
                details={
                    "consistent_count": int(consistent_count),
                    "inconsistent_count": len(df) - int(consistent_count),
                    "condition": rule.condition
                }
            )
        except Exception as e:
            return QualityMetric(
                metric_name=f"{rule.name}_consistency",
                value=0.0,
                threshold=rule.threshold,
                status="fail",
                details={"error": f"Consistency check error: {str(e)}"}
            )
    
    def _check_accuracy(self, df: pd.DataFrame, rule: QualityRule) -> QualityMetric:
        """正確性チェック（基本実装）"""
        # 正確性は複雑で、外部データソースとの比較が必要
        # ここでは基本的なフォーマット・範囲チェックのみ
        return QualityMetric(
            metric_name=f"{rule.name}_accuracy",
            value=1.0,  # 暫定
            threshold=rule.threshold,
            status="pass",
            details={"note": "Accuracy check not fully implemented"}
        )
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> List[QualityMetric]:
        """基本統計メトリクス計算"""
        metrics = []
        
        # データ量メトリクス
        metrics.append(QualityMetric(
            metric_name="record_count",
            value=float(len(df)),
            status="info",
            details={"total_records": len(df)}
        ))
        
        # カラム数
        metrics.append(QualityMetric(
            metric_name="column_count", 
            value=float(len(df.columns)),
            status="info",
            details={"total_columns": len(df.columns)}
        ))
        
        # 全体的な完全性
        if len(df) > 0:
            total_cells = len(df) * len(df.columns)
            non_null_cells = df.notna().sum().sum()
            overall_completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
            
            metrics.append(QualityMetric(
                metric_name="overall_completeness",
                value=overall_completeness,
                status="pass" if overall_completeness >= 0.8 else "warning",
                details={
                    "non_null_cells": int(non_null_cells),
                    "total_cells": total_cells
                }
            ))
        
        return metrics
    
    def _calculate_overall_score(self, metrics: List[QualityMetric]) -> float:
        """全体品質スコア計算"""
        if not metrics:
            return 0.0
        
        # 重み付けスコア計算
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            if metric.metric_name.endswith('_completeness'):
                weight = 3.0  # 完全性は重要
            elif metric.metric_name.endswith('_uniqueness'):
                weight = 3.0  # 一意性は重要
            elif metric.metric_name.endswith('_consistency'):
                weight = 2.0
            elif metric.metric_name.endswith('_validity'):
                weight = 2.0
            else:
                weight = 1.0
            
            # statusベースのスコア
            if metric.status == "pass":
                status_score = 1.0
            elif metric.status == "warning":
                status_score = 0.7
            elif metric.status == "fail":
                status_score = 0.3
            else:
                status_score = 0.5
            
            total_weighted_score += status_score * weight
            total_weight += weight
        
        return (total_weighted_score / total_weight) * 100 if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, df: pd.DataFrame, metrics: List[QualityMetric]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        for metric in metrics:
            if metric.status == "fail":
                if "completeness" in metric.metric_name:
                    recommendations.append(f"Improve {metric.metric_name}: Add missing data validation")
                elif "uniqueness" in metric.metric_name:
                    recommendations.append(f"Fix {metric.metric_name}: Remove or consolidate duplicate records")
                elif "validity" in metric.metric_name:
                    recommendations.append(f"Fix {metric.metric_name}: Add data validation rules")
                elif "consistency" in metric.metric_name:
                    recommendations.append(f"Fix {metric.metric_name}: Standardize data formats")
        
        # 一般的な推奨事項
        if len(df) == 0:
            recommendations.append("Table is empty - consider data ingestion")
        elif len(df) < 100:
            recommendations.append("Small dataset - monitor data collection processes")
        
        return recommendations
    
    async def _save_quality_report(self, report: QualityReport):
        """品質レポート保存"""
        try:
            # ファイル保存
            report_file = self.storage_dir / f"{report.table_name}_{report.check_timestamp.strftime('%Y%m%d_%H%M%S')}_quality_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Quality report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")
    
    async def get_quality_history(self, table_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """品質履歴取得"""
        history = []
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            for report_file in self.storage_dir.glob(f"{table_name}_*_quality_report.json"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    check_time = datetime.fromisoformat(report_data['check_timestamp'])
                    if check_time >= cutoff_date:
                        history.append(report_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to load report {report_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to read quality history: {e}")
        
        # 時系列順にソート
        return sorted(history, key=lambda x: x['check_timestamp'], reverse=True)


# 便利関数
async def run_quality_check(table_name: str, **kwargs) -> QualityReport:
    """品質チェック実行（簡単インターフェース）"""
    monitor = DataQualityMonitor()
    return await monitor.check_table_quality(table_name, **kwargs)