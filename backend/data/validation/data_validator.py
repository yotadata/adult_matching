"""
Comprehensive data validation system for the adult matching platform.
Provides schema validation, quality checks, and data integrity verification.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Individual validation rule definition"""
    name: str
    description: str
    field: Optional[str] = None
    rule_type: str = "custom"  # schema, range, format, custom
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info
    enabled: bool = True


@dataclass
class ValidationResult:
    """Validation result for a single rule or dataset"""
    rule_name: str
    passed: bool
    message: str
    severity: str
    field: Optional[str] = None
    affected_count: int = 0
    sample_violations: List[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    dataset_name: str
    validation_level: ValidationLevel
    total_records: int
    passed_validations: int
    failed_validations: int
    warnings: int
    errors: int
    validation_results: List[ValidationResult] = field(default_factory=list)
    quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate quality score based on validation results"""
        if self.total_records == 0:
            self.quality_score = 0.0
            return
            
        error_weight = 1.0
        warning_weight = 0.5
        
        total_issues = (self.errors * error_weight) + (self.warnings * warning_weight)
        max_possible_issues = self.total_records * error_weight
        
        if max_possible_issues > 0:
            self.quality_score = max(0.0, 1.0 - (total_issues / max_possible_issues))
        else:
            self.quality_score = 1.0


class DataValidator:
    """
    Comprehensive data validation system supporting multiple validation types
    and configurable validation rules for different data domains.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.schema_definitions: Dict[str, Dict] = {}
        
        # Load default validation rules
        self._load_default_rules()
        
        # Load custom configuration if provided
        if config_path and config_path.exists():
            self._load_custom_config()
    
    def _load_default_rules(self):
        """Load default validation rules for common data types"""
        
        # Video data validation rules
        video_rules = [
            ValidationRule(
                name="required_fields",
                description="Check for required video metadata fields",
                rule_type="schema",
                parameters={"required_fields": ["title", "content_id", "source", "thumbnails"]},
                severity="error"
            ),
            ValidationRule(
                name="content_id_format",
                description="Validate content ID format",
                field="content_id",
                rule_type="format",
                parameters={"pattern": r"^[A-Z0-9\-_]{3,20}$"},
                severity="error"
            ),
            ValidationRule(
                name="title_length",
                description="Validate title length",
                field="title",
                rule_type="range",
                parameters={"min_length": 1, "max_length": 500},
                severity="warning"
            ),
            ValidationRule(
                name="thumbnail_urls",
                description="Validate thumbnail URL format",
                field="thumbnails",
                rule_type="custom",
                parameters={"validation_function": "_validate_thumbnail_urls"},
                severity="warning"
            )
        ]
        
        # User interaction validation rules
        interaction_rules = [
            ValidationRule(
                name="user_id_exists",
                description="Check user ID exists and is valid",
                field="user_id",
                rule_type="custom",
                parameters={"validation_function": "_validate_user_id"},
                severity="error"
            ),
            ValidationRule(
                name="interaction_type",
                description="Validate interaction type values",
                field="interaction_type",
                rule_type="format",
                parameters={"allowed_values": ["like", "skip", "view", "share"]},
                severity="error"
            ),
            ValidationRule(
                name="timestamp_validity",
                description="Validate interaction timestamps",
                field="timestamp",
                rule_type="custom",
                parameters={"validation_function": "_validate_timestamp"},
                severity="error"
            )
        ]
        
        # Review data validation rules
        review_rules = [
            ValidationRule(
                name="rating_range",
                description="Validate rating values",
                field="rating",
                rule_type="range",
                parameters={"min_value": 1, "max_value": 5},
                severity="error"
            ),
            ValidationRule(
                name="review_text_quality",
                description="Check review text quality and length",
                field="review_text",
                rule_type="custom",
                parameters={"validation_function": "_validate_review_text"},
                severity="warning"
            )
        ]
        
        # Embedding validation rules
        embedding_rules = [
            ValidationRule(
                name="embedding_dimensions",
                description="Validate embedding vector dimensions",
                field="embedding",
                rule_type="custom",
                parameters={"validation_function": "_validate_embedding_dimensions"},
                severity="error"
            ),
            ValidationRule(
                name="embedding_values",
                description="Check embedding value ranges and normalization",
                field="embedding",
                rule_type="custom",
                parameters={"validation_function": "_validate_embedding_values"},
                severity="warning"
            )
        ]
        
        self.validation_rules.update({
            "videos": video_rules,
            "interactions": interaction_rules,
            "reviews": review_rules,
            "embeddings": embedding_rules
        })
    
    async def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        data_type: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        custom_rules: Optional[List[ValidationRule]] = None
    ) -> DataQualityReport:
        """
        Validate a pandas DataFrame against specified rules
        
        Args:
            df: DataFrame to validate
            data_type: Type of data (videos, interactions, reviews, etc.)
            validation_level: Validation strictness level
            custom_rules: Additional custom validation rules
            
        Returns:
            DataQualityReport with validation results
        """
        if df.empty:
            return DataQualityReport(
                dataset_name=data_type,
                validation_level=validation_level,
                total_records=0,
                passed_validations=0,
                failed_validations=0,
                warnings=0,
                errors=0,
                quality_score=0.0,
                recommendations=["Dataset is empty - no data to validate"]
            )
        
        # Get applicable validation rules
        rules = self._get_applicable_rules(data_type, validation_level, custom_rules)
        
        validation_results = []
        passed_count = 0
        failed_count = 0
        warning_count = 0
        error_count = 0
        
        # Execute each validation rule
        for rule in rules:
            if not rule.enabled:
                continue
                
            try:
                result = await self._execute_validation_rule(df, rule)
                validation_results.append(result)
                
                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    if result.severity == "error":
                        error_count += 1
                    elif result.severity == "warning":
                        warning_count += 1
                        
            except Exception as e:
                logger.error(f"Validation rule '{rule.name}' failed: {e}")
                error_result = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    message=f"Validation rule execution failed: {e}",
                    severity="error",
                    field=rule.field
                )
                validation_results.append(error_result)
                failed_count += 1
                error_count += 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, df)
        
        return DataQualityReport(
            dataset_name=data_type,
            validation_level=validation_level,
            total_records=len(df),
            passed_validations=passed_count,
            failed_validations=failed_count,
            warnings=warning_count,
            errors=error_count,
            validation_results=validation_results,
            recommendations=recommendations
        )
    
    def _get_applicable_rules(
        self, 
        data_type: str, 
        validation_level: ValidationLevel,
        custom_rules: Optional[List[ValidationRule]] = None
    ) -> List[ValidationRule]:
        """Get validation rules applicable for the given data type and level"""
        
        rules = self.validation_rules.get(data_type, []).copy()
        
        # Add custom rules if provided
        if custom_rules:
            rules.extend(custom_rules)
        
        # Filter rules based on validation level
        if validation_level == ValidationLevel.BASIC:
            rules = [r for r in rules if r.severity == "error"]
        elif validation_level == ValidationLevel.STANDARD:
            rules = [r for r in rules if r.severity in ["error", "warning"]]
        # STRICT and CUSTOM include all rules
        
        return rules
    
    async def _execute_validation_rule(
        self, 
        df: pd.DataFrame, 
        rule: ValidationRule
    ) -> ValidationResult:
        """Execute a single validation rule against the DataFrame"""
        
        if rule.rule_type == "schema":
            return self._validate_schema(df, rule)
        elif rule.rule_type == "range":
            return self._validate_range(df, rule)
        elif rule.rule_type == "format":
            return self._validate_format(df, rule)
        elif rule.rule_type == "custom":
            return await self._validate_custom(df, rule)
        else:
            raise ValueError(f"Unknown validation rule type: {rule.rule_type}")
    
    def _validate_schema(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate DataFrame schema requirements"""
        required_fields = rule.parameters.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Missing required fields: {missing_fields}",
                severity=rule.severity,
                field=rule.field,
                affected_count=len(missing_fields),
                sample_violations=missing_fields[:5]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            message="All required fields present",
            severity=rule.severity,
            field=rule.field
        )
    
    def _validate_range(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate field value ranges"""
        if rule.field not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Field '{rule.field}' not found in dataset",
                severity="error",
                field=rule.field
            )
        
        series = df[rule.field].dropna()
        violations = []
        
        # Check string length ranges
        if "min_length" in rule.parameters or "max_length" in rule.parameters:
            min_len = rule.parameters.get("min_length", 0)
            max_len = rule.parameters.get("max_length", float('inf'))
            
            string_lengths = series.astype(str).str.len()
            invalid_mask = (string_lengths < min_len) | (string_lengths > max_len)
            violations = series[invalid_mask].tolist()
        
        # Check numeric value ranges
        elif "min_value" in rule.parameters or "max_value" in rule.parameters:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                min_val = rule.parameters.get("min_value", float('-inf'))
                max_val = rule.parameters.get("max_value", float('inf'))
                
                invalid_mask = (numeric_series < min_val) | (numeric_series > max_val)
                violations = series[invalid_mask].tolist()
            except Exception:
                violations = ["Unable to convert to numeric values"]
        
        if violations:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Found {len(violations)} values outside acceptable range",
                severity=rule.severity,
                field=rule.field,
                affected_count=len(violations),
                sample_violations=violations[:5]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            message="All values within acceptable range",
            severity=rule.severity,
            field=rule.field
        )
    
    def _validate_format(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate field format requirements"""
        if rule.field not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Field '{rule.field}' not found in dataset",
                severity="error",
                field=rule.field
            )
        
        series = df[rule.field].dropna()
        violations = []
        
        # Pattern validation
        if "pattern" in rule.parameters:
            pattern = rule.parameters["pattern"]
            try:
                regex = re.compile(pattern)
                invalid_mask = ~series.astype(str).str.match(regex)
                violations = series[invalid_mask].tolist()
            except Exception as e:
                return ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    message=f"Invalid regex pattern: {e}",
                    severity="error",
                    field=rule.field
                )
        
        # Allowed values validation
        elif "allowed_values" in rule.parameters:
            allowed_values = set(rule.parameters["allowed_values"])
            invalid_mask = ~series.isin(allowed_values)
            violations = series[invalid_mask].tolist()
        
        if violations:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Found {len(violations)} values with invalid format",
                severity=rule.severity,
                field=rule.field,
                affected_count=len(violations),
                sample_violations=violations[:5]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            message="All values have valid format",
            severity=rule.severity,
            field=rule.field
        )
    
    async def _validate_custom(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Execute custom validation function"""
        function_name = rule.parameters.get("validation_function")
        if not function_name:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message="No validation function specified",
                severity="error",
                field=rule.field
            )
        
        # Get the validation function
        validation_func = getattr(self, function_name, None)
        if not validation_func:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Validation function '{function_name}' not found",
                severity="error",
                field=rule.field
            )
        
        try:
            return await validation_func(df, rule)
        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Custom validation failed: {e}",
                severity="error",
                field=rule.field
            )
    
    # Custom validation functions
    
    async def _validate_thumbnail_urls(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate thumbnail URL format and accessibility"""
        if rule.field not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Field '{rule.field}' not found",
                severity="error",
                field=rule.field
            )
        
        violations = []
        url_pattern = re.compile(r'https?://[^\s]+\.(jpg|jpeg|png|gif|webp)$', re.IGNORECASE)
        
        for idx, thumbnails in df[rule.field].dropna().items():
            if isinstance(thumbnails, str):
                try:
                    thumbnail_list = json.loads(thumbnails) if thumbnails.startswith('[') else [thumbnails]
                except:
                    thumbnail_list = [thumbnails]
            elif isinstance(thumbnails, list):
                thumbnail_list = thumbnails
            else:
                violations.append(f"Row {idx}: Invalid thumbnail format")
                continue
            
            for url in thumbnail_list:
                if not url_pattern.match(str(url)):
                    violations.append(f"Row {idx}: Invalid URL format - {url}")
                    break
        
        if violations:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Found {len(violations)} invalid thumbnail URLs",
                severity=rule.severity,
                field=rule.field,
                affected_count=len(violations),
                sample_violations=violations[:5]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            message="All thumbnail URLs are valid",
            severity=rule.severity,
            field=rule.field
        )
    
    async def _validate_user_id(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate user ID format and existence"""
        if rule.field not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Field '{rule.field}' not found",
                severity="error",
                field=rule.field
            )
        
        violations = []
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        for idx, user_id in df[rule.field].dropna().items():
            if not uuid_pattern.match(str(user_id)):
                violations.append(f"Row {idx}: Invalid UUID format - {user_id}")
        
        if violations:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Found {len(violations)} invalid user IDs",
                severity=rule.severity,
                field=rule.field,
                affected_count=len(violations),
                sample_violations=violations[:5]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            message="All user IDs are valid",
            severity=rule.severity,
            field=rule.field
        )
    
    async def _validate_timestamp(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate timestamp format and reasonable ranges"""
        if rule.field not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Field '{rule.field}' not found",
                severity="error",
                field=rule.field
            )
        
        violations = []
        now = datetime.now()
        min_date = datetime(2020, 1, 1)  # Reasonable minimum date
        
        for idx, timestamp in df[rule.field].dropna().items():
            try:
                if isinstance(timestamp, str):
                    parsed_date = pd.to_datetime(timestamp)
                elif isinstance(timestamp, (int, float)):
                    parsed_date = pd.to_datetime(timestamp, unit='s')
                else:
                    parsed_date = pd.to_datetime(timestamp)
                
                if parsed_date < min_date or parsed_date > now:
                    violations.append(f"Row {idx}: Timestamp out of range - {timestamp}")
                    
            except Exception:
                violations.append(f"Row {idx}: Invalid timestamp format - {timestamp}")
        
        if violations:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                message=f"Found {len(violations)} invalid timestamps",
                severity=rule.severity,
                field=rule.field,
                affected_count=len(violations),
                sample_violations=violations[:5]
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            message="All timestamps are valid",
            severity=rule.severity,
            field=rule.field
        )
    
    def _generate_recommendations(
        self, 
        validation_results: List[ValidationResult], 
        df: pd.DataFrame
    ) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        # Count issues by type
        error_count = sum(1 for r in validation_results if not r.passed and r.severity == "error")
        warning_count = sum(1 for r in validation_results if not r.passed and r.severity == "warning")
        
        if error_count > 0:
            recommendations.append(f"Address {error_count} critical errors before using this data")
        
        if warning_count > 0:
            recommendations.append(f"Consider fixing {warning_count} warnings to improve data quality")
        
        # Specific recommendations based on failed validations
        failed_rules = [r for r in validation_results if not r.passed]
        
        for result in failed_rules:
            if "missing" in result.message.lower():
                recommendations.append(f"Add missing data for field '{result.field}'")
            elif "format" in result.message.lower():
                recommendations.append(f"Standardize format for field '{result.field}'")
            elif "range" in result.message.lower():
                recommendations.append(f"Review value ranges for field '{result.field}'")
        
        return recommendations
    
    def export_validation_report(
        self, 
        report: DataQualityReport, 
        output_path: Path,
        format: str = "json"
    ) -> None:
        """Export validation report to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            report_dict = {
                "dataset_name": report.dataset_name,
                "validation_level": report.validation_level.value,
                "summary": {
                    "total_records": report.total_records,
                    "quality_score": report.quality_score,
                    "passed_validations": report.passed_validations,
                    "failed_validations": report.failed_validations,
                    "warnings": report.warnings,
                    "errors": report.errors
                },
                "validation_results": [
                    {
                        "rule_name": r.rule_name,
                        "passed": r.passed,
                        "message": r.message,
                        "severity": r.severity,
                        "field": r.field,
                        "affected_count": r.affected_count,
                        "sample_violations": r.sample_violations
                    }
                    for r in report.validation_results
                ],
                "recommendations": report.recommendations,
                "timestamp": report.timestamp.isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Validation report exported to {output_path}")