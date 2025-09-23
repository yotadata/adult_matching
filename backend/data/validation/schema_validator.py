"""
Schema Validator

スキーマ検証システム - データ構造、型定義、制約の検証
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from pathlib import Path
import json


class FieldType(Enum):
    """フィールド型定義"""
    STRING = "string"
    INTEGER = "integer"  
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"


class ConstraintType(Enum):
    """制約タイプ"""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    REGEX = "regex"
    ENUM_VALUES = "enum_values"


@dataclass
class FieldConstraint:
    """フィールド制約定義"""
    constraint_type: ConstraintType
    value: Any
    error_message: Optional[str] = None


@dataclass
class FieldDefinition:
    """フィールド定義"""
    name: str
    field_type: FieldType
    required: bool = False
    constraints: List[FieldConstraint] = field(default_factory=list)
    description: Optional[str] = None
    default_value: Optional[Any] = None


@dataclass
class SchemaDefinition:
    """スキーマ定義"""
    name: str
    version: str
    fields: List[FieldDefinition]
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def field_names(self) -> List[str]:
        """フィールド名リスト"""
        return [field.name for field in self.fields]
    
    @property
    def required_fields(self) -> List[str]:
        """必須フィールドリスト"""
        return [field.name for field in self.fields if field.required]


@dataclass
class FieldValidation:
    """フィールド検証結果"""
    field_name: str
    field_type: FieldType
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None


@dataclass
class SchemaValidationResult:
    """スキーマ検証結果"""
    schema_name: str
    data_source: str
    validation_timestamp: datetime
    total_records: int
    passed: bool
    field_validations: List[FieldValidation]
    structural_errors: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def failed_fields(self) -> List[str]:
        """失敗したフィールド"""
        return [fv.field_name for fv in self.field_validations if not fv.passed]
    
    @property
    def error_count(self) -> int:
        """エラー総数"""
        return len(self.structural_errors) + sum(len(fv.errors) for fv in self.field_validations)


@dataclass
class SchemaConflict:
    """スキーマ競合"""
    field_name: str
    expected: Any
    actual: Any
    conflict_type: str
    severity: str = "error"


class SchemaValidator:
    """スキーマ検証システム"""
    
    def __init__(self, schema: Optional[SchemaDefinition] = None):
        self.schema = schema
        self.logger = logging.getLogger(__name__)
    
    def load_schema_from_file(self, schema_path: Union[Path, str]) -> SchemaDefinition:
        """ファイルからスキーマ読み込み"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        fields = []
        for field_data in schema_data['fields']:
            constraints = []
            if 'constraints' in field_data:
                for constraint_data in field_data['constraints']:
                    constraints.append(FieldConstraint(
                        constraint_type=ConstraintType(constraint_data['type']),
                        value=constraint_data['value'],
                        error_message=constraint_data.get('error_message')
                    ))
            
            fields.append(FieldDefinition(
                name=field_data['name'],
                field_type=FieldType(field_data['type']),
                required=field_data.get('required', False),
                constraints=constraints,
                description=field_data.get('description'),
                default_value=field_data.get('default')
            ))
        
        return SchemaDefinition(
            name=schema_data['name'],
            version=schema_data['version'],
            fields=fields,
            description=schema_data.get('description')
        )
    
    def save_schema_to_file(self, schema: SchemaDefinition, output_path: Union[Path, str]):
        """スキーマをファイルに保存"""
        schema_data = {
            "name": schema.name,
            "version": schema.version,
            "description": schema.description,
            "created_at": schema.created_at.isoformat(),
            "fields": []
        }
        
        for field in schema.fields:
            field_data = {
                "name": field.name,
                "type": field.field_type.value,
                "required": field.required,
                "description": field.description,
                "default": field.default_value
            }
            
            if field.constraints:
                field_data["constraints"] = []
                for constraint in field.constraints:
                    constraint_data = {
                        "type": constraint.constraint_type.value,
                        "value": constraint.value
                    }
                    if constraint.error_message:
                        constraint_data["error_message"] = constraint.error_message
                    field_data["constraints"].append(constraint_data)
            
            schema_data["fields"].append(field_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)
    
    async def validate_data(self, data: pd.DataFrame, data_source: str = "unknown") -> SchemaValidationResult:
        """データのスキーマ検証"""
        if not self.schema:
            raise ValueError("スキーマが設定されていません")
        
        structural_errors = []
        field_validations = []
        
        # 構造的検証
        missing_fields = set(self.schema.required_fields) - set(data.columns)
        if missing_fields:
            structural_errors.append(f"必須フィールド不足: {', '.join(missing_fields)}")
        
        extra_fields = set(data.columns) - set(self.schema.field_names)
        if extra_fields:
            structural_errors.append(f"予期しないフィールド: {', '.join(extra_fields)}")
        
        # フィールド別検証
        for field_def in self.schema.fields:
            if field_def.name in data.columns:
                field_validation = await self._validate_field(data[field_def.name], field_def)
                field_validations.append(field_validation)
        
        # 全体評価
        passed = len(structural_errors) == 0 and all(fv.passed for fv in field_validations)
        
        # 統計情報生成
        summary = self._generate_validation_summary(data, field_validations, structural_errors)
        
        return SchemaValidationResult(
            schema_name=self.schema.name,
            data_source=data_source,
            validation_timestamp=datetime.now(),
            total_records=len(data),
            passed=passed,
            field_validations=field_validations,
            structural_errors=structural_errors,
            summary=summary
        )
    
    async def _validate_field(self, series: pd.Series, field_def: FieldDefinition) -> FieldValidation:
        """個別フィールド検証"""
        errors = []
        warnings = []
        statistics = {}
        
        # 基本統計
        statistics['null_count'] = int(series.isnull().sum())
        statistics['null_ratio'] = statistics['null_count'] / len(series)
        statistics['unique_count'] = int(series.nunique())
        
        # 型検証
        if not self._validate_field_type(series, field_def.field_type):
            errors.append(f"型不一致: 期待={field_def.field_type.value}, 実際={str(series.dtype)}")
        
        # 必須チェック
        if field_def.required and statistics['null_count'] > 0:
            errors.append(f"必須フィールドにNULL値: {statistics['null_count']}件")
        
        # 制約検証
        for constraint in field_def.constraints:
            constraint_error = self._validate_constraint(series, constraint)
            if constraint_error:
                errors.append(constraint_error)
        
        # フィールド型別統計
        if field_def.field_type in [FieldType.INTEGER, FieldType.FLOAT]:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) > 0:
                statistics.update({
                    'min': float(numeric_data.min()),
                    'max': float(numeric_data.max()),
                    'mean': float(numeric_data.mean()),
                    'std': float(numeric_data.std())
                })
        
        elif field_def.field_type == FieldType.STRING:
            string_data = series.dropna().astype(str)
            if len(string_data) > 0:
                lengths = string_data.str.len()
                statistics.update({
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'avg_length': float(lengths.mean())
                })
        
        passed = len(errors) == 0
        
        return FieldValidation(
            field_name=field_def.name,
            field_type=field_def.field_type,
            passed=passed,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )
    
    def _validate_field_type(self, series: pd.Series, expected_type: FieldType) -> bool:
        """フィールド型検証"""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return True
        
        if expected_type == FieldType.STRING:
            return True  # pandasでは大部分がobject型になる
        
        elif expected_type == FieldType.INTEGER:
            try:
                pd.to_numeric(non_null_series, errors='raise', downcast='integer')
                return True
            except (ValueError, TypeError):
                return False
        
        elif expected_type == FieldType.FLOAT:
            return pd.api.types.is_numeric_dtype(non_null_series)
        
        elif expected_type == FieldType.BOOLEAN:
            unique_values = non_null_series.unique()
            bool_values = {True, False, 'true', 'false', 'True', 'False', 1, 0, '1', '0'}
            return all(val in bool_values for val in unique_values)
        
        elif expected_type == FieldType.DATETIME:
            try:
                pd.to_datetime(non_null_series, errors='raise')
                return True
            except (ValueError, TypeError):
                return False
        
        elif expected_type == FieldType.JSON:
            return True  # JSON形式の詳細検証は複雑なので簡略化
        
        elif expected_type == FieldType.ARRAY:
            return True  # 配列形式の詳細検証は複雑なので簡略化
        
        return False
    
    def _validate_constraint(self, series: pd.Series, constraint: FieldConstraint) -> Optional[str]:
        """制約検証"""
        non_null_series = series.dropna()
        
        if constraint.constraint_type == ConstraintType.NOT_NULL:
            null_count = series.isnull().sum()
            if null_count > 0:
                return constraint.error_message or f"NULL値制約違反: {null_count}件"
        
        elif constraint.constraint_type == ConstraintType.UNIQUE:
            duplicate_count = series.duplicated().sum()
            if duplicate_count > 0:
                return constraint.error_message or f"一意制約違反: {duplicate_count}件の重複"
        
        elif constraint.constraint_type == ConstraintType.MIN_LENGTH:
            if len(non_null_series) > 0:
                string_data = non_null_series.astype(str)
                short_values = (string_data.str.len() < constraint.value).sum()
                if short_values > 0:
                    return constraint.error_message or f"最小長制約違反: {short_values}件"
        
        elif constraint.constraint_type == ConstraintType.MAX_LENGTH:
            if len(non_null_series) > 0:
                string_data = non_null_series.astype(str)
                long_values = (string_data.str.len() > constraint.value).sum()
                if long_values > 0:
                    return constraint.error_message or f"最大長制約違反: {long_values}件"
        
        elif constraint.constraint_type == ConstraintType.MIN_VALUE:
            if len(non_null_series) > 0:
                try:
                    numeric_data = pd.to_numeric(non_null_series, errors='coerce').dropna()
                    small_values = (numeric_data < constraint.value).sum()
                    if small_values > 0:
                        return constraint.error_message or f"最小値制約違反: {small_values}件"
                except:
                    pass
        
        elif constraint.constraint_type == ConstraintType.MAX_VALUE:
            if len(non_null_series) > 0:
                try:
                    numeric_data = pd.to_numeric(non_null_series, errors='coerce').dropna()
                    large_values = (numeric_data > constraint.value).sum()
                    if large_values > 0:
                        return constraint.error_message or f"最大値制約違反: {large_values}件"
                except:
                    pass
        
        elif constraint.constraint_type == ConstraintType.ENUM_VALUES:
            if len(non_null_series) > 0:
                invalid_values = ~non_null_series.isin(constraint.value)
                invalid_count = invalid_values.sum()
                if invalid_count > 0:
                    return constraint.error_message or f"列挙値制約違反: {invalid_count}件"
        
        elif constraint.constraint_type == ConstraintType.REGEX:
            if len(non_null_series) > 0:
                string_data = non_null_series.astype(str)
                invalid_pattern = ~string_data.str.match(constraint.value, na=False)
                invalid_count = invalid_pattern.sum()
                if invalid_count > 0:
                    return constraint.error_message or f"正規表現制約違反: {invalid_count}件"
        
        return None
    
    def _generate_validation_summary(self, data: pd.DataFrame, field_validations: List[FieldValidation], 
                                   structural_errors: List[str]) -> Dict[str, Any]:
        """検証サマリー生成"""
        return {
            "total_fields": len(self.schema.fields),
            "validated_fields": len(field_validations),
            "passed_fields": len([fv for fv in field_validations if fv.passed]),
            "failed_fields": len([fv for fv in field_validations if not fv.passed]),
            "structural_errors": len(structural_errors),
            "data_shape": data.shape,
            "schema_version": self.schema.version,
            "field_statistics": {fv.field_name: fv.statistics for fv in field_validations}
        }
    
    def compare_schemas(self, other_schema: SchemaDefinition) -> List[SchemaConflict]:
        """スキーマ比較と競合検出"""
        if not self.schema:
            raise ValueError("比較元スキーマが設定されていません")
        
        conflicts = []
        
        # フィールド名比較
        self_fields = {field.name: field for field in self.schema.fields}
        other_fields = {field.name: field for field in other_schema.fields}
        
        # 不足フィールド
        missing_fields = set(self_fields.keys()) - set(other_fields.keys())
        for field_name in missing_fields:
            conflicts.append(SchemaConflict(
                field_name=field_name,
                expected="exists",
                actual="missing",
                conflict_type="missing_field"
            ))
        
        # 追加フィールド
        extra_fields = set(other_fields.keys()) - set(self_fields.keys())
        for field_name in extra_fields:
            conflicts.append(SchemaConflict(
                field_name=field_name,
                expected="not_exists",
                actual="exists",
                conflict_type="extra_field",
                severity="warning"
            ))
        
        # 共通フィールドの詳細比較
        common_fields = set(self_fields.keys()) & set(other_fields.keys())
        for field_name in common_fields:
            self_field = self_fields[field_name]
            other_field = other_fields[field_name]
            
            # 型比較
            if self_field.field_type != other_field.field_type:
                conflicts.append(SchemaConflict(
                    field_name=field_name,
                    expected=self_field.field_type.value,
                    actual=other_field.field_type.value,
                    conflict_type="type_mismatch"
                ))
            
            # 必須性比較
            if self_field.required != other_field.required:
                conflicts.append(SchemaConflict(
                    field_name=field_name,
                    expected=self_field.required,
                    actual=other_field.required,
                    conflict_type="required_mismatch",
                    severity="warning"
                ))
        
        return conflicts
    
    def create_video_schema(self) -> SchemaDefinition:
        """動画データスキーマ作成"""
        fields = [
            FieldDefinition(
                name="external_id",
                field_type=FieldType.STRING,
                required=True,
                constraints=[
                    FieldConstraint(ConstraintType.NOT_NULL, True),
                    FieldConstraint(ConstraintType.UNIQUE, True),
                    FieldConstraint(ConstraintType.MIN_LENGTH, 1)
                ]
            ),
            FieldDefinition(
                name="title",
                field_type=FieldType.STRING,
                required=True,
                constraints=[
                    FieldConstraint(ConstraintType.NOT_NULL, True),
                    FieldConstraint(ConstraintType.MIN_LENGTH, 1),
                    FieldConstraint(ConstraintType.MAX_LENGTH, 500)
                ]
            ),
            FieldDefinition(
                name="duration",
                field_type=FieldType.INTEGER,
                required=False,
                constraints=[
                    FieldConstraint(ConstraintType.MIN_VALUE, 0),
                    FieldConstraint(ConstraintType.MAX_VALUE, 14400)
                ]
            ),
            FieldDefinition(
                name="price",
                field_type=FieldType.INTEGER,
                required=False,
                constraints=[
                    FieldConstraint(ConstraintType.MIN_VALUE, 0)
                ]
            ),
            FieldDefinition(
                name="rating",
                field_type=FieldType.FLOAT,
                required=False,
                constraints=[
                    FieldConstraint(ConstraintType.MIN_VALUE, 0.0),
                    FieldConstraint(ConstraintType.MAX_VALUE, 5.0)
                ]
            ),
            FieldDefinition(
                name="source",
                field_type=FieldType.STRING,
                required=True,
                constraints=[
                    FieldConstraint(ConstraintType.ENUM_VALUES, ["dmm", "fanza", "manual"])
                ]
            ),
            FieldDefinition(
                name="created_at",
                field_type=FieldType.DATETIME,
                required=True,
                constraints=[
                    FieldConstraint(ConstraintType.NOT_NULL, True)
                ]
            )
        ]
        
        return SchemaDefinition(
            name="video_data_schema",
            version="1.0.0",
            fields=fields,
            description="動画データスキーマ定義"
        )
    
    def create_user_schema(self) -> SchemaDefinition:
        """ユーザーデータスキーマ作成"""
        fields = [
            FieldDefinition(
                name="user_id",
                field_type=FieldType.STRING,
                required=True,
                constraints=[
                    FieldConstraint(ConstraintType.NOT_NULL, True),
                    FieldConstraint(ConstraintType.UNIQUE, True)
                ]
            ),
            FieldDefinition(
                name="email",
                field_type=FieldType.STRING,
                required=False,
                constraints=[
                    FieldConstraint(ConstraintType.REGEX, r'^[^@]+@[^@]+\.[^@]+$', "無効なメール形式")
                ]
            ),
            FieldDefinition(
                name="age",
                field_type=FieldType.INTEGER,
                required=False,
                constraints=[
                    FieldConstraint(ConstraintType.MIN_VALUE, 18),
                    FieldConstraint(ConstraintType.MAX_VALUE, 100)
                ]
            ),
            FieldDefinition(
                name="created_at",
                field_type=FieldType.DATETIME,
                required=True,
                constraints=[
                    FieldConstraint(ConstraintType.NOT_NULL, True)
                ]
            )
        ]
        
        return SchemaDefinition(
            name="user_data_schema",
            version="1.0.0",
            fields=fields,
            description="ユーザーデータスキーマ定義"
        )