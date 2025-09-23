"""
Schema management system for data validation and type enforcement.
Provides JSON schema definitions and validation for data structures.
"""

from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for schema validation"""
    VIDEO = "video"
    USER = "user"
    INTERACTION = "interaction"
    REVIEW = "review"
    EMBEDDING = "embedding"
    TAG = "tag"
    PERFORMER = "performer"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"


@dataclass
class SchemaValidationResult:
    """Result of schema validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: Optional[str] = None
    validated_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class SchemaManager:
    """
    Manages JSON schemas for data validation and type enforcement.
    Provides schema definitions, validation, and evolution support.
    """
    
    def __init__(self, schema_dir: Optional[Path] = None):
        self.schema_dir = schema_dir or Path(__file__).parent / "definitions"
        self.schemas: Dict[str, Dict] = {}
        
        # Load default schemas
        self._load_default_schemas()
        
        # Load custom schemas if directory exists
        if self.schema_dir.exists():
            self._load_custom_schemas()
    
    def _load_default_schemas(self):
        """Load default schema definitions"""
        
        # Video schema
        video_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://adult-matching.com/schemas/video.json",
            "title": "Video Metadata Schema",
            "type": "object",
            "required": ["id", "title", "content_id", "source"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique video identifier"
                },
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": "Video title"
                },
                "content_id": {
                    "type": "string",
                    "pattern": "^[A-Z0-9\\-_]{3,20}$",
                    "description": "External content identifier"
                },
                "source": {
                    "type": "string",
                    "enum": ["dmm", "fanza", "scraped", "api"],
                    "description": "Data source"
                },
                "external_id": {
                    "type": "string",
                    "description": "External system identifier"
                },
                "thumbnails": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Thumbnail image URLs"
                },
                "description": {
                    "type": ["string", "null"],
                    "maxLength": 5000,
                    "description": "Video description"
                },
                "duration_minutes": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 1440,
                    "description": "Duration in minutes"
                },
                "price": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Price in yen"
                },
                "release_date": {
                    "type": ["string", "null"],
                    "description": "Release date"
                },
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Genre tags"
                },
                "performers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Performer names"
                },
                "studio": {
                    "type": ["string", "null"],
                    "description": "Production studio"
                },
                "rating": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Average rating"
                },
                "view_count": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "description": "View count"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata"
                },
                "created_at": {
                    "type": "string",
                    "description": "Creation timestamp"
                },
                "updated_at": {
                    "type": "string",
                    "description": "Update timestamp"
                }
            },
            "additionalProperties": True
        }
        
        # User schema
        user_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://adult-matching.com/schemas/user.json",
            "title": "User Profile Schema",
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "User identifier"
                },
                "email": {
                    "type": ["string", "null"],
                    "description": "User email"
                },
                "profile": {
                    "type": "object",
                    "properties": {
                        "age_range": {
                            "type": ["string", "null"],
                            "enum": ["18-25", "26-35", "36-45", "46-55", "55+", null]
                        },
                        "gender": {
                            "type": ["string", "null"],
                            "enum": ["male", "female", "other", null]
                        },
                        "preferences": {
                            "type": "object",
                            "properties": {
                                "genres": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "performers": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "studios": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "activity_stats": {
                    "type": "object",
                    "properties": {
                        "total_likes": {"type": "integer", "minimum": 0},
                        "total_views": {"type": "integer", "minimum": 0},
                        "last_activity": {"type": "string"}
                    }
                },
                "created_at": {
                    "type": "string",
                    "description": "Creation timestamp"
                },
                "updated_at": {
                    "type": "string",
                    "description": "Update timestamp"
                }
            },
            "additionalProperties": True
        }
        
        # Interaction schema
        interaction_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://adult-matching.com/schemas/interaction.json",
            "title": "User Interaction Schema",
            "type": "object",
            "required": ["user_id", "video_id", "interaction_type", "timestamp"],
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier"
                },
                "video_id": {
                    "type": "string",
                    "description": "Video identifier"
                },
                "interaction_type": {
                    "type": "string",
                    "enum": ["like", "skip", "view", "share", "favorite"],
                    "description": "Type of interaction"
                },
                "timestamp": {
                    "type": "string",
                    "description": "Interaction timestamp"
                },
                "context": {
                    "type": "object",
                    "properties": {
                        "device_type": {"type": "string"},
                        "user_agent": {"type": "string"},
                        "session_id": {"type": "string"},
                        "source": {"type": "string"}
                    },
                    "description": "Interaction context"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional interaction metadata"
                }
            },
            "additionalProperties": True
        }
        
        # Review schema
        review_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://adult-matching.com/schemas/review.json",
            "title": "Review Data Schema",
            "type": "object",
            "required": ["content_id", "rating"],
            "properties": {
                "content_id": {
                    "type": "string",
                    "description": "Content identifier"
                },
                "rating": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Rating score"
                },
                "review_text": {
                    "type": ["string", "null"],
                    "minLength": 10,
                    "maxLength": 10000,
                    "description": "Review text content"
                },
                "reviewer_id": {
                    "type": ["string", "null"],
                    "description": "Reviewer identifier"
                },
                "helpfulness": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "description": "Helpfulness votes"
                },
                "verified_purchase": {
                    "type": "boolean",
                    "description": "Whether purchase is verified"
                },
                "review_date": {
                    "type": "string",
                    "description": "Review timestamp"
                },
                "language": {
                    "type": "string",
                    "pattern": "^[a-z]{2}$",
                    "description": "Review language code"
                }
            },
            "additionalProperties": True
        }
        
        # Embedding schema
        embedding_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://adult-matching.com/schemas/embedding.json",
            "title": "Vector Embedding Schema",
            "type": "object",
            "required": ["entity_id", "entity_type", "embedding", "model_version"],
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity identifier"
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["user", "video", "tag", "performer"],
                    "description": "Type of entity"
                },
                "embedding": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 384,
                    "maxItems": 1024,
                    "description": "Vector embedding values"
                },
                "model_version": {
                    "type": "string",
                    "description": "Model version used for embedding"
                },
                "dimensions": {
                    "type": "integer",
                    "minimum": 384,
                    "maximum": 1024,
                    "description": "Embedding dimensions"
                },
                "created_at": {
                    "type": "string",
                    "description": "Creation timestamp"
                },
                "metadata": {
                    "type": "object",
                    "description": "Embedding metadata"
                }
            },
            "additionalProperties": True
        }
        
        self.schemas.update({
            DataType.VIDEO.value: video_schema,
            DataType.USER.value: user_schema,
            DataType.INTERACTION.value: interaction_schema,
            DataType.REVIEW.value: review_schema,
            DataType.EMBEDDING.value: embedding_schema
        })
    
    def _load_custom_schemas(self):
        """Load custom schema definitions from files"""
        if not self.schema_dir.exists():
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                
                schema_name = schema_file.stem
                self.schemas[schema_name] = schema
                
                logger.info(f"Loaded custom schema: {schema_name}")
                
            except Exception as e:
                logger.error(f"Failed to load schema from {schema_file}: {e}")
    
    def get_schema(self, schema_name: str) -> Optional[Dict]:
        """Get schema definition by name"""
        return self.schemas.get(schema_name)
    
    def list_schemas(self) -> List[str]:
        """List all available schema names"""
        return list(self.schemas.keys())
    
    def validate_data(
        self, 
        data: Union[Dict, List[Dict]], 
        schema_name: str
    ) -> SchemaValidationResult:
        """
        Validate data against a schema (basic validation)
        
        Args:
            data: Data to validate (single dict or list of dicts)
            schema_name: Name of schema to validate against
            
        Returns:
            SchemaValidationResult with validation outcome
        """
        if schema_name not in self.schemas:
            return SchemaValidationResult(
                is_valid=False,
                errors=[f"Schema '{schema_name}' not found"],
                schema_version=None,
                validated_count=0
            )
        
        schema = self.schemas[schema_name]
        errors = []
        warnings = []
        validated_count = 0
        
        # Handle single dict or list of dicts
        data_items = data if isinstance(data, list) else [data]
        
        for i, item in enumerate(data_items):
            try:
                # Basic validation - check required fields
                required_fields = schema.get('required', [])
                for field in required_fields:
                    if field not in item:
                        errors.append(f"Item {i}: Missing required field '{field}'")
                
                # Check data types for known fields
                properties = schema.get('properties', {})
                for field, value in item.items():
                    if field in properties:
                        prop_def = properties[field]
                        expected_type = prop_def.get('type')
                        
                        if expected_type and not self._check_type(value, expected_type):
                            errors.append(f"Item {i}: Field '{field}' has incorrect type")
                
                if not any(f"Item {i}:" in error for error in errors[-10:]):  # No errors for this item
                    validated_count += 1
                    
            except Exception as e:
                errors.append(f"Item {i}: Validation error - {str(e)}")
        
        # Generate warnings for common issues
        if isinstance(data, list) and len(data) == 0:
            warnings.append("Empty data list provided")
        
        schema_version = schema.get("version", "1.0")
        
        return SchemaValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            schema_version=schema_version,
            validated_count=validated_count
        )
    
    def _check_type(self, value, expected_type):
        """Basic type checking helper"""
        if isinstance(expected_type, list):
            # Handle union types like ["string", "null"]
            return any(self._check_single_type(value, t) for t in expected_type)
        else:
            return self._check_single_type(value, expected_type)
    
    def _check_single_type(self, value, type_name):
        """Check if value matches a single type"""
        if type_name == "null" and value is None:
            return True
        elif type_name == "string" and isinstance(value, str):
            return True
        elif type_name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        elif type_name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        elif type_name == "boolean" and isinstance(value, bool):
            return True
        elif type_name == "array" and isinstance(value, list):
            return True
        elif type_name == "object" and isinstance(value, dict):
            return True
        return False
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        schema_name: str,
        sample_size: Optional[int] = None
    ) -> SchemaValidationResult:
        """
        Validate DataFrame against schema
        
        Args:
            df: DataFrame to validate
            schema_name: Schema name
            sample_size: If provided, validate only a sample of rows
            
        Returns:
            SchemaValidationResult
        """
        if df.empty:
            return SchemaValidationResult(
                is_valid=True,
                warnings=["DataFrame is empty"],
                schema_version=self.schemas.get(schema_name, {}).get("version", "1.0"),
                validated_count=0
            )
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Validating sample of {sample_size} rows from {len(df)} total rows")
        else:
            sample_df = df
        
        # Convert DataFrame to list of dictionaries
        data_list = sample_df.to_dict(orient='records')
        
        # Validate the data
        result = self.validate_data(data_list, schema_name)
        
        # Adjust validated count if we sampled
        if sample_size and len(df) > sample_size:
            result.warnings.append(f"Validated sample of {sample_size} rows from {len(df)} total")
        
        return result
    
    def infer_schema_from_dataframe(
        self, 
        df: pd.DataFrame,
        schema_name: str,
        description: str = "Auto-generated schema"
    ) -> Dict:
        """
        Infer JSON schema from DataFrame structure
        
        Args:
            df: DataFrame to analyze
            schema_name: Name for the schema
            description: Schema description
            
        Returns:
            JSON schema dictionary
        """
        if df.empty:
            raise ValueError("Cannot infer schema from empty DataFrame")
        
        properties = {}
        required_fields = []
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                # All null column
                properties[column] = {
                    "type": ["null"],
                    "description": f"Column {column} (all null values)"
                }
                continue
            
            # Determine the primary type
            if pd.api.types.is_integer_dtype(col_data):
                col_type = "integer"
                type_constraints = {
                    "minimum": int(col_data.min()),
                    "maximum": int(col_data.max())
                }
            elif pd.api.types.is_float_dtype(col_data):
                col_type = "number"
                type_constraints = {
                    "minimum": float(col_data.min()),
                    "maximum": float(col_data.max())
                }
            elif pd.api.types.is_bool_dtype(col_data):
                col_type = "boolean"
                type_constraints = {}
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_type = "string"
                type_constraints = {}
            else:
                col_type = "string"
                if col_data.dtype == 'object':
                    # Check for string length constraints
                    string_lengths = col_data.astype(str).str.len()
                    type_constraints = {
                        "minLength": int(string_lengths.min()),
                        "maxLength": int(string_lengths.max())
                    }
                else:
                    type_constraints = {}
            
            # Check if column has null values
            null_count = df[column].isnull().sum()
            if null_count == 0:
                # Required field
                required_fields.append(column)
                properties[column] = {
                    "type": col_type,
                    "description": f"Column {column}",
                    **type_constraints
                }
            else:
                # Optional field
                properties[column] = {
                    "type": [col_type, "null"],
                    "description": f"Column {column} (nullable)",
                    **type_constraints
                }
        
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://adult-matching.com/schemas/{schema_name}.json",
            "title": f"{schema_name.replace('_', ' ').title()} Schema",
            "description": description,
            "type": "object",
            "required": required_fields,
            "properties": properties,
            "additionalProperties": True
        }
        
        return schema
    
    def add_custom_schema(
        self, 
        schema_name: str, 
        schema: Dict,
        save_to_file: bool = True
    ) -> bool:
        """
        Add a custom schema definition
        
        Args:
            schema_name: Name for the schema
            schema: JSON schema dictionary
            save_to_file: Whether to save to file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add to memory
            self.schemas[schema_name] = schema
            
            # Save to file if requested
            if save_to_file:
                self.schema_dir.mkdir(parents=True, exist_ok=True)
                schema_file = self.schema_dir / f"{schema_name}.json"
                
                with open(schema_file, 'w') as f:
                    json.dump(schema, f, indent=2)
                
                logger.info(f"Schema '{schema_name}' saved to {schema_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add schema '{schema_name}': {e}")
            return False
    
    def export_schemas(self, output_dir: Path, format: str = "json") -> List[Path]:
        """
        Export all schemas to files
        
        Args:
            output_dir: Output directory
            format: Export format ("json")
            
        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        for schema_name, schema in self.schemas.items():
            if format.lower() == "json":
                output_file = output_dir / f"{schema_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(schema, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            created_files.append(output_file)
            logger.info(f"Exported schema '{schema_name}' to {output_file}")
        
        return created_files