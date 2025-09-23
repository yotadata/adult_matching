"""
Backend ML Package

Unified machine learning package for Two-Tower recommendation models
"""

from pathlib import Path

# Package configuration
__version__ = "1.0.0"
__author__ = "ML Team"

# Package root directory
PACKAGE_ROOT = Path(__file__).parent

# Standard configuration
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_HIDDEN_UNITS = [512, 256, 128]
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001

# Standard model configuration
STANDARD_MODEL_CONFIG = {
    "user_embedding_dim": DEFAULT_EMBEDDING_DIM,
    "item_embedding_dim": DEFAULT_EMBEDDING_DIM,
    "user_hidden_units": DEFAULT_HIDDEN_UNITS,
    "item_hidden_units": DEFAULT_HIDDEN_UNITS,
    "batch_size": DEFAULT_BATCH_SIZE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": 100,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "dropout_rate": 0.2,
    "l2_regularization": 0.01
}

# Performance requirements
PERFORMANCE_REQUIREMENTS = {
    "max_training_time_hours": 2,
    "max_inference_latency_ms": 500,
    "min_auc_roc": 0.85,
    "min_auc_pr": 0.70,
    "max_model_size_mb": 200
}

# Data configuration
DATA_CONFIG = {
    "min_user_interactions": 5,
    "min_item_interactions": 3,
    "negative_sampling_ratio": 2,
    "max_features_per_entity": 50
}

# Exports
__all__ = [
    'PACKAGE_ROOT',
    'DEFAULT_EMBEDDING_DIM',
    'DEFAULT_HIDDEN_UNITS',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_LEARNING_RATE',
    'STANDARD_MODEL_CONFIG',
    'PERFORMANCE_REQUIREMENTS',
    'DATA_CONFIG'
]