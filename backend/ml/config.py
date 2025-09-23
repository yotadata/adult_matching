"""
ML Configuration Classes

Machine Learning configuration data classes
"""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration class"""
    # Model architecture
    user_embedding_dim: int = 768
    item_embedding_dim: int = 768
    user_hidden_units: List[int] = None
    item_hidden_units: List[int] = None
    dropout_rate: float = 0.2
    l2_regularization: float = 0.01
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Early stopping & scheduling
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    min_learning_rate: float = 1e-6
    
    # Data parameters
    negative_sampling_ratio: int = 2
    shuffle: bool = True
    
    # Optimization
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.user_hidden_units is None:
            self.user_hidden_units = [512, 256, 128]
        if self.item_hidden_units is None:
            self.item_hidden_units = [512, 256, 128]
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "auc"]

@dataclass
class TrainingStats:
    """Training statistics"""
    start_time: str
    end_time: Optional[str] = None
    total_duration_seconds: Optional[float] = None
    
    # Data statistics
    user_count: int = 0
    item_count: int = 0
    interaction_count: int = 0
    train_samples: int = 0
    validation_samples: int = 0
    
    # Model statistics
    total_params: int = 0
    trainable_params: int = 0
    user_tower_params: int = 0
    item_tower_params: int = 0
    
    # Training performance
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    final_train_accuracy: Optional[float] = None
    final_val_accuracy: Optional[float] = None
    best_epoch: int = 0
    
    # Evaluation metrics
    test_auc: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None