"""
Embedding Model Finetuning Framework

A comprehensive, hardware-agnostic framework for finetuning embedding models 
using the latest techniques and optimizations.

Example usage:
    from embedding_finetuner import EmbeddingFinetuner
    from embedding_finetuner.data import create_sample_data
    
    # Create sample data
    train_data = create_sample_data(num_samples=1000, data_format="pairs")
    
    # Initialize finetuner
    finetuner = EmbeddingFinetuner(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        loss_function="MultipleNegativesRankingLoss",
        use_lora=True,
        mixed_precision="fp16"
    )
    
    # Train model
    finetuner.train(
        train_data=train_data,
        epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        output_dir="./output"
    )
"""

from .trainer import EmbeddingFinetuner
from .models import EmbeddingModel
from .losses import get_loss_function
from .data import EmbeddingDataset, load_dataset, create_sample_data
from .evaluation import evaluate_model, compute_embedding_quality_metrics
from .utils import (
    set_seed, 
    get_device_info, 
    setup_logging, 
    count_parameters, 
    get_memory_usage,
    DEFAULT_CONFIG,
    LORA_CONFIG
)

__version__ = "0.1.0"
__author__ = "Embedding Finetuner Team"
__email__ = "support@example.com"
__description__ = "Hardware-agnostic embedding model finetuning framework"

__all__ = [
    # Main classes
    "EmbeddingFinetuner",
    "EmbeddingModel",
    "EmbeddingDataset",
    
    # Functions
    "get_loss_function",
    "load_dataset",
    "create_sample_data",
    "evaluate_model",
    "compute_embedding_quality_metrics",
    
    # Utilities
    "set_seed",
    "get_device_info", 
    "setup_logging",
    "count_parameters",
    "get_memory_usage",
    
    # Configuration
    "DEFAULT_CONFIG",
    "LORA_CONFIG",
]

# Version info
def get_version():
    """Get the current version of the package."""
    return __version__

def get_device_status():
    """Get current device status and availability."""
    return get_device_info()

# Package info
PACKAGE_INFO = {
    "name": "embedding-finetuner",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "Hardware agnostic (CPU, GPU, Apple Silicon)",
        "Parameter-efficient finetuning with LoRA",
        "Mixed precision training (FP16/BF16)",
        "Multiple loss functions",
        "Distributed training support", 
        "Comprehensive evaluation metrics",
        "W&B integration",
        "Checkpointing and resuming"
    ]
}