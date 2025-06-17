import os
import json
import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_info() -> str:
    """Get information about available devices"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        return f"CUDA ({device_count} GPUs available, using {device_name}, {memory_total:.1f}GB)"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    else:
        return "CPU"

def save_config(config: Dict[str, Any], save_path: Path):
    """Save configuration to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )

def count_parameters(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters in model"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info["cuda_allocated"] = torch.cuda.memory_allocated() / 1e9
        memory_info["cuda_reserved"] = torch.cuda.memory_reserved() / 1e9
        memory_info["cuda_max_allocated"] = torch.cuda.max_memory_allocated() / 1e9
    
    try:
        import psutil
        process = psutil.Process()
        memory_info["ram_usage"] = process.memory_info().rss / 1e9
        memory_info["ram_percent"] = process.memory_percent()
    except ImportError:
        pass
    
    return memory_info

# Configuration templates
DEFAULT_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "loss_function": "MultipleNegativesRankingLoss",
    "use_lora": True,
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
    "max_length": 512,
    "pooling_mode": "mean",
    "normalize_embeddings": True,
    "seed": 42,
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "eval_steps": 500,
    "save_steps": 1000,
    "logging_steps": 100
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["query", "key", "value", "dense"],
    "lora_dropout": 0.1,
    "bias": "none"
}