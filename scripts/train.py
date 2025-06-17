#!/usr/bin/env python3
"""
Training script for embedding finetuning
"""

import argparse
import logging
from pathlib import Path
import yaml

from embedding_finetuner import EmbeddingFinetuner
from embedding_finetuner.data import load_dataset, create_sample_data
from embedding_finetuner.utils import setup_logging, DEFAULT_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Train embedding model")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--loss_function", type=str, default="MultipleNegativesRankingLoss")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="embedding-finetuning")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume training from checkpoint")
    parser.add_argument("--create_sample_data", action="store_true", help="Create sample data for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser.parse_args()

def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        file_config = load_config_file(args.config)
        config.update(file_config)
    
    # Override with command line arguments
    config.update({
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "use_lora": args.use_lora,
        "mixed_precision": args.mixed_precision,
        "loss_function": args.loss_function,
        "seed": args.seed
    })
    
    logger.info(f"Configuration: {config}")
    
    # Create sample data if requested
    if args.create_sample_data:
        logger.info("Creating sample data...")
        train_data = create_sample_data(num_samples=1000, data_format="pairs")
        val_data = create_sample_data(num_samples=200, data_format="pairs")
    else:
        # Load training data
        if not args.train_data:
            raise ValueError("Either provide --train_data or use --create_sample_data")
        
        logger.info(f"Loading training data from {args.train_data}")
        train_data = load_dataset(args.train_data)
        
        val_data = None
        if args.val_data:
            logger.info(f"Loading validation data from {args.val_data}")
            val_data = load_dataset(args.val_data)
    
    logger.info(f"Training samples: {len(train_data)}")
    if val_data:
        logger.info(f"Validation samples: {len(val_data)}")
    
    # Initialize finetuner
    finetuner = EmbeddingFinetuner(
        model_name=config["model_name"],
        loss_function=config["loss_function"],
        use_lora=config["use_lora"],
        mixed_precision=config["mixed_precision"],
        seed=config["seed"]
    )
    
    # Start training
    logger.info("Starting training...")
    finetuner.train(
        train_data=train_data,
        val_data=val_data,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()