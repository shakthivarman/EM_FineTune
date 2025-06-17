#!/usr/bin/env python3
"""
Evaluation script for embedding models
"""

import argparse
import logging
import json
from pathlib import Path

from embedding_finetuner import EmbeddingFinetuner
from embedding_finetuner.data import load_dataset
from embedding_finetuner.evaluation import evaluate_model
from embedding_finetuner.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embedding model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metrics", nargs="+", default=["similarity", "retrieval", "classification"])
    parser.add_argument("--output_file", type=str, help="Path to save evaluation results")
    parser.add_argument("--log_level", type=str, default="INFO")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    finetuner = EmbeddingFinetuner.from_pretrained(args.model_path)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = load_dataset(args.test_data)
    logger.info(f"Test samples: {len(test_data)}")
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate_model(
        finetuner.model,
        test_data,
        batch_size=args.batch_size,
        metrics=args.metrics
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()