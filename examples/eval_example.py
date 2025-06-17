"""
Evaluation example
"""

from embedding_finetuner import EmbeddingFinetuner
from embedding_finetuner.data import create_sample_data
from embedding_finetuner.evaluation import evaluate_model

def main():
    # Load trained model
    model_path = "./output/basic_training/final_model"
    
    try:
        finetuner = EmbeddingFinetuner.from_pretrained(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("No trained model found. Please run basic_training.py first.")
        return
    
    # Create test data
    test_data = create_sample_data(num_samples=500, data_format="pairs")
    
    # Evaluate model
    results = evaluate_model(
        finetuner.model,
        test_data,
        batch_size=32,
        metrics=["similarity", "classification"]
    )
    
    # Print results
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()