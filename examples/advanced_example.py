"""
Advanced training example with custom configuration
"""

from embedding_finetuner import EmbeddingFinetuner
from embedding_finetuner.data import load_dataset, create_sample_data
from embedding_finetuner.utils import setup_logging

def main():
    # Setup logging
    setup_logging("INFO")
    
    # Load your own data (replace with actual data paths)
    # train_data = load_dataset("path/to/your/train_data.jsonl")
    # val_data = load_dataset("path/to/your/val_data.jsonl")
    
    # For demo, create sample data
    train_data = create_sample_data(num_samples=5000, data_format="pairs")
    val_data = create_sample_data(num_samples=1000, data_format="pairs")
    
    # Advanced configuration
    lora_config = {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["query", "key", "value", "dense"],
        "lora_dropout": 0.05
    }
    
    loss_kwargs = {
        "scale": 20.0
    }
    
    # Initialize finetuner with advanced settings
    finetuner = EmbeddingFinetuner(
        model_name="sentence-transformers/all-mpnet-base-v2",
        loss_function="MultipleNegativesRankingLoss",
        loss_kwargs=loss_kwargs,
        use_lora=True,
        lora_config=lora_config,
        mixed_precision="bf16",  # Use bfloat16 if supported
        gradient_checkpointing=True,
        max_length=512,
        pooling_mode="mean",
        normalize_embeddings=True
    )
    
    # Advanced training configuration
    finetuner.train(
        train_data=train_data,
        val_data=val_data,
        epochs=5,
        batch_size=16,  # Smaller batch size for larger model
        learning_rate=1e-5,  # Lower learning rate
        weight_decay=0.01,
        warmup_steps=1000,
        max_grad_norm=1.0,
        eval_steps=250,
        save_steps=500,
        logging_steps=50,
        output_dir="./output/advanced_training",
        data_format="pairs",
        negative_sampling=True,
        num_negatives=3,  # More negatives
        use_wandb=True,  # Enable W&B logging
        wandb_project="embedding-finetuning-advanced"
    )
    
    print("Advanced training completed!")

if __name__ == "__main__":
    main()