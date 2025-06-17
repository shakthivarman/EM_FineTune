# Quick fix for the training error
# Save this as quick_fix.py and run it

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from embedding_finetuner import EmbeddingFinetuner
from embedding_finetuner.data import create_sample_data

print("🚀 Starting quick training fix...")

# Create sample data
print("Creating sample data...")
train_data = create_sample_data(num_samples=200, data_format="pairs")
val_data = create_sample_data(num_samples=50, data_format="pairs")

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Initialize finetuner
print("Initializing model...")
finetuner = EmbeddingFinetuner(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    loss_function="MultipleNegativesRankingLoss",
    use_lora=True,
    mixed_precision="fp16"
)

# Train with fixed settings (no negative sampling)
print("Starting training...")
finetuner.train(
    train_data=train_data,
    val_data=val_data,
    epochs=2,
    batch_size=8,  # Smaller batch size
    learning_rate=2e-5,
    output_dir="./output",
    negative_sampling=False,  # Disable negative sampling to fix the error
    logging_steps=10,
    eval_steps=100,
    save_steps=200
)

print("✅ Training completed! Model saved to ./output")
print("🎉 Quick fix successful!")