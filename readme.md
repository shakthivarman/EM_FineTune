# Embedding Model Finetuning Framework

A comprehensive, hardware-agnostic framework for finetuning embedding models using the latest techniques and optimizations.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

## 🔥 Features

### Hardware Agnostic Support
- **Auto-detection**: Automatically detects and uses CUDA, Apple Silicon (MPS), or CPU
- **Distributed Training**: Built-in support with Accelerate library
- **Mixed Precision**: FP16/BF16 support for memory efficiency

### Latest Training Techniques
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient finetuning with PEFT library
- **Gradient Checkpointing**: Reduces memory usage significantly
- **Multiple Loss Functions**: 
  - MultipleNegativesRankingLoss (most popular)
  - InfoNCE, Contrastive, Triplet, AnglE losses

### Modern Architecture Support
- **Sentence Transformers**: Native support for popular embedding models
- **Any Transformer**: Also works with raw transformer models from HuggingFace
- **Flexible Pooling**: Mean, CLS, Max pooling strategies

### Comprehensive Evaluation
- **Similarity Tasks**: Accuracy, F1, correlation metrics
- **Retrieval Tasks**: Hits@K, MRR evaluation
- **Classification**: K-NN and clustering-based evaluation
- **Embedding Quality**: Intrinsic metrics for embedding analysis

### Production Ready
- **Checkpointing**: Resume training from any point
- **Logging**: Comprehensive logging with W&B integration
- **Configuration**: YAML configs and command-line interface
- **Memory Optimization**: Smart batch processing and memory management

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project files
mkdir embedding-finetuning
cd embedding-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch (GPU version recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the project
pip install -e .
```

### Verify Installation

```bash
# Test installation
python -c "import embedding_finetuner; print('✅ Installation successful!')"

# Check device detection
python -c "import embedding_finetuner; print(f'Device: {embedding_finetuner.get_device_info()}')"
```

### Run Your First Training (2 minutes)

```bash
# Quick training with sample data
python scripts/train.py --create_sample_data --epochs 2 --batch_size 16

# Or use the quick fix script
python quick_fix.py
```

### Basic Python Usage

```python
from embedding_finetuner import EmbeddingFinetuner
from embedding_finetuner.data import create_sample_data

# Create sample data
train_data = create_sample_data(num_samples=1000, data_format="pairs")

# Initialize and train
finetuner = EmbeddingFinetuner(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    loss_function="MultipleNegativesRankingLoss",
    use_lora=True,
    mixed_precision="fp16"
)

finetuner.train(
    train_data=train_data,
    epochs=3,
    batch_size=32,
    output_dir="./output"
)
```

## 📁 Project Structure

```
embedding-finetuning/
├── src/
│   └── embedding_finetuner/
│       ├── __init__.py          # Package initialization
│       ├── trainer.py           # Main training logic
│       ├── models.py            # Model implementations
│       ├── losses.py            # Loss functions
│       ├── data.py              # Data handling
│       ├── evaluation.py        # Evaluation metrics
│       └── utils.py             # Utility functions
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── examples/
│   ├── basic_training.py        # Basic usage example
│   ├── advanced_training.py     # Advanced configuration
│   └── evaluation_example.py    # Evaluation example
├── configs/
│   ├── default.yaml             # Default configuration
│   ├── lora_advanced.yaml       # Advanced LoRA settings
│   └── production.yaml          # Production configuration
├── quick_fix.py                 # Quick training fix
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🎯 Usage Examples

### Command Line Training

```bash
# Basic training with sample data
python scripts/train.py --create_sample_data --epochs 3 --batch_size 32

# Train with your own data
python scripts/train.py --train_data train.jsonl --val_data val.jsonl --epochs 5

# Use configuration file
python scripts/train.py --config configs/default.yaml --train_data train.jsonl

# With Weights & Biases logging
python scripts/train.py --create_sample_data --use_wandb --wandb_project my-embeddings

# Advanced LoRA training
python scripts/train.py --config configs/lora_advanced.yaml --create_sample_data
```

### Run Examples

```bash
# Quick test (30 seconds)
python examples/basic_training.py

# Advanced training (5 minutes)
python examples/advanced_training.py

# Evaluate trained model
python examples/evaluation_example.py
```

### Evaluate Models

```bash
# Comprehensive evaluation
python scripts/evaluate.py --model_path ./output/final_model --test_data test.jsonl

# Specific metrics only
python scripts/evaluate.py --model_path ./output/final_model --test_data test.jsonl --metrics similarity retrieval
```

## 📊 Data Formats

The framework supports multiple data formats:

### Pairs Format (most common)
```json
{"text1": "First sentence", "text2": "Second sentence", "label": 1}
{"text1": "Another sentence", "text2": "Different sentence", "label": 0}
```

### Triplets Format
```json
{"anchor": "Query text", "positive": "Relevant text", "negative": "Irrelevant text"}
```

### Single Text Format
```json
{"text": "Some text", "label": "category_name"}
```

## ⚙️ Configuration

The framework supports YAML configuration files for easy customization:

### Default Configuration (`configs/default.yaml`)
```yaml
model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  max_length: 512
  pooling_mode: "mean"

training:
  loss_function: "MultipleNegativesRankingLoss"
  epochs: 3
  batch_size: 32
  learning_rate: 2e-5

optimization:
  use_lora: true
  mixed_precision: "fp16"
  gradient_checkpointing: true

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["query", "key", "value", "dense"]
```

### Available Configurations
- **`configs/default.yaml`**: Basic configuration for getting started
- **`configs/lora_advanced.yaml`**: Advanced LoRA settings for better performance
- **`configs/production.yaml`**: Production-ready configuration for large-scale training

## 🔧 Advanced Usage

### Custom Loss Functions

```python
# Use different loss functions
finetuner = EmbeddingFinetuner(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    loss_function="InfoNCELoss",
    loss_kwargs={"temperature": 0.05}
)
```

### Advanced LoRA Configuration

```python
lora_config = {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": ["query", "key", "value", "dense"],
    "lora_dropout": 0.05
}

finetuner = EmbeddingFinetuner(
    model_name="sentence-transformers/all-mpnet-base-v2",
    use_lora=True,
    lora_config=lora_config,
    mixed_precision="bf16"
)
```

### Training with Your Own Data

```python
from embedding_finetuner.data import load_dataset

# Load your data
train_data = load_dataset("my_training_data.jsonl")
val_data = load_dataset("my_validation_data.jsonl")

# Train
finetuner.train(
    train_data=train_data,
    val_data=val_data,
    epochs=5,
    batch_size=32,
    negative_sampling=False,  # Disable if you encounter batch errors
    use_wandb=True,
    wandb_project="my-embedding-project"
)
```

### Evaluation

```python
from embedding_finetuner.evaluation import evaluate_model

# Load trained model
finetuner = EmbeddingFinetuner.from_pretrained("./output/final_model")

# Comprehensive evaluation
results = evaluate_model(
    finetuner.model,
    test_data,
    metrics=["similarity", "retrieval", "classification"]
)

print(f"Similarity F1: {results['similarity_f1']:.4f}")
print(f"Retrieval Hits@10: {results['hits_at_10']:.4f}")
```

### Using Trained Models

```python
# Load your trained model
finetuner = EmbeddingFinetuner.from_pretrained("./output/final_model")

# Get embeddings
texts = ["Hello world", "Machine learning is amazing"]
embeddings = finetuner.model.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)
print(f"Similarity: {similarity[0][1]:.4f}")
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'embedding_finetuner'"
```bash
# Make sure you're in the project directory and run:
pip install -e .
```

#### 2. KeyError: 'input_ids_neg0' (Batch collation error)
```bash
# Use the quick fix script or disable negative sampling:
python quick_fix.py
# Or add negative_sampling=False to your training call
```

#### 3. CUDA Out of Memory
```bash
# Reduce batch size:
python scripts/train.py --create_sample_data --batch_size 8

# Or use CPU:
python scripts/train.py --create_sample_data --batch_size 16 --mixed_precision no
```

#### 4. Import/Installation Errors
```bash
# Reinstall with specific PyTorch version:
pip uninstall torch transformers
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

#### 5. Windows Build Errors
```bash
# Install pre-compiled wheels:
pip install torch torchvision torchaudio
pip install -e .
```

### Hardware-Specific Setup

#### **GPU (CUDA)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

#### **Apple Silicon (M1/M2)**
```bash
# Standard installation (MPS auto-detected)
pip install -e .
```

#### **CPU Only**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## 📈 Monitoring and Logging

### Weights & Biases Integration

```bash
# Install wandb
pip install wandb
wandb login

# Train with W&B logging
python scripts/train.py --create_sample_data --use_wandb --wandb_project my-embeddings
```

### Console Logging

Training progress is automatically displayed with:
- Current loss and learning rate
- Training speed (samples/sec)
- Memory usage
- Evaluation metrics

## 🎯 Performance Tips

1. **Use GPU**: Training is 10-100x faster on GPU
2. **Optimize Batch Size**: Start with 32, reduce if out of memory
3. **Enable Mixed Precision**: Use `--mixed_precision fp16` for faster training
4. **Use LoRA**: Reduces memory usage and training time
5. **Monitor with W&B**: Track experiments and compare results

## 🧪 Testing Your Installation

### Quick Tests

```bash
# 1. Import test
python -c "import embedding_finetuner; print('✅ Import successful')"

# 2. Device test
python -c "import embedding_finetuner; print(embedding_finetuner.get_device_info())"

# 3. Quick training test (30 seconds)
python quick_fix.py

# 4. Example test
python examples/basic_training.py
```

### Comprehensive Test

```python
# test_installation.py
import embedding_finetuner
from embedding_finetuner.data import create_sample_data

print("🧪 Testing installation...")

# Test 1: Basic import
print("✅ Import successful")

# Test 2: Device detection
print(f"Device: {embedding_finetuner.get_device_info()}")

# Test 3: Sample data creation
data = create_sample_data(10)
print(f"✅ Created {len(data)} samples")

# Test 4: Model initialization
finetuner = embedding_finetuner.EmbeddingFinetuner(
    "sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Model initialization successful")

# Test 5: Quick encoding
embeddings = finetuner.model.encode(["test sentence"])
print(f"✅ Encoding successful: {embeddings.shape}")

print("🎉 All tests passed!")
```

## 📚 Additional Resources

- **Compatibility Analysis**: See `COMPATIBILITY_ANALYSIS.md` for detailed dependency information
- **Quick Start Guide**: See `QUICK_START_GUIDE.md` for step-by-step setup
- **Examples Directory**: Check `examples/` for working code samples
- **Configuration Files**: See `configs/` for different training setups

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for the excellent embedding framework
- [Hugging Face](https://huggingface.co/) for transformers and PEFT libraries
- [Microsoft](https://github.com/microsoft/LoRA) for the LoRA technique

## 📧 Support

For questions and support, please:
1. Check the troubleshooting section above
2. Run the quick fix script: `python quick_fix.py`
3. Try the examples: `python examples/basic_training.py`
4. Open an issue on GitHub with error details

---

**Ready to start?** Run `python quick_fix.py` for immediate results! 🚀