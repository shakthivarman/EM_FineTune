import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset
from typing import List, Dict, Union, Optional, Tuple
import random
import logging

logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):
    """
    Flexible dataset for embedding training supporting multiple formats
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        data_format: str = "pairs",
        negative_sampling: bool = True,
        num_negatives: int = 1
    ):
        """
        Args:
            data: List of data samples
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            data_format: Format of data - 'pairs', 'triplets', 'single'
            negative_sampling: Whether to perform negative sampling
            num_negatives: Number of negative samples per positive
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_format = data_format
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
        
        # Create text corpus for negative sampling
        if negative_sampling:
            self.text_corpus = self._build_text_corpus()
    
    def _build_text_corpus(self) -> List[str]:
        """Build corpus of all texts for negative sampling"""
        corpus = set()
        
        for item in self.data:
            if self.data_format == "pairs":
                corpus.add(item.get("text1", ""))
                corpus.add(item.get("text2", ""))
                if "negative" in item:
                    corpus.add(item["negative"])
            elif self.data_format == "triplets":
                corpus.add(item.get("anchor", ""))
                corpus.add(item.get("positive", ""))
                corpus.add(item.get("negative", ""))
            elif self.data_format == "single":
                corpus.add(item.get("text", ""))
        
        return list(corpus)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.data_format == "pairs":
            return self._get_pair_item(item)
        elif self.data_format == "triplets":
            return self._get_triplet_item(item)
        elif self.data_format == "single":
            return self._get_single_item(item)
        else:
            raise ValueError(f"Unknown data format: {self.data_format}")
    
    def _get_pair_item(self, item):
        """Process pair format data"""
        text1 = item.get("text1", "")
        text2 = item.get("text2", "")
        label = item.get("label", 1)
        
        # Tokenize texts
        encoded1 = self.tokenizer(
            text1,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoded2 = self.tokenizer(
            text2,
            padding="max_length", 
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        result = {
            "input_ids1": encoded1["input_ids"].squeeze(0),
            "attention_mask1": encoded1["attention_mask"].squeeze(0),
            "input_ids2": encoded2["input_ids"].squeeze(0),
            "attention_mask2": encoded2["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }
        
        # Add negative samples if enabled
        if self.negative_sampling and label == 1:
            negatives = self._sample_negatives(text1, text2)
            for i, neg_text in enumerate(negatives):
                encoded_neg = self.tokenizer(
                    neg_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                result[f"input_ids_neg{i}"] = encoded_neg["input_ids"].squeeze(0)
                result[f"attention_mask_neg{i}"] = encoded_neg["attention_mask"].squeeze(0)
        
        return result
    
    def _get_triplet_item(self, item):
        """Process triplet format data"""
        anchor = item.get("anchor", "")
        positive = item.get("positive", "")
        negative = item.get("negative", "")
        
        # If no negative provided, sample one
        if not negative and self.negative_sampling:
            negative = self._sample_negatives(anchor, positive, num_samples=1)[0]
        
        # Tokenize texts
        encoded_anchor = self.tokenizer(
            anchor,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoded_positive = self.tokenizer(
            positive,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoded_negative = self.tokenizer(
            negative,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids_anchor": encoded_anchor["input_ids"].squeeze(0),
            "attention_mask_anchor": encoded_anchor["attention_mask"].squeeze(0),
            "input_ids_positive": encoded_positive["input_ids"].squeeze(0),
            "attention_mask_positive": encoded_positive["attention_mask"].squeeze(0),
            "input_ids_negative": encoded_negative["input_ids"].squeeze(0),
            "attention_mask_negative": encoded_negative["attention_mask"].squeeze(0),
        }
    
    def _get_single_item(self, item):
        """Process single text format"""
        text = item.get("text", "")
        
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "text": text
        }
    
    def _sample_negatives(self, exclude_text1: str, exclude_text2: str, num_samples: Optional[int] = None) -> List[str]:
        """Sample negative examples from corpus"""
        if num_samples is None:
            num_samples = self.num_negatives
        
        # Filter out the positive examples
        available_texts = [t for t in self.text_corpus if t not in [exclude_text1, exclude_text2]]
        
        if len(available_texts) < num_samples:
            # If not enough unique texts, sample with replacement
            return random.choices(available_texts, k=num_samples)
        else:
            return random.sample(available_texts, num_samples)

def load_dataset(
    data_path: str,
    data_format: str = "auto",
    split: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """
    Load dataset from various sources and formats
    
    Args:
        data_path: Path to data file or HuggingFace dataset name
        data_format: Format of the data ('auto', 'jsonl', 'json', 'csv', 'huggingface')
        split: Dataset split for HuggingFace datasets
    """
    
    if data_format == "auto":
        data_format = _detect_format(data_path)
    
    if data_format == "jsonl":
        return _load_jsonl(data_path)
    elif data_format == "json":
        return _load_json(data_path)
    elif data_format == "huggingface":
        return _load_huggingface(data_path, split, **kwargs)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

def _detect_format(data_path: str) -> str:
    """Auto-detect data format from file path"""
    if data_path.endswith(".jsonl"):
        return "jsonl"
    elif data_path.endswith(".json"):
        return "json"
    elif "/" in data_path and not data_path.startswith("/"):
        # Likely a HuggingFace dataset
        return "huggingface"
    else:
        return "jsonl"  # Default

def _load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def _load_json(file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # If it's a dict, try to extract the data
        if "data" in data:
            data = data["data"]
        elif "examples" in data:
            data = data["examples"]
        else:
            # Convert dict to list format
            data = [data]
    
    return data

def _load_huggingface(dataset_name: str, split: Optional[str] = None, **kwargs) -> List[Dict]:
    """Load data from HuggingFace datasets"""
    try:
        dataset = hf_load_dataset(dataset_name, split=split, **kwargs)
        return [dict(item) for item in dataset]
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
        raise

def create_dataloader(
    dataset: EmbeddingDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create DataLoader with appropriate settings"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )

# Sample data creation utilities
def create_sample_data(num_samples: int = 1000, data_format: str = "pairs") -> List[Dict]:
    """Create sample data for testing"""
    
    # Sample sentences for different domains
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information.",
        "Climate change poses significant challenges for our planet.",
        "Artificial intelligence will reshape the future of work.",
        "Renewable energy sources are becoming more cost-effective.",
        "The stock market experienced significant volatility today.",
        "Healthcare innovations are improving patient outcomes.",
        "Remote work has changed traditional office dynamics.",
        "Quantum computing promises revolutionary computational power.",
        "Social media platforms influence modern communication patterns."
    ]
    
    data = []
    
    for i in range(num_samples):
        if data_format == "pairs":
            # Create positive and negative pairs
            idx1, idx2 = random.sample(range(len(sentences)), 2)
            
            # Positive pair (same or related sentences)
            if random.random() < 0.5:
                data.append({
                    "text1": sentences[idx1],
                    "text2": sentences[idx1] + " This is a paraphrase.",
                    "label": 1
                })
            else:
                # Negative pair
                data.append({
                    "text1": sentences[idx1],
                    "text2": sentences[idx2],
                    "label": 0
                })
        
        elif data_format == "triplets":
            idx1, idx2, idx3 = random.sample(range(len(sentences)), 3)
            data.append({
                "anchor": sentences[idx1],
                "positive": sentences[idx1] + " This is similar.",
                "negative": sentences[idx2]
            })
        
        elif data_format == "single":
            idx = random.randint(0, len(sentences) - 1)
            data.append({
                "text": sentences[idx],
                "category": f"category_{idx % 3}"
            })
    
    return data