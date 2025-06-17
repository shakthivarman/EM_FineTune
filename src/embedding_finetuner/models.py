import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel(nn.Module):
    """
    Flexible embedding model wrapper supporting multiple architectures
    """
    
    def __init__(
        self,
        model_name: str,
        pooling_mode: str = "mean",
        normalize_embeddings: bool = True,
        use_sentence_transformers: bool = True,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.pooling_mode = pooling_mode
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"
        
        self.device = device
        
        # Initialize model
        if use_sentence_transformers:
            try:
                self.model = SentenceTransformer(model_name, device=device)
                self.tokenizer = self.model.tokenizer
                self.is_sentence_transformer = True
            except Exception as e:
                logger.warning(f"Failed to load as SentenceTransformer: {e}")
                self.is_sentence_transformer = False
                self._init_transformers_model()
        else:
            self.is_sentence_transformer = False
            self._init_transformers_model()
    
    def _init_transformers_model(self):
        """Initialize using transformers library"""
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Add pooling layer if needed
        if self.pooling_mode == "cls":
            self.pooling = self._cls_pooling
        elif self.pooling_mode == "mean":
            self.pooling = self._mean_pooling
        elif self.pooling_mode == "max":
            self.pooling = self._max_pooling
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _max_pooling(self, token_embeddings, attention_mask):
        """Max pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]
    
    def _cls_pooling(self, token_embeddings, attention_mask):
        """CLS token pooling"""
        return token_embeddings[:, 0]
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> torch.Tensor:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.is_sentence_transformer:
            return self.model.encode(
                texts, 
                batch_size=batch_size, 
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings
            )
        else:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._encode_batch(batch_texts)
                embeddings.append(batch_embeddings)
            
            embeddings = torch.cat(embeddings, dim=0)
            
            if self.normalize_embeddings:
                embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings
    
    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = self.pooling(outputs.last_hidden_state, encoded["attention_mask"])
        
        return embeddings
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """Forward pass for training"""
        if self.is_sentence_transformer:
            # For sentence transformers, we need to handle this differently
            outputs = self.model[0].auto_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self.model[1]({"token_embeddings": outputs.last_hidden_state, "attention_mask": attention_mask})
            return embeddings["sentence_embedding"]
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            embeddings = self.pooling(outputs.last_hidden_state, attention_mask)
            
            if self.normalize_embeddings:
                embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.is_sentence_transformer:
            return self.model.get_sentence_embedding_dimension()
        else:
            return self.config.hidden_size
    
    def save_pretrained(self, save_directory: str):
        """Save the model"""
        if self.is_sentence_transformer:
            self.model.save(save_directory)
        else:
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load a pretrained model"""
        return cls(model_path, **kwargs)