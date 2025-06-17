import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any
import logging
import json
from pathlib import Path
import numpy as np

from .models import EmbeddingModel
from .losses import get_loss_function
from .data import EmbeddingDataset, create_dataloader
from .utils import set_seed, get_device_info, save_config

logger = logging.getLogger(__name__)

class EmbeddingFinetuner:
    """
    Main trainer class for embedding model finetuning
    """
    
    def __init__(
        self,
        model_name: str,
        loss_function: str = "MultipleNegativesRankingLoss",
        loss_kwargs: Optional[Dict] = None,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        mixed_precision: str = "fp16",
        gradient_checkpointing: bool = True,
        max_length: int = 512,
        pooling_mode: str = "mean",
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize the embedding finetuner
        
        Args:
            model_name: Name or path of the base model
            loss_function: Name of the loss function to use
            loss_kwargs: Additional arguments for loss function
            use_lora: Whether to use LoRA for parameter-efficient finetuning
            lora_config: Configuration for LoRA
            mixed_precision: Mixed precision training ("no", "fp16", "bf16")
            gradient_checkpointing: Enable gradient checkpointing to save memory
            max_length: Maximum sequence length
            pooling_mode: Pooling strategy ("mean", "cls", "max")
            normalize_embeddings: Whether to normalize embeddings
            device: Device to use (auto-detected if None)
            seed: Random seed for reproducibility
        """
        
        set_seed(seed)
        
        # Initialize accelerator for distributed training and mixed precision
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=1,
            log_with="wandb" if wandb.run else None
        )
        
        self.device = self.accelerator.device
        self.mixed_precision = mixed_precision
        self.use_lora = use_lora
        
        # Log device information
        device_info = get_device_info()
        logger.info(f"Using device: {device_info}")
        
        # Initialize model
        self.model = EmbeddingModel(
            model_name=model_name,
            pooling_mode=pooling_mode,
            normalize_embeddings=normalize_embeddings,
            max_length=max_length,
            device=str(self.device)
        )
        
        # Apply LoRA if enabled
        if use_lora:
            self._setup_lora(lora_config or {})
        
        # Enable gradient checkpointing
        if gradient_checkpointing and hasattr(self.model.model, 'gradient_checkpointing_enable'):
            self.model.model.gradient_checkpointing_enable()
        
        # Initialize loss function
        loss_kwargs = loss_kwargs or {}
        self.loss_fn = get_loss_function(loss_function, **loss_kwargs)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = float('-inf')
        
        # Store configuration
        self.config = {
            "model_name": model_name,
            "loss_function": loss_function,
            "loss_kwargs": loss_kwargs,
            "use_lora": use_lora,
            "lora_config": lora_config,
            "mixed_precision": mixed_precision,
            "gradient_checkpointing": gradient_checkpointing,
            "max_length": max_length,
            "pooling_mode": pooling_mode,
            "normalize_embeddings": normalize_embeddings,
            "seed": seed
        }
    
    def _setup_lora(self, lora_config: Dict):
        """Setup LoRA configuration"""
        default_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["query", "key", "value", "dense"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.FEATURE_EXTRACTION
        }
        default_lora_config.update(lora_config)
        
        peft_config = LoraConfig(**default_lora_config)
        
        if self.model.is_sentence_transformer:
            # For sentence transformers, apply LoRA to the transformer model
            self.model.model[0].auto_model = get_peft_model(
                self.model.model[0].auto_model, 
                peft_config
            )
        else:
            self.model.model = get_peft_model(self.model.model, peft_config)
        
        logger.info(f"Applied LoRA with config: {default_lora_config}")
    
    def prepare_data(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        data_format: str = "pairs",
        batch_size: int = 32,
        negative_sampling: bool = True,
        num_negatives: int = 1,
        val_split: float = 0.1
    ) -> tuple:
        """
        Prepare training and validation datasets
        """
        
        # Split training data if no validation data provided
        if val_data is None and val_split > 0:
            split_idx = int(len(train_data) * (1 - val_split))
            val_data = train_data[split_idx:]
            train_data = train_data[:split_idx]
        
        # Create datasets
        train_dataset = EmbeddingDataset(
            data=train_data,
            tokenizer=self.model.tokenizer,
            max_length=self.model.max_length,
            data_format=data_format,
            negative_sampling=negative_sampling,
            num_negatives=num_negatives
        )
        
        val_dataset = None
        if val_data:
            val_dataset = EmbeddingDataset(
                data=val_data,
                tokenizer=self.model.tokenizer,
                max_length=self.model.max_length,
                data_format=data_format,
                negative_sampling=False  # No negative sampling for validation
            )
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        val_loader = None
        if val_dataset:
            val_loader = create_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4 if torch.cuda.is_available() else 0
            )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        output_dir: str = "./output",
        data_format: str = "pairs",
        negative_sampling: bool = True,
        num_negatives: int = 1,
        val_split: float = 0.1,
        resume_from_checkpoint: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "embedding-finetuning",
        **kwargs
    ):
        """
        Train the embedding model
        """
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if use_wandb and not wandb.run:
            wandb.init(
                project=wandb_project,
                config=self.config,
                name=f"embedding-ft-{self.config['model_name'].split('/')[-1]}"
            )
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            train_data=train_data,
            val_data=val_data,
            data_format=data_format,
            batch_size=batch_size,
            negative_sampling=negative_sampling,
            num_negatives=num_negatives,
            val_split=val_split
        )
        
        # Setup optimizer and scheduler
        total_steps = len(train_loader) * epochs
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare for distributed training
        self.model, self.optimizer, train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, self.scheduler
        )
        
        if val_loader:
            val_loader = self.accelerator.prepare(val_loader)
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        
        self.model.train()
        
        for epoch in range(self.current_epoch, epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                # Forward pass
                loss = self._training_step(batch, data_format)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Logging
                if self.global_step % logging_steps == 0:
                    metrics = {
                        "train/loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(train_loader),
                        "train/global_step": self.global_step
                    }
                    
                    if use_wandb:
                        wandb.log(metrics)
                    
                    if self.accelerator.is_local_main_process:
                        logger.info(f"Step {self.global_step}: {metrics}")
                
                # Evaluation
                if val_loader and self.global_step % eval_steps == 0:
                    eval_metrics = self._evaluate(val_loader, data_format)
                    
                    if use_wandb:
                        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                    
                    if self.accelerator.is_local_main_process:
                        logger.info(f"Evaluation at step {self.global_step}: {eval_metrics}")
                    
                    # Save best model
                    if eval_metrics.get("loss", float('inf')) < self.best_score:
                        self.best_score = eval_metrics["loss"]
                        self.save_model(output_dir / "best_model")
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")
            
            # End of epoch
            self.current_epoch = epoch + 1
            avg_epoch_loss = epoch_loss / len(train_loader)
            
            if self.accelerator.is_local_main_process:
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_model(output_dir / "final_model")
        
        if use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor], data_format: str) -> torch.Tensor:
        """Execute a single training step"""
        
        if data_format == "pairs":
            # Extract embeddings for pairs
            embeddings1 = self.model(
                input_ids=batch["input_ids1"],
                attention_mask=batch["attention_mask1"]
            )
            embeddings2 = self.model(
                input_ids=batch["input_ids2"], 
                attention_mask=batch["attention_mask2"]
            )
            
            # Compute loss based on loss function type - fixed compatibility check
            loss_name = self.loss_fn.__class__.__name__
            if loss_name in ['MultipleNegativesRankingLoss', 'InfoNCELoss', 'AnglELoss']:
                loss = self.loss_fn(embeddings1, embeddings2)
            else:
                # For losses that need labels
                labels = batch.get("labels", torch.ones(embeddings1.size(0), device=self.device))
                loss = self.loss_fn(embeddings1, embeddings2, labels)
        
        elif data_format == "triplets":
            # Extract embeddings for triplets
            anchor_emb = self.model(
                input_ids=batch["input_ids_anchor"],
                attention_mask=batch["attention_mask_anchor"]
            )
            positive_emb = self.model(
                input_ids=batch["input_ids_positive"],
                attention_mask=batch["attention_mask_positive"] 
            )
            negative_emb = self.model(
                input_ids=batch["input_ids_negative"],
                attention_mask=batch["attention_mask_negative"]
            )
            
            loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
        
        else:
            raise ValueError(f"Unsupported data format for training: {data_format}")
        
        return loss
    
    def _evaluate(self, val_loader: DataLoader, data_format: str) -> Dict[str, float]:
        """Evaluate the model on validation data"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._training_step(batch, data_format)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches
        }
    
    def save_model(self, save_path: Union[str, Path]):
        """Save the finetuned model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Unwrap model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model
        unwrapped_model.save_pretrained(str(save_path))
        
        # Save configuration
        save_config(self.config, save_path / "training_config.json")
        
        logger.info(f"Model saved to {save_path}")
    
    def _save_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Save training checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save using accelerator
        self.accelerator.save_state(str(checkpoint_path))
        
        # Save additional training state
        training_state = {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "config": self.config
        }
        
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load accelerator state
        self.accelerator.load_state(str(checkpoint_path))
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            
            self.current_epoch = training_state.get("current_epoch", 0)
            self.global_step = training_state.get("global_step", 0)
            self.best_score = training_state.get("best_score", float('-inf'))
        
        logger.info(f"Resumed training from {checkpoint_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load a finetuned model"""
        config_path = Path(model_path) / "training_config.json"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Override config with any provided kwargs
            config.update(kwargs)
            
            # Initialize trainer with saved config
            trainer = cls(model_name=model_path, **config)
            
            return trainer
        else:
            # Initialize with default config
            return cls(model_name=model_path, **kwargs)