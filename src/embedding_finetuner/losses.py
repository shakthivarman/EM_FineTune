import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss as described in:
    https://arxiv.org/abs/1705.00652
    """
    
    def __init__(self, scale: float = 20.0, similarity_fct=F.cosine_similarity):
        super().__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_a: Anchor embeddings [batch_size, embedding_dim]
            embeddings_b: Positive embeddings [batch_size, embedding_dim]
        """
        batch_size = embeddings_a.size(0)
        
        # Compute similarity matrix - fixed for proper broadcasting
        if len(embeddings_a.shape) == 2 and len(embeddings_b.shape) == 2:
            # Standard cosine similarity matrix computation
            embeddings_a_norm = F.normalize(embeddings_a, p=2, dim=1)
            embeddings_b_norm = F.normalize(embeddings_b, p=2, dim=1)
            scores = torch.matmul(embeddings_a_norm, embeddings_b_norm.T) * self.scale
        else:
            # Fallback to pairwise similarity
            scores = self.similarity_fct(embeddings_a.unsqueeze(1), embeddings_b.unsqueeze(0)) * self.scale
        
        # Labels are diagonal (each anchor matches with corresponding positive)
        labels = torch.arange(batch_size, device=embeddings_a.device)
        
        return self.cross_entropy_loss(scores, labels)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for siamese networks
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_a: First set of embeddings [batch_size, embedding_dim]
            embeddings_b: Second set of embeddings [batch_size, embedding_dim]
            labels: Binary labels (1 for similar, 0 for dissimilar) [batch_size]
        """
        if self.distance_metric == "euclidean":
            distances = F.pairwise_distance(embeddings_a, embeddings_b)
        elif self.distance_metric == "cosine":
            distances = 1 - F.cosine_similarity(embeddings_a, embeddings_b)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Contrastive loss
        loss_positive = labels * torch.pow(distances, 2)
        loss_negative = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        return torch.mean(loss_positive + loss_negative)

class TripletLoss(nn.Module):
    """
    Triplet loss with hard negative mining
    """
    
    def __init__(self, margin: float = 0.5, distance_metric: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
        """
        if self.distance_metric == "euclidean":
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
        elif self.distance_metric == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(loss)

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_a: Query embeddings [batch_size, embedding_dim]
            embeddings_b: Key embeddings [batch_size, embedding_dim]
        """
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, dim=-1)
        embeddings_b = F.normalize(embeddings_b, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature
        
        # Labels are diagonal
        labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)
        
        # Symmetric loss
        loss_a = self.cross_entropy(logits, labels)
        loss_b = self.cross_entropy(logits.T, labels)
        
        return (loss_a + loss_b) / 2

class AnglELoss(nn.Module):
    """
    AnglE loss for text embeddings
    https://arxiv.org/abs/2309.12871
    """
    
    def __init__(self, w1: float = 1.0, w2: float = 1.0, w3: float = 1.0, 
                 angle_tau: float = 1.0, cosine_tau: float = 20.0):
        super().__init__()
        self.w1 = w1
        self.w2 = w2  
        self.w3 = w3
        self.angle_tau = angle_tau
        self.cosine_tau = cosine_tau
    
    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, dim=-1)
        embeddings_b = F.normalize(embeddings_b, dim=-1)
        
        batch_size = embeddings_a.size(0)
        
        # Compute angles and cosines
        cosine_sim = torch.sum(embeddings_a * embeddings_b, dim=-1)
        angles = torch.acos(torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7))
        
        # Positive pairs (diagonal)
        pos_angles = torch.diag(angles)
        pos_cosines = torch.diag(cosine_sim)
        
        # Negative pairs (off-diagonal)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings_a.device)
        neg_angles = angles[mask]
        neg_cosines = cosine_sim[mask]
        
        # AnglE loss components
        angle_loss = -torch.log(torch.exp(-self.angle_tau * pos_angles).mean())
        cosine_loss = -torch.log(torch.exp(self.cosine_tau * pos_cosines).mean())
        
        # Additional regularization
        reg_loss = torch.mean(torch.abs(pos_angles - math.pi/4))
        
        return self.w1 * angle_loss + self.w2 * cosine_loss + self.w3 * reg_loss

def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name
    """
    loss_functions = {
        "MultipleNegativesRankingLoss": MultipleNegativesRankingLoss,
        "ContrastiveLoss": ContrastiveLoss,
        "TripletLoss": TripletLoss,
        "InfoNCELoss": InfoNCELoss,
        "AnglELoss": AnglELoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)