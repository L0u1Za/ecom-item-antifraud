import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class Fusion(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.3,
                 type="early"):
        """
        Fusion module that projects features to same dimension and applies normalization.
        
        Args:
            input_dim: Total dimension size after concatenating all modalities
            output_dim: Dimension of the output fused features
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fusion_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, 
                embeds: List[torch.Tensor]
                ) -> torch.Tensor:
        """
        Forward pass for Fusion.
        
        Args:
            embeds: List of modality embeddings, each of shape [batch_size, modality_dim]
                   Order depends on which modalities are enabled in configuration
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        if not embeds:
            raise ValueError("No embeddings provided to fusion module")
        
        # Validate all embeddings have the same batch size
        batch_sizes = [emb.shape[0] for emb in embeds]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"All embeddings must have the same batch size. Got: {batch_sizes}")
        
        # Concatenate all modality embeddings
        fused = torch.cat(embeds, dim=1)
        
        # Apply fusion normalization and dropout
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        
        # Project to final dimension
        output = self.output_proj(fused)
        
        return output

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, output_dim, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeds: List[torch.Tensor]):
        # embeds: list of [B, D] tensors
        x = torch.stack(embeds, dim=1)  # [B, M, D]
        x = self.norm(x)
        attn_weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)  # [B, M]
        fused = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        fused = self.dropout(fused)
        return self.proj(fused)