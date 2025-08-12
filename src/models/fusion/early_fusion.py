import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusion(nn.Module):
    def __init__(self, 
                 input_dims: dict,
                 output_dim: int,
                 dropout: float = 0.3):
        """
        Early fusion module that projects features to same dimension and applies normalization.
        
        Args:
            input_dims: Dictionary with dimension sizes for each modality
                       {'text': dim1, 'image': dim2, 'tabular': dim3}
            output_dim: Dimension of the output fused features
            dropout: Dropout rate
        """
        super().__init__()
        
        # Projection layers to common dimension
        self.text_proj = nn.Linear(input_dims['text'], output_dim)
        self.image_proj = nn.Linear(input_dims['image'], output_dim)
        self.tabular_proj = nn.Linear(input_dims['tabular'], output_dim)
        
        # Layer normalization for each modality
        self.text_norm = nn.LayerNorm(output_dim)
        self.image_norm = nn.LayerNorm(output_dim)
        self.tabular_norm = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.fusion_norm = nn.LayerNorm(output_dim * 3)
        self.output_proj = nn.Linear(output_dim * 3, output_dim)
        
    def forward(self, 
                text_emb: torch.Tensor,
                image_emb: torch.Tensor,
                tabular_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for early fusion.
        
        Args:
            text_emb: Text embeddings [batch_size, text_dim]
            image_emb: Image embeddings [batch_size, image_dim]
            tabular_emb: Tabular embeddings [batch_size, tabular_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Project each modality to same dimension
        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)
        tabular_proj = self.tabular_proj(tabular_emb)
        
        # Apply layer normalization
        text_norm = self.text_norm(text_proj)
        image_norm = self.image_norm(image_proj)
        tabular_norm = self.tabular_norm(tabular_proj)
        
        # Concatenate normalized features
        fused = torch.cat([text_norm, image_norm, tabular_norm], dim=1)
        
        # Apply fusion normalization and dropout
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        
        # Project to final dimension
        output = self.output_proj(fused)
        
        return output