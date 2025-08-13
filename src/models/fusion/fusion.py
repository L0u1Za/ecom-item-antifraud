import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, 
                 input_dims: dict,
                 output_dim: int,
                 dropout: float = 0.3):
        """
        Fusion module that projects features to same dimension and applies normalization.
        
        Args:
            input_dims: Dictionary with dimension sizes for each modality
                       {'text': dim1, 'image': dim2, 'tabular': dim3}
            output_dim: Dimension of the output fused features
            dropout: Dropout rate
        """
        super().__init__()
        input_dim = input_dims['text'] + input_dims['image'] + input_dims['tabular']
        self.dropout = nn.Dropout(dropout)
        self.fusion_norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, 
                text_emb: torch.Tensor,
                image_emb: torch.Tensor,
                tabular_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Fusion.
        
        Args:
            text_emb: Text embeddings [batch_size, text_dim]
            image_emb: Image embeddings [batch_size, image_dim]
            tabular_emb: Tabular embeddings [batch_size, tabular_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        
        # Concatenate normalized features
        fused = torch.cat([text_emb, image_emb, tabular_emb], dim=1)
        
        # Apply fusion normalization and dropout
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        
        # Project to final dimension
        output = self.output_proj(fused)
        
        return output