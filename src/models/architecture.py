import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from transformers import AutoModel
import timm
import hydra
from omegaconf import DictConfig
from tab_transformer_pytorch import FTTransformer

class TextTower(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.model.text.name)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, cfg.model.text.projection_dim)
        # Layer normalization for each modality
        self.text_norm = nn.LayerNorm(cfg.model.text.projection_dim)

        self.dropout = nn.Dropout(cfg.model.text.dropout)
        if cfg.model.text.get("unfreeze", False):
            self.unfreeze()
        else:
            self.freeze()
        
    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_outputs = self.bert(**text_inputs)
        text_norm = self.text_norm(self.text_proj(text_outputs.pooler_output))
        text_emb = self.dropout(text_norm)
        return text_emb
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.bert.parameters():
            param.requires_grad = True

class ImageTower(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model.image.name,
            pretrained=cfg.model.image.pretrained,
            num_classes=0
        )
        self.proj = nn.Linear(self.backbone.num_features, cfg.model.image.projection_dim)
        self.image_norm = nn.LayerNorm(cfg.model.image.projection_dim)
        self.dropout = nn.Dropout(cfg.model.image.dropout)

        if cfg.model.image.get("unfreeze", False):
            self.unfreeze()
        else:
            self.freeze()
        
    def forward(self, image_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_inputs: (B, C, H, W) tensor of images
        Returns:
            (B, proj_dim) tensor of image embeddings
        """
        features = self.backbone(image_inputs)
        projected = self.proj(features)
        normalized = self.image_norm(projected)
        img_emb = self.dropout(normalized)
        return img_emb
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True

class TabularTower(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        tab_cfg = cfg.model.tabular
        # Expect categories (list of int cardinalities) and num_continuous
        self.model = FTTransformer(
            categories=tuple(tab_cfg.categories),
            num_continuous=int(tab_cfg.num_continuous),
            dim=int(tab_cfg.dim),
            dim_out=int(tab_cfg.projection_dim),
            depth=int(tab_cfg.depth),
            heads=int(tab_cfg.heads),
            attn_dropout=float(tab_cfg.attn_dropout),
            ff_dropout=float(tab_cfg.ff_dropout)
        )

    def forward(self, tabular_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # model expects (x_categ, x_cont)
        x_categ = tabular_inputs['categorical']  # [B, num_cats] long
        x_cont = tabular_inputs['continuous']    # [B, num_cont]
        return self.model(x_categ, x_cont)

class FraudDetectionModel(nn.Module):
    def __init__(self, cfg: DictConfig, training=True):
        super().__init__()
        
        fusion_emb_size = 0
        
        if cfg.model.text.enabled:
            self.text_tower = TextTower(cfg)
            fusion_emb_size += cfg.model.text.projection_dim
        if cfg.model.image.enabled:
            self.image_tower = ImageTower(cfg)
            fusion_emb_size += cfg.model.image.projection_dim
        if cfg.model.tabular.enabled:
            self.tabular_tower = TabularTower(cfg)
            fusion_emb_size += cfg.model.tabular.projection_dim
        
        # Load fusion module using Hydra
        self.fusion = hydra.utils.instantiate(cfg.fusion, input_dim=fusion_emb_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.fusion.output_dim),
            nn.Linear(cfg.fusion.output_dim, cfg.model.classifier.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.model.classifier.dropout),
            nn.Linear(cfg.model.classifier.hidden_dim, cfg.model.classifier.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(cfg.model.classifier.dropout),
            nn.Linear(cfg.model.classifier.hidden_dim2, 1)
        )
        
        # Modality dropout for training robustness
        self.modality_dropout = cfg.model.get('modality_dropout', 0.0)
        self.training = training

    def _apply_modality_dropout(self, embeds: list, batch_size: int) -> list:
        """
        Apply modality dropout during training to randomly disable enabled modalities.
        Only applies to modalities that are already enabled in the config.
        During training, randomly masks some enabled modality embeddings to zero.
        
        Args:
            embeds: List of modality embeddings (only for enabled modalities)
            batch_size: Batch size for creating dropout masks
            
        Returns:
            List of embeddings with some potentially zeroed out during training
        """
        if not self.training or self.modality_dropout == 0.0:
            return embeds
            
        # Create dropout mask for each enabled modality
        dropout_masks = []
        for embed in embeds:
            # Create mask: 1 means keep, 0 means drop
            # Single mask per modality: either all zeros or all ones
            mask = torch.bernoulli(torch.tensor([1 - self.modality_dropout]))
            mask = mask.to(embed.device)
            # Expand mask to match embedding dimensions: [1] -> [batch_size, 1] -> [batch_size, embed_dim]
            mask = mask.expand(batch_size, 1).expand_as(embed)
            dropout_masks.append(mask)
        
        # Apply masks to enabled modalities
        dropped_embeds = []
        for embed, mask in zip(embeds, dropout_masks):
            if embed is not None and mask is not None:
                dropped_embeds.append(embed * mask)  # Apply dropout mask
            else:
                dropped_embeds.append(embed)
                
        return dropped_embeds

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict]:
        embeds = []
        batch_size = None
        
        # Collect embeddings from enabled modalities only
        if hasattr(self, 'text_tower'):
            text_emb = self.text_tower(batch['text'])
            embeds.append(text_emb)
            batch_size = text_emb.shape[0]
        if hasattr(self, 'image_tower'):
            img_emb = self.image_tower(batch['images'])
            embeds.append(img_emb)
            batch_size = img_emb.shape[0]
        if hasattr(self, 'tabular_tower'):
            tab_emb = self.tabular_tower(batch['tabular'])
            embeds.append(tab_emb)
            batch_size = tab_emb.shape[0]

        # Apply modality dropout during training only - randomly mask enabled modalities
        embeds = self._apply_modality_dropout(embeds, batch_size)
        # All embeddings should now be valid tensors for enabled modalities
        fused_features = self.fusion(embeds)
        
        logits = self.classifier(fused_features)
        
        return logits, {
            'embeds': embeds,
            'fused_features': fused_features
        }