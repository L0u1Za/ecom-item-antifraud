import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from transformers import AutoModel
import timm
import hydra
from omegaconf import DictConfig

class TextTower(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.model.text.name)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, cfg.model.text.projection_dim)
        # Layer normalization for each modality
        self.text_norm = nn.LayerNorm(cfg.model.text.projection_dim)

        self.dropout = nn.Dropout(cfg.model.text.dropout)
        
    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_outputs = self.bert(**text_inputs['text'])
        text_norm = self.text_norm(self.text_proj(text_outputs.pooler_output))
        text_emb = self.dropout(text_norm)
        return text_emb
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
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
        # Layer normalization for each modality
        self.image_norm = nn.LayerNorm(cfg.model.image.projection_dim)
        self.dropout = nn.Dropout(cfg.model.image.dropout)
        
    def forward(self, image_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        img_features = self.backbone(image_inputs['image'])
        img_norm = self.image_norm(self.proj(img_features))
        img_emb = self.dropout(img_norm)
        return img_emb
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

class TabularTower(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.model.tabular.input_dim, cfg.model.tabular.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.model.tabular.dropout),
            nn.Linear(cfg.model.tabular.hidden_dim, cfg.model.tabular.projection_dim),
            nn.LayerNorm(cfg.model.tabular.projection_dim)
        )
    
    def forward(self, tabular_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(tabular_inputs['tabular_features'])
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

@hydra.main(config_path="../../configs", config_name="config")
class FraudDetectionModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.text_tower = TextTower(cfg)
        self.image_tower = ImageTower(cfg)
        self.tabular_tower = TabularTower(cfg)
        
        # Load fusion module using Hydra
        self.fusion = hydra.utils.instantiate(cfg.model.fusion)
        if cfg.model.fusion.type == "early":
            self.text_tower.freeze()
            self.image_tower.freeze()
            self.tabular_tower.freeze()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.model.fusion.output_dim, cfg.model.classifier.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.model.classifier.dropout),
            nn.Linear(cfg.model.classifier.hidden_dim, 1)
        )

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict]:
        text_emb = self.text_tower(batch['text'])
        image_emb = self.image_tower(batch['image'])
        tabular_emb = self.tabular_tower(batch['tabular'])
        
        fused_features = self.fusion(
            text_emb=text_emb,
            image_emb=image_emb,
            tabular_emb=tabular_emb
        )
        
        logits = self.classifier(fused_features)
        
        return logits, {
            'text_emb': text_emb,
            'image_emb': image_emb,
            'tabular_emb': tabular_emb,
            'fused_features': fused_features
        }

@hydra.main(config_path="../../configs", config_name="config")
def create_model(cfg: DictConfig) -> FraudDetectionModel:
    return FraudDetectionModel(cfg)