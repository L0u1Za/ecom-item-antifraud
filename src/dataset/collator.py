from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig
from transformers import AutoTokenizer
import hydra

class MultiModalCollator:
    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra configuration object containing model and preprocessing settings
        """
        self.config = config
        
        # Initialize tokenizer from config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.text.name,
            cache_dir=config.model.cache_dir,
            use_fast=True
        )
        self.max_length = config.preprocessing.text.max_length
        
        # Get fraud indicators config
        self.use_fraud_indicators = config.preprocessing.text.get('add_fraud_indicators', False)
        self.fraud_indicator_dim = config.model.get('fraud_indicator_dim', 10)
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for multimodal batches
        
        Args:
            batch: List of dictionaries containing:
                - text: Dict with 'title' and 'description'
                - image: Dict with 'image' tensor
                - tabular: Dict with 'features' tensor
                
        Returns:
            Dictionary with batched and padded tensors
        """
        # Separate modalities
        text_samples = [b['text'] for b in batch]
        image_samples = [b['image'] for b in batch]
        tabular_samples = [b['tabular'] for b in batch]
        
        # Process text
        titles = [t['title'] for t in text_samples]
        descriptions = [t['description'] for t in text_samples]
        
        # Tokenize text (handles padding automatically)
        encoded_text = self.tokenizer(
            titles,
            descriptions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process images (already tensors from ImageProcessor)
        images = torch.stack([img['image'] for img in image_samples])
            
        # Process tabular features (already tensors)
        tabular = torch.stack([t['features'] for t in tabular_samples])
        
        # Handle fraud indicators if present
        if self.use_fraud_indicators:
            fraud_indicators = []
            for t in text_samples:
                if 'fraud_indicators' in t:
                    # Convert dict of boolean indicators to fixed-size tensor
                    indicators = torch.zeros(self.fraud_indicator_dim)
                    for i, (key, value) in enumerate(t['fraud_indicators'].items()):
                        if i < self.fraud_indicator_dim:
                            indicators[i] = float(value)
                    fraud_indicators.append(indicators)
            
            if fraud_indicators:
                fraud_tensor = torch.stack(fraud_indicators)
                # Optionally concatenate with tabular features
                tabular = torch.cat([tabular, fraud_tensor], dim=1)
                
        # Construct final batch
        batch_dict = {
            'text': {
                'input_ids': encoded_text['input_ids'],
                'attention_mask': encoded_text['attention_mask'],
                'token_type_ids': encoded_text.get('token_type_ids')
            },
            'image': {
                'pixels': images
            },
            'tabular': tabular
        }

        return batch_dict