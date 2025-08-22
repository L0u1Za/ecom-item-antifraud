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
            config.model.model.text.name,
            cache_dir=config.model.model.cache_dir,
            use_fast=True
        )
        self.max_length = config.preprocessing.text.max_length
        
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
        labels = [b[1] for b in batch]
        batch_dict = {}

        # Process text if present
        text_samples = [b[0]['text'] for b in batch if b[0].get('text') is not None]
        if text_samples:
            titles = [t['title'] for t in text_samples]
            descriptions = [t['description'] for t in text_samples]
            encoded_text = self.tokenizer(
                titles, descriptions, padding=True, truncation=True,
                max_length=self.max_length, return_tensors='pt'
            )
            batch_dict['text'] = {
                'input_ids': encoded_text['input_ids'],
                'attention_mask': encoded_text['attention_mask'],
                'token_type_ids': encoded_text.get('token_type_ids')
            }

        # Process images if present
        image_samples = [b[0]['images'] for b in batch if b[0].get('images') is not None]
        if image_samples:
            images = [item['images'] for item in image_samples]
            batch_dict['images'] = torch.stack(images)

        # Process tabular if present
        tabular_samples = [b[0]['tabular'] for b in batch if b[0].get('tabular') is not None]
        if tabular_samples:
            cat_list = [t['categorical'] for t in tabular_samples]
            cont_list = [t['continuous'] for t in tabular_samples]
            
            categorical = torch.stack(cat_list) if cat_list else torch.zeros((len(batch), 0), dtype=torch.long)
            continuous = torch.stack(cont_list) if cont_list and cont_list[0].numel() > 0 else torch.zeros((len(batch), 0), dtype=torch.float32)

            if getattr(self.config.preprocessing.image, 'compute_clip_similarity', False) and image_samples:
                clip_similarities = [item.get('text_image_similarity', torch.tensor(0.0)) for item in image_samples]
                clip_similarities = torch.stack(clip_similarities).unsqueeze(1)
                continuous = torch.cat([continuous, clip_similarities], dim=1) if continuous.numel() > 0 else clip_similarities
            
            batch_dict['tabular'] = {'categorical': categorical, 'continuous': continuous}

        return batch_dict, torch.stack(labels)

class MultiModalCollatorTest:
    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra configuration object containing model and preprocessing settings
        """
        self.config = config
        
        # Initialize tokenizer from config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model.text.name,
            cache_dir=config.model.model.cache_dir,
            use_fast=True
        )
        self.max_length = config.preprocessing.text.max_length
        
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
        item_ids = [b['item_id'] for b in batch]
        batch_dict = {"item_id": item_ids}

        # Process text if present
        text_samples = [b['text'] for b in batch if b.get('text') is not None]
        if text_samples:
            titles = [t['title'] for t in text_samples]
            descriptions = [t['description'] for t in text_samples]
            encoded_text = self.tokenizer(
                titles, descriptions, padding=True, truncation=True,
                max_length=self.max_length, return_tensors='pt'
            )
            batch_dict['text'] = {
                'input_ids': encoded_text['input_ids'],
                'attention_mask': encoded_text['attention_mask'],
                'token_type_ids': encoded_text.get('token_type_ids')
            }

        # Process images if present
        image_samples = [b['images'] for b in batch if b.get('images') is not None]
        if image_samples:
            images = [item['images'] for item in image_samples]
            batch_dict['images'] = torch.stack(images)

        # Process tabular if present
        tabular_samples = [b['tabular'] for b in batch if b.get('tabular') is not None]
        if tabular_samples:
            cat_list = [t['categorical'] for t in tabular_samples]
            cont_list = [t['continuous'] for t in tabular_samples]
            
            categorical = torch.stack(cat_list) if cat_list else torch.zeros((len(batch), 0), dtype=torch.long)
            continuous = torch.stack(cont_list) if cont_list and cont_list[0].numel() > 0 else torch.zeros((len(batch), 0), dtype=torch.float32)

            if getattr(self.config.preprocessing.image, 'compute_clip_similarity', False) and image_samples:
                clip_similarities = [item.get('text_image_similarity', torch.tensor(0.0)) for item in image_samples]
                clip_similarities = torch.stack(clip_similarities).unsqueeze(1)
                continuous = torch.cat([continuous, clip_similarities], dim=1) if continuous.numel() > 0 else clip_similarities
            
            batch_dict['tabular'] = {'categorical': categorical, 'continuous': continuous}

        return batch_dict