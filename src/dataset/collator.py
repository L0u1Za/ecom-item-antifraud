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
        
        # Get fraud indicators config
        self.use_fraud_indicators = config.preprocessing.text.get('add_fraud_indicators', False)
        self.fraud_indicator_dim = config.preprocessing.text.get('fraud_indicator_dim', 20)
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
        text_samples = [b[0]['text'] for b in batch]
        image_samples = [b[0]['images'] for b in batch]
        tabular_samples = [b[0]['tabular'] for b in batch]
        labels = [b[1] for b in batch]
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
        
        max_num_images = max(len(item['images']) for item in image_samples)
        image_shape = self.config.preprocessing.image.size

        padded_images = []
        if max_num_images == 0:
            # Ensure at least one placeholder image per item to avoid empty stack
            images = torch.stack([
                torch.zeros((1, 3, image_shape[0], image_shape[1]))
                for _ in image_samples
            ])
        else:
            for item in image_samples:
                imgs = item['images']
                if len(imgs) < max_num_images:
                    pad_count = max_num_images - len(imgs)
                    imgs = imgs + [torch.zeros((3, image_shape[0], image_shape[1]))] * pad_count
                elif len(imgs) > max_num_images:
                    imgs = imgs[:max_num_images]
                padded_images.append(torch.stack(imgs))
            images = torch.stack(padded_images)

        # Process tabular features to FTTransformer-compatible tensors
        cat_list = []
        cont_list = []
        for t in tabular_samples:
            cat_list.append(t['categorical'])
            cont_list.append(t['continuous'])

        categorical = torch.stack(cat_list) if len(cat_list) > 0 else torch.zeros((len(batch), 0), dtype=torch.long)
        # Handle possible zero-length continuous features
        if cont_list and cont_list[0].numel() > 0:
            continuous = torch.stack(cont_list)
        else:
            continuous = torch.zeros((len(batch), 0), dtype=torch.float32)
        
        # Handle fraud indicators if present
        if self.use_fraud_indicators:
            fraud_indicators = []
            for t in text_samples:
                if 'fraud_indicators' in t:
                    indicators = torch.zeros(self.fraud_indicator_dim)
                    for i, (key, value) in enumerate(t['fraud_indicators'].items()):
                        if i < self.fraud_indicator_dim:
                            indicators[i] = float(value)
                    fraud_indicators.append(indicators)
                else:
                    fraud_indicators.append(torch.zeros(self.fraud_indicator_dim))

            fraud_tensor = torch.stack(fraud_indicators)
            # Append additional boolean indicators to continuous features
            continuous = torch.cat([continuous, fraud_tensor], dim=1) if continuous.numel() > 0 else fraud_tensor
        
        if getattr(self.config.preprocessing.image, 'compute_clip_similarity', False):
            clip_similarities = [item.get('text_image_similarity', torch.tensor(0.0)) for item in image_samples]
            clip_similarities = torch.stack(clip_similarities).unsqueeze(1)
            continuous = torch.cat([continuous, clip_similarities], dim=1) if continuous.numel() > 0 else clip_similarities
        
        # Construct final batch
        batch_dict = {
            'text': {
                'input_ids': encoded_text['input_ids'],
                'attention_mask': encoded_text['attention_mask'],
                'token_type_ids': encoded_text.get('token_type_ids')
            },
            'images': images,
            'tabular': {
                'categorical': categorical,
                'continuous': continuous
            }
        }

        return batch_dict, torch.stack(labels)