import torch
from transformers import AutoTokenizer
from torchvision import transforms
import numpy as np
from typing import Dict, Any, List

from preprocessing.text.cleaner import TextCleaner
from preprocessing.text.normalizer import normalize_text
from preprocessing.text.business_rules import BusinessRulesChecker

from preprocessing.image import (
    get_image_augmentations,
    compute_clip_embeddings,
    compute_image_text_similarity
)
from preprocessing.tabular import (
    encode_metadata,
    scale_numerical_features
)

class TextProcessor:
    def __init__(self,
                 model_name='bert-base-uncased',
                 max_length=512,
                 apply_cleaning=True,
                 apply_lemmatization=True,
                 add_fraud_indicators=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.apply_cleaning = apply_cleaning
        self.apply_lemmatization = apply_lemmatization
        self.add_fraud_indicators = add_fraud_indicators
        self.cleaner = TextCleaner()

    def preprocess_text(self, text: str) -> str:
        if self.apply_cleaning:
            text = self.cleaner.clean_text(text)
        if self.apply_lemmatization:
            text = normalize_text(text)

        return text
        
    def __call__(self, text_data: Dict[str, str]) -> Dict[str, torch.Tensor]:
        # Process title and description
        title = text_data['title']
        description = text_data['description']

        # Get suspicious patterns first
        if self.add_fraud_indicators:
            checker = BusinessRulesChecker()
            fraud_indicators = checker(title, description)

        # Process title and description
        title = self.preprocess_text(title)
        description = self.preprocess_text(description)

        # Tokenize
        title_tokens = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        desc_tokens = self.tokenizer(
            description,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        obj = {
            'title': {k: v.squeeze(0) for k, v in title_tokens.items()},
            'description': {k: v.squeeze(0) for k, v in desc_tokens.items()},
        }
        if self.add_fraud_indicators:
            obj['fraud_indicators'] = fraud_indicators
        return obj

class ImageProcessor:
    def __init__(self, 
                 image_size=224,
                 clip_model_name="openai/clip-vit-base-patch32",
                 cache_dir=None):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.augmentations = get_image_augmentations()
        self.clip_model_name = clip_model_name
        self.cache_dir = cache_dir
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        image = data['image']
        text = data.get('title', '')  # Get title for CLIP similarity
        
        # Basic transform
        img_tensor = self.transform(image)
        
        # Get augmented versions
        aug_tensors = [aug(image) for aug in self.augmentations]
        
        # Compute CLIP embeddings and similarity
        clip_emb = compute_clip_embeddings(
            image, 
            model_name=self.clip_model_name,
            cache_dir=self.cache_dir
        )
        
        text_img_similarity = compute_image_text_similarity(
            image, 
            text, 
            model_name=self.clip_model_name
        )
        
        return {
            'image': img_tensor,
            'augmented_images': torch.stack(aug_tensors),
            'clip_embedding': clip_emb,
            'text_image_similarity': text_img_similarity
        }

class TabularProcessor:
    def __init__(self, 
                 categorical_cols: List[str] = None, 
                 numerical_cols: List[str] = None):
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.cat_encoders = {}
        self.num_scaler = None
        
    def fit(self, data):
        # Encode categorical features
        self.cat_encoders = encode_metadata(
            data, 
            self.categorical_cols
        )
        
        # Scale numerical features
        self.num_scaler = scale_numerical_features(
            data, 
            self.numerical_cols
        )
    
    def __call__(self, row) -> Dict[str, torch.Tensor]:
        processed = []
        
        # Process categorical
        for col in self.categorical_cols:
            encoded = torch.zeros(len(self.cat_encoders[col]))
            idx = self.cat_encoders[col].get(row[col], -1)
            if idx != -1:
                encoded[idx] = 1
            processed.append(encoded)
            
        # Process numerical
        for col in self.numerical_cols:
            val = (row[col] - self.num_scaler[col]['mean']) / self.num_scaler[col]['std']
            processed.append(torch.tensor([val], dtype=torch.float32))
            
        # Combine all features
        combined = torch.cat(processed)
        
        return {
            'tabular_features': combined,
            'categorical_features': torch.cat(processed[:len(self.categorical_cols)]),
            'numerical_features': torch.cat(processed[len(self.categorical_cols):])
        }