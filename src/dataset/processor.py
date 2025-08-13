import torch
from torchvision import transforms
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig

from preprocessing.text.cleaner import TextCleaner
from preprocessing.text.normalizer import normalize_text
from preprocessing.text.business_rules import BusinessRulesChecker
from preprocessing.text import TextVectorizer

from preprocessing.image.clip_validator import CLIPValidator

from preprocessing.tabular import (
    encode_metadata,
    scale_numerical_features
)

class TextProcessor:
    def __init__(self,
                 max_length=512,
                 apply_cleaning=True,
                 apply_lemmatization=True,
                 add_fraud_indicators=True):
        self.max_length = max_length
        self.apply_cleaning = apply_cleaning
        self.apply_lemmatization = apply_lemmatization
        self.add_fraud_indicators = add_fraud_indicators
        self.cleaner = TextCleaner()

    def preprocess_text(self, text: str) -> str:
        if self.apply_cleaning:
            text = self.cleaner.clean_text(text)
            text = self.cleaner.clean_repeating_chars(text)
            text = self.cleaner.truncate_text(text, self.max_length)

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

        obj = {
            'title': title,
            'description': description
        }
        if self.add_fraud_indicators:
            obj['fraud_indicators'] = fraud_indicators

        return obj

class ImageProcessor:
    def __init__(self, 
                 config: DictConfig,
                 compute_clip_similarity: bool = False,
                 clip_model = None,
                 training=True):
        """
        Args:
            config: Configuration containing image preprocessing settings
            compute_clip_similarity: Whether to compute CLIP similarity
            clip_model: CLIP model instance if similarity is enabled
        """
        self.config = config
        self.training = training
        size = tuple(config.preprocessing.image.size)
        
        # Basic transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create augmentation pipeline from config
        self.augmentations = []
        if hasattr(config.preprocessing.image, 'augmentations'):
            for aug_config in config.preprocessing.image.augmentations:
                # Each augmentation should be fully defined in config
                aug_transform = hydra.utils.instantiate(
                    aug_config.transform,
                    _convert_="partial"
                )
                self.augmentations.append((
                    transforms.Compose([
                        transforms.Resize(size),
                        aug_transform,
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                    ]),
                    aug_config.probability
                ))

        if compute_clip_similarity:
            self.clip_validator = CLIPValidator(clip_model, self.transform)
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        image = data['image']
        text = data.get('title', '')  # Get title for CLIP similarity
        
        # During training, apply augmentations to create one modified version
        if self.training and self.augmentations:
            # Apply augmentations sequentially with their probabilities
            aug_image = image
            for aug_transform, prob in self.augmentations:
                if torch.rand(1).item() < prob:
                    aug_image = aug_transform(aug_image)
            
            # Use augmented image as main input
            img_tensor = self.transform(aug_image)
        else:
            # During inference, just use basic transform
            img_tensor = self.transform(image)
        
        
        obj = {
            'image': img_tensor
        }
        
        if hasattr(self, 'clip_validator'):
            # Compute CLIP embeddings and similarity
            text_img_similarity = self.clip_validator.validate(
                image,
                text
            )
            obj['text_image_similarity'] = text_img_similarity
        
        return obj

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