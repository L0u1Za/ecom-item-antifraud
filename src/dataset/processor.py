import torch
from torchvision import transforms
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig
import math
import numpy as np
import pandas as pd

from preprocessing.text.cleaner import TextCleaner
from preprocessing.text.normalizer import normalize_text
from preprocessing.text.business_rules import BusinessRulesChecker

from preprocessing.image.clip_validator import CLIPValidator

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
        brand_name = text_data['brand_name']

        # Get suspicious patterns first
        if self.add_fraud_indicators:
            checker = BusinessRulesChecker()
            fraud_indicators = checker(brand_name, description)

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

        if config.preprocessing.image.compute_clip_similarity:
            self.clip_validator = CLIPValidator(config.preprocessing.image.clip_model, self.transform)
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = data['images']
        if images is None:
            return {
                "images": []
            }
        text = data.get('title', '')  # Get title for CLIP similarity
        
        # During training, apply augmentations to create one modified version
        img_tensors = []
        for image in images:
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
            img_tensors.append(img_tensor)
        
        obj = {
            'images': img_tensors
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
                 numerical_cols: List[str] = None,
                 scaling: str = "standard"):
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.scaling = scaling

        # Learned state
        self.category_value_to_index: Dict[str, Dict[Any, int]] = {}
        self.category_cardinalities: List[int] = []
        self.num_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit encoders and scalers on training dataframe.

        - Categorical columns: build value->index mapping with 0 reserved for unknown/missing
        - Numerical columns: compute mean/std (standard) or min/max (minmax)
        """
        # Fit categoricals
        self.category_value_to_index = {}
        self.category_cardinalities = []
        for col in self.categorical_cols:
            # Get unique values excluding NaNs
            if col not in df.columns:
                # still register unknown-only category
                self.category_value_to_index[col] = {}
                self.category_cardinalities.append(1)
                continue
            values = pd.Series(df[col]).astype(str)
            # Include only non-null values
            unique_values = pd.Index(values[values != 'nan'].unique())
            # Reserve 0 for unknown
            mapping = {val: i + 1 for i, val in enumerate(unique_values)}
            self.category_value_to_index[col] = mapping
            # cardinality includes index 0 for unknown
            self.category_cardinalities.append(len(mapping) + 1)

        # Fit numerical scaler
        self.num_stats = {}
        for col in self.numerical_cols:
            if col not in df.columns:
                self.num_stats[col] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            if self.scaling == "standard":
                mean = float(series.mean()) if not math.isnan(series.mean()) else 0.0
                std = float(series.std(ddof=0)) if not math.isnan(series.std(ddof=0)) else 1.0
                if std == 0.0:
                    std = 1.0
                self.num_stats[col] = {"mean": mean, "std": std}
            elif self.scaling == "minmax":
                min_v = float(series.min()) if not math.isnan(series.min()) else 0.0
                max_v = float(series.max()) if not math.isnan(series.max()) else 1.0
                if max_v == min_v:
                    max_v = min_v + 1.0
                self.num_stats[col] = {"min": min_v, "max": max_v}
            else:
                # No scaling
                self.num_stats[col] = {}

    @property
    def categories_cardinalities(self) -> List[int]:
        return list(self.category_cardinalities)

    @property
    def num_continuous(self) -> int:
        return len(self.numerical_cols)

    def _encode_category_value(self, col: str, value: Any) -> int:
        mapping = self.category_value_to_index.get(col, {})
        if pd.isna(value):
            return 0
        key = str(value)
        return mapping.get(key, 0)

    def _scale_numeric_value(self, col: str, value: Any) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            # impute with mean (standard) or min (minmax) or 0
            if self.scaling == "standard":
                return float(self.num_stats.get(col, {}).get("mean", 0.0))
            elif self.scaling == "minmax":
                return float(self.num_stats.get(col, {}).get("min", 0.0))
            else:
                return 0.0
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if self.scaling == "standard":
            mean = self.num_stats.get(col, {}).get("mean", 0.0)
            std = self.num_stats.get(col, {}).get("std", 1.0)
            return (v - mean) / std
        if self.scaling == "minmax":
            min_v = self.num_stats.get(col, {}).get("min", 0.0)
            max_v = self.num_stats.get(col, {}).get("max", 1.0)
            return (v - min_v) / (max_v - min_v)
        return v

    def __call__(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Categorical indices
        cat_indices: List[int] = []
        for col in self.categorical_cols:
            val = row.get(col, None)
            idx = self._encode_category_value(col, val)
            cat_indices.append(idx)

        # Continuous values
        cont_values: List[float] = []
        for col in self.numerical_cols:
            val = row.get(col, None)
            scaled = self._scale_numeric_value(col, val)
            cont_values.append(float(scaled))

        categorical = torch.tensor(cat_indices, dtype=torch.long)
        continuous = torch.tensor(cont_values, dtype=torch.float32) if cont_values else torch.zeros(0, dtype=torch.float32)

        return {
            'categorical': categorical,
            'continuous': continuous
        }