import torch
from torchvision import transforms
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig
import math
import pandas as pd
import os

from preprocessing.image.clip_validator import CLIPValidator

class TextProcessor:
    def __init__(self,
                 max_length=512,
                 **kwargs):
        """
        Simplified TextProcessor for inference/training pipeline.
        Data cleaning and fraud indicators are now handled in the data preparation script.
        """
        self.max_length = max_length

    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing for inference.
        Heavy cleaning is done during data preparation.
        """
        if isinstance(text, str):
            # Basic truncation for inference
            if len(text) > self.max_length:
                text = text[:self.max_length]
        else:
            text = str(text) if text is not None else ""
        
        return text
        
    def __call__(self, text_data: Dict[str, str]) -> Dict[str, str]:
        """
        Process text data for model input.
        Assumes data has already been cleaned during preparation.
        """
        # Process title and description with basic preprocessing
        title = text_data.get('title', '')
        description = text_data.get('description', '')

        # Apply basic preprocessing
        title = self.preprocess_text(title)
        description = self.preprocess_text(description)

        obj = {
            'title': title,
            'description': description
        }

        return obj

class ImageProcessor:
    def get_empty_image(self):
        size = tuple(self.config.preprocessing.image.size)
        if hasattr(self, 'clip_validator'):
            return {
                'images': [torch.zeros(3, size[0], size[1])],
                'text_image_similarity': torch.tensor(0.0)
            }
        return {
            'images': [torch.zeros(3, size[0], size[1])]
        }
        

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
        
    def __call__(self, image_path: str) -> Dict[str, torch.Tensor]:
        if not image_path or not os.path.exists(image_path):
            return self.get_empty_image()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return self.get_empty_image()

        text = "" # Placeholder, as title is not passed directly anymore
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
        
        img_tensors = [img_tensor]
        
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
        self.num_categorical_features: int = 0
        self.num_continuous_features: int = 0

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

        # Store feature counts
        self.num_categorical_features = len(self.categorical_cols)
        self.num_continuous_features = len(self.numerical_cols)

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