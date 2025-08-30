#!/usr/bin/env python3
"""
Data preparation script for e-commerce anti-fraud project.
This script should be run once to prepare the training and test data.

Extracts data preparation logic from notebooks and text processing logic from TextProcessor.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import argparse
import os
import sys
import pickle
import cv2
from PIL import Image, ExifTags
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel
import imagehash
from pathlib import Path
import json
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from preprocessing.text.cleaner import TextCleaner
from preprocessing.text.normalizer import normalize_text
from preprocessing.text.business_rules import BusinessRulesChecker


class ImageFeatureExtractor:
    """Extract comprehensive features from product images for fraud detection"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', enable_clip=True, enable_resnet=True):
        self.device = device
        self.enable_clip = enable_clip
        self.enable_resnet = enable_resnet
        self.setup_models()
        
    def setup_models(self):
        """Initialize all feature extraction models"""
        print("Loading image feature extraction models...")
        
        # Only load CLIP if enabled
        if self.enable_clip:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Only load ResNet if enabled
        if self.enable_resnet:
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.fc = nn.Identity()  # Remove final classification layer
            self.resnet.eval()
            self.resnet.to(self.device)
            
            # Image preprocessing for ResNet
            self.resnet_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def extract_visual_embeddings(self, image):
        """Extract CLIP and ResNet embeddings from image (PIL.Image)"""
        try:
            embeddings = {}
            # CLIP embedding (only if enabled)
            if self.enable_clip:
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    clip_features = self.clip_model.get_image_features(**inputs)
                    embeddings['clip_embedding'] = clip_features.cpu().numpy().flatten()
            else:
                embeddings['clip_embedding'] = np.zeros(512)
            # ResNet embedding (only if enabled)
            if self.enable_resnet:
                resnet_input = self.resnet_transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    resnet_features = self.resnet(resnet_input)
                    embeddings['resnet_embedding'] = resnet_features.cpu().numpy().flatten()
            else:
                embeddings['resnet_embedding'] = np.zeros(2048)
            return embeddings
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            return {
                'clip_embedding': np.zeros(512),
                'resnet_embedding': np.zeros(2048)
            }
    
    def compute_clip_similarity(self, image, text_description):
        """Compute CLIP similarity between image (PIL.Image) and text description"""
        if not self.enable_clip:
            return 0.0
        try:
            if not text_description or text_description.lower() in ['nan', 'none', '']:
                return 0.0
            if len(text_description) > 300:
                text_description = text_description[:300]
            inputs = self.clip_processor(
                text=[text_description], 
                images=image, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                similarity = outputs.logits_per_image[0, 0].item()
            return similarity
        except Exception as e:
            print(f"Error computing CLIP similarity: {e}")
            return 0.0
    
    def extract_quality_features(self, pil_image, cv_image):
        """Extract image quality and manipulation detection features from PIL and OpenCV images"""
        try:
            features = {}
            features['width'] = pil_image.width
            features['height'] = pil_image.height
            features['aspect_ratio'] = pil_image.width / pil_image.height
            features['total_pixels'] = pil_image.width * pil_image.height
            # File size cannot be computed from image object, must be passed separately if needed
            features['file_size'] = 0
            features['compression_ratio'] = 0
            if cv_image is not None:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                features['blurriness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
                features['mean_brightness'] = np.mean(gray)
                features['std_brightness'] = np.std(gray)
                features['mean_saturation'] = np.mean(hsv[:,:,1])
                features['std_saturation'] = np.std(hsv[:,:,1])
                b, g, r = cv2.split(cv_image)
                features['is_grayscale'] = np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                features['edge_density'] = edge_density
                features['has_clean_background'] = edge_density < 0.1
            try:
                exif_dict = pil_image._getexif()
                features['has_exif'] = exif_dict is not None and len(exif_dict) > 0
                features['exif_count'] = len(exif_dict) if features['has_exif'] else 0
            except:
                features['has_exif'] = False
                features['exif_count'] = 0
            return features
        except Exception as e:
            print(f"Error extracting quality features: {e}")
            return {
                'width': 0, 'height': 0, 'aspect_ratio': 1.0, 'total_pixels': 0,
                'file_size': 0, 'compression_ratio': 0, 'blurriness': 0,
                'mean_brightness': 0, 'std_brightness': 0, 'mean_saturation': 0,
                'std_saturation': 0, 'is_grayscale': False, 'edge_density': 0,
                'has_clean_background': False, 'has_exif': False, 'exif_count': 0
            }
    
    def extract_perceptual_hashes(self, pil_image):
        """Extract perceptual hashes for duplicate detection from PIL.Image"""
        try:
            return {
                'ahash': str(imagehash.average_hash(pil_image)),
                'phash': str(imagehash.phash(pil_image)),
                'dhash': str(imagehash.dhash(pil_image)),
                'whash': str(imagehash.whash(pil_image))
            }
        except Exception as e:
            print(f"Error extracting hashes: {e}")
            return {
                'ahash': '0' * 16,
                'phash': '0' * 16, 
                'dhash': '0' * 16,
                'whash': '0' * 16
            }
    
    def extract_all_features(self, image_path, text_name=None, text_brand=None, text_category=None, fast_mode=False):
        """
        Extract all features (embeddings, quality, hashes, similarities) from a single image,
        opening the image only once and reusing it for all feature extractors.
        """
        if image_path is None:
            # Return default features for missing image
            return {
                'embeddings': {
                    'clip_embedding': np.zeros(512),
                    'resnet_embedding': np.zeros(2048)
                },
                'quality': {
                    'width': 0, 'height': 0, 'aspect_ratio': 1.0, 'total_pixels': 0,
                    'file_size': 0, 'compression_ratio': 0, 'blurriness': 0,
                    'mean_brightness': 0, 'std_brightness': 0, 'mean_saturation': 0,
                    'std_saturation': 0, 'is_grayscale': False, 'edge_density': 0,
                    'has_clean_background': False, 'has_exif': False, 'exif_count': 0
                },
                'hashes': {
                    'ahash': '0' * 16,
                    'phash': '0' * 16, 
                    'dhash': '0' * 16,
                    'whash': '0' * 16
                },
                'similarities': {
                    'clip_text_similarity_name': 0.0,
                    'clip_text_similarity_brand': 0.0,
                    'clip_text_similarity_category': 0.0
                }
            }
        try:
            pil_image = Image.open(image_path).convert('RGB')
            cv_image = cv2.imread(str(image_path))
            # Embeddings
            embeddings = {}
            if self.enable_clip:
                inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    clip_features = self.clip_model.get_image_features(**inputs)
                    embeddings['clip_embedding'] = clip_features.cpu().numpy().flatten()
            else:
                embeddings['clip_embedding'] = np.zeros(512)
            if self.enable_resnet:
                resnet_input = self.resnet_transform(pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    resnet_features = self.resnet(resnet_input)
                    embeddings['resnet_embedding'] = resnet_features.cpu().numpy().flatten()
            else:
                embeddings['resnet_embedding'] = np.zeros(2048)
            # Quality
            features = {}
            features['width'] = pil_image.width
            features['height'] = pil_image.height
            features['aspect_ratio'] = pil_image.width / pil_image.height
            features['total_pixels'] = pil_image.width * pil_image.height
            features['file_size'] = os.path.getsize(image_path)
            features['compression_ratio'] = features['file_size'] / features['total_pixels']
            if cv_image is not None:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                features['blurriness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
                features['mean_brightness'] = np.mean(gray)
                features['std_brightness'] = np.std(gray)
                features['mean_saturation'] = np.mean(hsv[:,:,1])
                features['std_saturation'] = np.std(hsv[:,:,1])
                b, g, r = cv2.split(cv_image)
                features['is_grayscale'] = np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                features['edge_density'] = edge_density
                features['has_clean_background'] = edge_density < 0.1
            try:
                exif_dict = pil_image._getexif()
                features['has_exif'] = exif_dict is not None and len(exif_dict) > 0
                features['exif_count'] = len(exif_dict) if features['has_exif'] else 0
            except:
                features['has_exif'] = False
                features['exif_count'] = 0
            # Hashes
            hashes = {
                'ahash': str(imagehash.average_hash(pil_image)),
                'phash': str(imagehash.phash(pil_image)),
                'dhash': str(imagehash.dhash(pil_image)),
                'whash': str(imagehash.whash(pil_image))
            }
            # Similarities
            similarities = {
                'clip_text_similarity_name': 0.0,
                'clip_text_similarity_brand': 0.0,
                'clip_text_similarity_category': 0.0
            }
            if not fast_mode and self.enable_clip:
                for key, text in zip(['name', 'brand', 'category'], [text_name, text_brand, text_category]):
                    if text and text.lower() not in ['nan', 'none', '']:
                        # Truncate text for CLIP
                        if len(text) > 300:
                            text = text[:300]
                        inputs = self.clip_processor(
                            text=[text], images=pil_image, return_tensors="pt", padding=True, truncation=True, max_length=77
                        ).to(self.device)
                        with torch.no_grad():
                            outputs = self.clip_model(**inputs)
                            similarity = outputs.logits_per_image[0, 0].item()
                        similarities[f'clip_text_similarity_{key}'] = similarity
            return {
                'embeddings': embeddings,
                'quality': features,
                'hashes': hashes,
                'similarities': similarities
            }
        except Exception as e:
            print(f"Error extracting all features from {image_path}: {e}")
            return {
                'embeddings': {
                    'clip_embedding': np.zeros(512),
                    'resnet_embedding': np.zeros(2048)
                },
                'quality': {
                    'width': 0, 'height': 0, 'aspect_ratio': 1.0, 'total_pixels': 0,
                    'file_size': 0, 'compression_ratio': 0, 'blurriness': 0,
                    'mean_brightness': 0, 'std_brightness': 0, 'mean_saturation': 0,
                    'std_saturation': 0, 'is_grayscale': False, 'edge_density': 0,
                    'has_clean_background': False, 'has_exif': False, 'exif_count': 0
                },
                'hashes': {
                    'ahash': '0' * 16,
                    'phash': '0' * 16, 
                    'dhash': '0' * 16,
                    'whash': '0' * 16
                },
                'similarities': {
                    'clip_text_similarity_name': 0.0,
                    'clip_text_similarity_brand': 0.0,
                    'clip_text_similarity_category': 0.0
                }
            }


def extract_image_features(df, image_dir, config, batch_size=32):
    """Extract comprehensive image features for all items using batching and multiprocessing"""
    print("Extracting image features with optimizations...")
    
    # Check for fast mode configuration
    fast_mode = config.preprocessing.image.get('fast_mode', False)
    enable_clip = not fast_mode and config.preprocessing.image.get('enable_clip', True)
    enable_resnet = not fast_mode and config.preprocessing.image.get('enable_resnet', True)
    
    if fast_mode:
        print("⚡ FAST MODE ENABLED - Using lightweight feature extraction only")
        batch_size = 128  # Increase batch size for faster processing
    
    extractor = ImageFeatureExtractor(enable_clip=enable_clip, enable_resnet=enable_resnet)
    
    # Use ItemID column name from the dataset
    id_column = 'ItemID' if 'ItemID' in df.columns else 'item_id'
    
    image_extensions = ['.png']
    
    # Process images one by one (no batching)
    embeddings_data = {}
    print("Processing image features...")
    def process_row(row):
        item_id = row[id_column]
        image_path = None
        for ext in image_extensions:
            potential_path = Path(image_dir) / f"{item_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        if image_path is None:
            features = {
                id_column: item_id,
                'image_exists': False,
                'clip_text_similarity_name': 0.0,
                'clip_text_similarity_brand': 0.0,
                'clip_text_similarity_category': 0.0,
                'quality_width': 0, 'quality_height': 0, 'quality_aspect_ratio': 1.0,
                'quality_total_pixels': 0, 'quality_file_size': 0, 'quality_compression_ratio': 0,
                'quality_blurriness': 0, 'quality_mean_brightness': 0, 'quality_std_brightness': 0,
                'quality_mean_saturation': 0, 'quality_std_saturation': 0, 'quality_is_grayscale': False,
                'quality_edge_density': 0, 'quality_has_clean_background': False,
                'quality_has_exif': False, 'quality_exif_count': 0,
                'hash_ahash': '0' * 16, 'hash_phash': '0' * 16, 'hash_dhash': '0' * 16, 'hash_whash': '0' * 16
            }
            embeddings_data[str(item_id)] = {
                'clip_embedding': np.zeros(512),
                'resnet_embedding': np.zeros(2048)
            }
        else:
            try:
                from PIL import Image
                import cv2
                pil_image = Image.open(image_path).convert('RGB')
                cv_image = cv2.imread(str(image_path))
                embeddings = extractor.extract_visual_embeddings(pil_image)
                quality_features = extractor.extract_quality_features(pil_image, cv_image)
                hash_features = extractor.extract_perceptual_hashes(pil_image)
                name = str(row.get('name_rus', ''))
                brand_name = str(row.get('brand_name', ''))
                category = str(row.get('CommercialTypeName4', ''))
                if fast_mode:
                    clip_similarity_name = 0.0
                    clip_similarity_brand = 0.0
                    clip_similarity_category = 0.0
                else:
                    clip_similarity_name = extractor.compute_clip_similarity(pil_image, name)
                    clip_similarity_brand = extractor.compute_clip_similarity(pil_image, brand_name)
                    clip_similarity_category = extractor.compute_clip_similarity(pil_image, category)
                features = {
                    id_column: item_id,
                    'image_exists': True,
                    'clip_text_similarity_name': clip_similarity_name,
                    'clip_text_similarity_brand': clip_similarity_brand,
                    'clip_text_similarity_category': clip_similarity_category,
                    **{f'quality_{k}': v for k, v in quality_features.items()},
                    **{f'hash_{k}': v for k, v in hash_features.items()}
                }
                embeddings_data[str(item_id)] = embeddings
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                features = {
                    id_column: item_id,
                    'image_exists': False,
                    'clip_text_similarity_name': 0.0,
                    'clip_text_similarity_brand': 0.0,
                    'clip_text_similarity_category': 0.0,
                    'quality_width': 0, 'quality_height': 0, 'quality_aspect_ratio': 1.0,
                    'quality_total_pixels': 0, 'quality_file_size': 0, 'quality_compression_ratio': 0,
                    'quality_blurriness': 0, 'quality_mean_brightness': 0, 'quality_std_brightness': 0,
                    'quality_mean_saturation': 0, 'quality_std_saturation': 0, 'quality_is_grayscale': False,
                    'quality_edge_density': 0, 'quality_has_clean_background': False,
                    'quality_has_exif': False, 'quality_exif_count': 0,
                    'hash_ahash': '0' * 16, 'hash_phash': '0' * 16, 'hash_dhash': '0' * 16, 'hash_whash': '0' * 16
                }
                embeddings_data[str(item_id)] = {
                    'clip_embedding': np.zeros(512),
                    'resnet_embedding': np.zeros(2048)
                }
        return features

    features_df = df.apply(process_row, axis=1, result_type='expand')
    return features_df, embeddings_data


def add_consistency_features(features_df, original_df, embeddings_data, reference_features_df=None, reference_original_df=None, reference_embeddings_data=None, is_train=True):
    """Add consistency and anomaly detection features
    
    Args:
        features_df: Current dataset features
        original_df: Current dataset original data
        embeddings_data: Current dataset embeddings
        reference_features_df: Reference dataset features (train data when processing test)
        reference_original_df: Reference dataset original data
        reference_embeddings_data: Reference dataset embeddings
        is_train: Whether processing training data
    """
    dataset_type = "train" if is_train else "test"
    print(f"Computing consistency features for {dataset_type} set...")
    
    # Determine column names
    id_column = 'ItemID' if 'ItemID' in features_df.columns else 'item_id'
    category_col = 'CommercialTypeName4'
    brand_col = 'brand_name'
    price_col = 'PriceDiscounted'
    
    # Get available columns for merging
    merge_cols = [id_column]
    if category_col in original_df.columns:
        merge_cols.append(category_col)
    if brand_col in original_df.columns:
        merge_cols.append(brand_col)
    if price_col in original_df.columns:
        merge_cols.append(price_col)
    
    # Merge with original data for category, brand, price info
    merged_df = features_df.merge(original_df[merge_cols], on=id_column, how='left')
    
    # Prepare comparison datasets
    if is_train:
        # Train: use only train data for consistency calculations
        comparison_merged_df = merged_df
        comparison_embeddings_data = embeddings_data
    else:
        # Test: use train+test data combined for better consistency estimates
        if reference_features_df is not None and reference_original_df is not None:
            reference_merged_df = reference_features_df.merge(
                reference_original_df[merge_cols], on=id_column, how='left'
            )
            comparison_merged_df = pd.concat([reference_merged_df, merged_df], ignore_index=True)
            comparison_embeddings_data = {**reference_embeddings_data, **embeddings_data}
        else:
            comparison_merged_df = merged_df
            comparison_embeddings_data = embeddings_data
    
    # Category consistency using comparison dataset
    category_embeddings = {}
    if category_col in comparison_merged_df.columns:
        for category in comparison_merged_df[category_col].unique():
            if pd.isna(category):
                continue
            category_items = comparison_merged_df[comparison_merged_df[category_col] == category][id_column].tolist()
            category_embs = [comparison_embeddings_data[str(item_id)]['clip_embedding'] 
                            for item_id in category_items 
                            if str(item_id) in comparison_embeddings_data and comparison_embeddings_data[str(item_id)]['clip_embedding'].sum() != 0]
        
            if category_embs:
                category_embeddings[category] = np.mean(category_embs, axis=0)
    
    # Compute category consistency scores
    category_consistency_scores = []
    for _, row in merged_df.iterrows():
        item_id = row[id_column]
        category = row.get(category_col) if category_col in merged_df.columns else None
        
        if (pd.isna(category) or category not in category_embeddings or 
            str(item_id) not in embeddings_data or 
            embeddings_data[str(item_id)]['clip_embedding'].sum() == 0):
            category_consistency_scores.append(0.0)
        else:
            item_emb = embeddings_data[str(item_id)]['clip_embedding']
            category_emb = category_embeddings[category]
            similarity = cosine_similarity([item_emb], [category_emb])[0][0]
            category_consistency_scores.append(similarity)
    
    merged_df['category_consistency'] = category_consistency_scores
    
    # Price anomaly detection using comparison dataset
    price_anomaly_scores = []
    for _, row in merged_df.iterrows():
        item_id = row[id_column]
        price = row.get(price_col) if price_col in merged_df.columns else None
        
        if (pd.isna(price) or str(item_id) not in embeddings_data or 
            embeddings_data[str(item_id)]['clip_embedding'].sum() == 0):
            price_anomaly_scores.append(0.0)
        else:
            # Find items in similar price range from comparison dataset
            price_range = comparison_merged_df[
                (comparison_merged_df[price_col] >= price * 0.8) & 
                (comparison_merged_df[price_col] <= price * 1.2) &
                (comparison_merged_df[id_column] != item_id)
            ][id_column].tolist()
            
            if len(price_range) < 5:  # Not enough similar-priced items
                price_anomaly_scores.append(0.5)
            else:
                similar_embs = [comparison_embeddings_data[str(pid)]['clip_embedding'] 
                               for pid in price_range[:50]  # Limit for performance
                               if str(pid) in comparison_embeddings_data and comparison_embeddings_data[str(pid)]['clip_embedding'].sum() != 0]
                
                if similar_embs:
                    item_emb = embeddings_data[str(item_id)]['clip_embedding']
                    similarities = [cosine_similarity([item_emb], [emb])[0][0] for emb in similar_embs]
                    avg_similarity = np.mean(similarities)
                    price_anomaly_scores.append(1.0 - avg_similarity)  # Higher score = more anomalous
                else:
                    price_anomaly_scores.append(0.5)
    
    merged_df['price_anomaly_score'] = price_anomaly_scores
    
    return merged_df


def detect_anomalies_by_category(features_df, original_df, embeddings_data, is_train=True, train_stats=None):
    """Detect anomalies based on category-specific patterns
    
    Args:
        features_df: DataFrame with image features
        original_df: Original dataset with metadata
        embeddings_data: Dictionary containing embeddings for each item
        is_train: Whether this is training data (True) or test data (False)
        train_stats: Pre-computed statistics from training data (for test set)
    """
    print(f"Detecting category-based anomalies for {'train' if is_train else 'test'} set...")
    
    # Determine column names
    id_column = 'ItemID' if 'ItemID' in features_df.columns else 'item_id'
    category_col = 'CommercialTypeName4'
    brand_col = 'brand_name'
    price_col = 'PriceDiscounted'
    
    # Get available columns for merging
    merge_cols = [id_column]
    if category_col in original_df.columns:
        merge_cols.append(category_col)
    if brand_col in original_df.columns:
        merge_cols.append(brand_col)
    if price_col in original_df.columns:
        merge_cols.append(price_col)
    
    # Merge features with original data
    merged_df = features_df.merge(original_df[merge_cols], on=id_column, how='left')
    
    if is_train:
        # For training data: compute statistics and anomaly scores
        train_statistics = {}
        
        # Compute category-wise price statistics
        if price_col in merged_df.columns:
            price_stats = merged_df.groupby(category_col)[price_col].agg(['mean', 'std', 'min', 'max']).to_dict('index')
            train_statistics['price_stats'] = price_stats
        
        # Compute category-wise embedding centroids
        category_centroids = {}
        for category in merged_df[category_col].unique():
            if pd.notna(category):
                cat_items = merged_df[merged_df[category_col] == category][id_column].tolist()
                cat_embeddings = [embeddings_data[str(item_id)]['clip_embedding'] 
                                for item_id in cat_items 
                                if str(item_id) in embeddings_data]
                if cat_embeddings:
                    category_centroids[category] = np.mean(cat_embeddings, axis=0)
        train_statistics['category_centroids'] = category_centroids
        
        # Calculate anomaly scores using train data only
        price_anomaly_scores = []
        for _, row in merged_df.iterrows():
            item_id = row[id_column]
            price = row.get(price_col, 0)
            category = row.get(category_col)
            
            if price <= 0 or pd.isna(category) or category not in price_stats:
                price_anomaly_scores.append(0.5)
            else:
                # Compare to category statistics
                cat_stats = price_stats[category]
                if cat_stats['std'] > 0:
                    z_score = abs(price - cat_stats['mean']) / cat_stats['std']
                    anomaly_score = min(z_score / 3.0, 1.0) # Normalize to [0,1]
                else:
                    anomaly_score = 0.0
                price_anomaly_scores.append(anomaly_score)
        
        merged_df['price_anomaly_score'] = price_anomaly_scores
        return merged_df, train_statistics
        
    else:
        # For test data: use pre-computed train statistics
        if train_stats is None:
            raise ValueError("train_stats must be provided for test data")
            
        price_stats = train_stats.get('price_stats', {})
        category_centroids = train_stats.get('category_centroids', {})
        
        # Calculate anomaly scores using train statistics
        price_anomaly_scores = []
        for _, row in merged_df.iterrows():
            item_id = row[id_column]
            price = row.get(price_col, 0)
            category = row.get(category_col)
            
            if price <= 0 or pd.isna(category) or category not in price_stats:
                price_anomaly_scores.append(0.5)
            else:
                # Compare to train category statistics
                cat_stats = price_stats[category]
                if cat_stats['std'] > 0:
                    z_score = abs(price - cat_stats['mean']) / cat_stats['std']
                    anomaly_score = min(z_score / 3.0, 1.0)
                else:
                    anomaly_score = 0.0
                price_anomaly_scores.append(anomaly_score)
        
        merged_df['price_anomaly_score'] = price_anomaly_scores
        return merged_df


def detect_duplicate_images(features_df, reference_df=None, is_train=True):
    """Detect duplicate and near-duplicate images using perceptual hashes
    
    Args:
        features_df: Current dataset features
        reference_df: Reference dataset to compare against (for test set)
        is_train: Whether this is training data
    """
    dataset_type = "train" if is_train else "test"
    print(f"Detecting duplicate images for {dataset_type} set...")
    
    # Group by hash values to find duplicates
    hash_columns = ['hash_ahash', 'hash_phash', 'hash_dhash', 'hash_whash']
    id_column = 'ItemID' if 'ItemID' in features_df.columns else 'item_id'
    
    duplicate_features = []
    
    # Prepare comparison dataset
    if is_train:
        # Train: compare only within train set
        comparison_df = features_df
    else:
        # Test: compare against train+test combined
        comparison_df = pd.concat([reference_df, features_df]) if reference_df is not None else features_df
    
    for _, row in features_df.iterrows():
        item_id = row[id_column]
        
        # Count exact hash matches
        exact_duplicates = 0
        near_duplicates = 0
        
        for hash_col in hash_columns:
            hash_value = row[hash_col]
            if hash_value and hash_value != '0' * 16:
                
                # Count exact matches in comparison dataset (excluding self)
                exact_matches = len(comparison_df[
                    (comparison_df[hash_col] == hash_value) & 
                    (comparison_df[id_column] != item_id)
                ])
                exact_duplicates = max(exact_duplicates, exact_matches)
                
                # Count near matches (Hamming distance <= 5)
                near_matches = 0
                for other_hash in comparison_df[comparison_df[id_column] != item_id][hash_col]:
                    if other_hash and other_hash != '0' * 16:
                        hamming_dist = sum(c1 != c2 for c1, c2 in zip(hash_value, other_hash))
                        if hamming_dist <= 5:
                            near_matches += 1
                near_duplicates = max(near_duplicates, near_matches)
        
        duplicate_info = {
            id_column: item_id,
            'exact_duplicates': exact_duplicates,
            'near_duplicates': near_duplicates,
            'has_duplicates': exact_duplicates > 0 or near_duplicates > 0
        }
        
        duplicate_features.append(duplicate_info)
    
    return pd.DataFrame(duplicate_features)


def analyze_cross_seller_patterns(features_df, original_df, reference_df=None, reference_original_df=None, is_train=True):
    """Analyze cross-seller image sharing patterns
    
    Args:
        features_df: Current dataset features
        original_df: Current dataset original data
        reference_df: Reference dataset features (train data when processing test)
        reference_original_df: Reference dataset original data
        is_train: Whether processing training data
    """
    dataset_type = "train" if is_train else "test"
    print(f"Analyzing cross-seller patterns for {dataset_type} set...")
    
    # Determine column names
    id_column = 'ItemID' if 'ItemID' in features_df.columns else 'item_id'
    seller_col = 'SellerID' if 'SellerID' in original_df.columns else 'seller_id'
    
    if seller_col not in original_df.columns:
        print("No seller information available, skipping cross-seller analysis")
        return pd.DataFrame({id_column: features_df[id_column], 'cross_seller_matches': 0, 'suspicious_sharing': False})
    
    # Merge with seller information
    seller_df = features_df.merge(original_df[[id_column, seller_col]], on=id_column, how='left')
    
    # Prepare comparison dataset
    if is_train:
        # Train: compare only within train set
        comparison_seller_df = seller_df
    else:
        # Test: compare against train set for cross-dataset patterns
        if reference_df is not None and reference_original_df is not None:
            reference_seller_df = reference_df.merge(
                reference_original_df[[id_column, seller_col]], on=id_column, how='left'
            )
            # Combine for cross-dataset analysis
            comparison_seller_df = pd.concat([reference_seller_df, seller_df])
        else:
            comparison_seller_df = seller_df
    
    cross_seller_features = []
    
    for _, row in seller_df.iterrows():
        item_id = row[id_column]
        seller_id = row[seller_col]
        
        # Count how many other sellers use similar images (based on hashes)
        hash_columns = ['hash_ahash', 'hash_phash', 'hash_dhash', 'hash_whash']
        
        cross_seller_matches_within = 0  # Within same dataset
        cross_seller_matches_cross = 0   # Cross-dataset (test vs train)
        
        for hash_col in hash_columns:
            hash_value = row[hash_col]
            if hash_value and hash_value != '0' * 16:
                
                if is_train:
                    # Train: only count within train set
                    other_sellers = seller_df[
                        (seller_df[hash_col] == hash_value) & 
                        (seller_df[seller_col] != seller_id)
                    ][seller_col].nunique()
                    cross_seller_matches_within = max(cross_seller_matches_within, other_sellers)
                    
                else:
                    # Test: count within test and cross-dataset separately
                    # Within test set
                    within_sellers = seller_df[
                        (seller_df[hash_col] == hash_value) & 
                        (seller_df[seller_col] != seller_id)
                    ][seller_col].nunique()
                    cross_seller_matches_within = max(cross_seller_matches_within, within_sellers)
                    
                    # Cross-dataset (test vs train)
                    if reference_df is not None and reference_original_df is not None:
                        reference_seller_df = reference_df.merge(
                            reference_original_df[[id_column, seller_col]], on=id_column, how='left'
                        )
                        cross_sellers = reference_seller_df[
                            (reference_seller_df[hash_col] == hash_value) &
                            (reference_seller_df[seller_col] != seller_id)
                        ][seller_col].nunique()
                        cross_seller_matches_cross = max(cross_seller_matches_cross, cross_sellers)
        
        feature_info = {
            id_column: item_id,
            'cross_seller_matches_within': cross_seller_matches_within,
            'suspicious_sharing_within': cross_seller_matches_within > 2
        }
        
        if not is_train:
            # Add cross-dataset info for test set
            feature_info.update({
                'cross_seller_matches_cross': cross_seller_matches_cross,
                'suspicious_sharing_cross': cross_seller_matches_cross > 2
            })
        
        cross_seller_features.append(feature_info)
    
    return pd.DataFrame(cross_seller_features)


def clean_and_process_text(df, config, text_columns=['description', 'name_rus']):
    """
    Clean and process text columns, add fraud indicators based on config.
    Extracted from TextProcessor class.
    """
    text_config = config.preprocessing.text
    
    # Add fraud indicators based on brand_name and description
    if text_config.add_fraud_indicators:
        print("Adding fraud indicators...")
        if 'brand_name' in df.columns and 'description' in df.columns:
            checker = BusinessRulesChecker()
            
            # Collect all fraud indicators for all rows
            all_indicators = []
            for idx, row in df.iterrows():
                brand_name = str(row['brand_name']) if pd.notna(row['brand_name']) else ""
                description = str(row['description']) if pd.notna(row['description']) else ""
                title = str(row['name_rus']) if pd.notna(row['description']) else ""
                indicators_desc = checker(brand_name, description)
                indicators_title = checker(brand_name, title)
                indicators_desc = {f"desc_{indicator[0]}": indicator[1] for indicator in indicators_desc.items()}
                indicators_title = {f"title_{indicator[0]}": indicator[1] for indicator in indicators_title.items()}
                all_indicators.append(indicators_desc)
                all_indicators[-1].update(indicators_title)
            
            # Get all possible fraud indicator keys
            all_keys = set()
            for indicators in all_indicators:
                all_keys.update(indicators.keys())
            
            # Create separate columns for each fraud indicator
            print(f"Creating {len(all_keys)} fraud indicator columns...")
            for key in sorted(all_keys):
                column_name = f"fraud_{key}"
                df[column_name] = [indicators.get(key, False) for indicators in all_indicators]
                print(f"  - {column_name}")
    else:
        print("Skipping fraud indicators (disabled in config)")
    
    if text_config.apply_cleaning:
        print("Cleaning and processing text data...")
        
        # Initialize text processing components
        cleaner = TextCleaner(text_config.nltk_data_dir)
        
        for col in text_columns:
            if col in df.columns:
                print(f"Processing {col}...")
                # Clean text
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: cleaner.clean_text(x))
                df[col] = df[col].apply(lambda x: cleaner.clean_repeating_chars(x))
                df[col] = df[col].apply(lambda x: cleaner.truncate_text(x, text_config.max_length))
                
                # Apply normalization if enabled
                if text_config.apply_lemmatization:
                    df[col] = df[col].apply(lambda x: normalize_text(x))
    else:
        print("Skipping text cleaning (disabled in config)")
    
    return df


def add_engineered_features(df_train, df_test):
    """
    Add engineered features to both train and test datasets.
    For train: seller stats computed on train data only.
    For test: seller stats computed on train+test data combined.
    """
    print("Adding engineered features...")
    
    for df in [df_train, df_test]:
        # --- Ratios & rates ---
        df['return_rate_30'] = df['item_count_returns30'] / (df['item_count_sales30'] + 1e-6)
        df['fake_return_rate_30'] = df['item_count_fake_returns30'] / (df['item_count_sales30'] + 1e-6)
        df['refund_value_ratio_30'] = df['ExemplarReturnedValueTotal30'] / (df['GmvTotal30'] + 1e-6)

        # --- Growth / trend features ---
        df['sales_growth_7_30'] = (df['item_count_sales7']+1) / (df['item_count_sales30']+1)
        df['sales_growth_30_90'] = (df['item_count_sales30']+1) / (df['item_count_sales90']+1)

        # --- Activity rates ---
        df['sales_velocity'] = df['item_count_sales30'] / (df['item_time_alive'] + 1e-6)
        df['seller_velocity'] = df['item_count_sales30'] / (df['seller_time_alive'] + 1e-6)

        # --- Text features ---
        df['desc_len'] = df['description'].astype(str).str.len()
        df['desc_word_count'] = df['description'].astype(str).str.split().str.len()
        df['name_len'] = df['name_rus'].astype(str).str.len()

        # --- Interaction features ---
        df['price_return_interaction'] = df['PriceDiscounted'] * df['return_rate_30']
        df['gmv_per_day'] = df['GmvTotal30'] / (df['item_time_alive'] + 1)

    # --- Seller-level aggregations ---
    # For train: compute on train data only
    print("Computing seller-level aggregations for train set (train data only)...")
    seller_stats_train = df_train.groupby('SellerID').agg(
        seller_total_items=('ItemID','count'),
        seller_total_sales=('item_count_sales30','sum'),
        seller_avg_return_rate=('return_rate_30','mean'),
        # Additional features:
        seller_total_gmv_30=('GmvTotal30', 'sum'),
        seller_total_gmv_90=('GmvTotal90', 'sum'),
        seller_avg_price=('PriceDiscounted', 'mean'),
        seller_median_price=('PriceDiscounted', 'median'),
        seller_total_returns_30=('item_count_returns30', 'sum'),
        seller_total_returns_90=('item_count_returns90', 'sum'),
        seller_total_fake_returns_30=('item_count_fake_returns30', 'sum'),
        seller_total_fake_returns_90=('item_count_fake_returns90', 'sum'),
        seller_sum_rating_1=('rating_1_count', 'sum'),
        seller_sum_rating_2=('rating_2_count', 'sum'),
        seller_sum_rating_3=('rating_3_count', 'sum'),
        seller_sum_rating_4=('rating_4_count', 'sum'),
        seller_sum_rating_5=('rating_5_count', 'sum'),
        seller_total_photos=('photos_published_count', 'sum'),
        seller_total_videos=('videos_published_count', 'sum'),
        seller_avg_item_time_alive=('item_time_alive', 'mean'),
        seller_variety_mean=('ItemVarietyCount', 'mean'),
        seller_available_mean=('ItemAvailableCount', 'mean'),
        seller_total_exemplar_accepted_30=('ExemplarAcceptedCountTotal30', 'sum'),
        seller_total_order_accepted_30=('OrderAcceptedCountTotal30', 'sum'),
        seller_total_exemplar_returned_30=('ExemplarReturnedCountTotal30', 'sum'),
        seller_total_exemplar_returned_value_30=('ExemplarReturnedValueTotal30', 'sum'),
        seller_time_alive=('seller_time_alive', 'mean'),
    ).reset_index()
    
    # For test: compute on train+test data combined
    print("Computing seller-level aggregations for test set (train+test data combined)...")
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    seller_stats_test = df_combined.groupby('SellerID').agg(
        seller_total_items=('ItemID','count'),
        seller_total_sales=('item_count_sales30','sum'),
        seller_avg_return_rate=('return_rate_30','mean'),
        # Additional features:
        seller_total_gmv_30=('GmvTotal30', 'sum'),
        seller_total_gmv_90=('GmvTotal90', 'sum'),
        seller_avg_price=('PriceDiscounted', 'mean'),
        seller_median_price=('PriceDiscounted', 'median'),
        seller_total_returns_30=('item_count_returns30', 'sum'),
        seller_total_returns_90=('item_count_returns90', 'sum'),
        seller_total_fake_returns_30=('item_count_fake_returns30', 'sum'),
        seller_total_fake_returns_90=('item_count_fake_returns90', 'sum'),
        seller_sum_rating_1=('rating_1_count', 'sum'),
        seller_sum_rating_2=('rating_2_count', 'sum'),
        seller_sum_rating_3=('rating_3_count', 'sum'),
        seller_sum_rating_4=('rating_4_count', 'sum'),
        seller_sum_rating_5=('rating_5_count', 'sum'),
        seller_total_photos=('photos_published_count', 'sum'),
        seller_total_videos=('videos_published_count', 'sum'),
        seller_avg_item_time_alive=('item_time_alive', 'mean'),
        seller_variety_mean=('ItemVarietyCount', 'mean'),
        seller_available_mean=('ItemAvailableCount', 'mean'),
        seller_total_exemplar_accepted_30=('ExemplarAcceptedCountTotal30', 'sum'),
        seller_total_order_accepted_30=('OrderAcceptedCountTotal30', 'sum'),
        seller_total_exemplar_returned_30=('ExemplarReturnedCountTotal30', 'sum'),
        seller_total_exemplar_returned_value_30=('ExemplarReturnedValueTotal30', 'sum'),
        seller_time_alive=('seller_time_alive', 'mean'),
    ).reset_index()
    
    # Compute derived seller features for both
    for seller_stats in [seller_stats_train, seller_stats_test]:
        seller_stats['seller_avg_rating_5_share'] = (
            seller_stats['seller_sum_rating_5'] /
            (seller_stats['seller_sum_rating_1'] +
            seller_stats['seller_sum_rating_2'] +
            seller_stats['seller_sum_rating_3'] +
            seller_stats['seller_sum_rating_4'] +
            seller_stats['seller_sum_rating_5'] + 1e-6)
        )
    
    # --- Apply seller statistics ---
    print("Applying seller statistics...")

    # Merge different seller stats to each dataset
    df_train = df_train.merge(seller_stats_train, on='SellerID', how='left')
    df_test = df_test.merge(seller_stats_test, on='SellerID', how='left')
    
    # Get numeric columns for anomaly detection (after seller stats merge)
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target column and id column if present
    columns_to_remove = ['resolution', 'id']
    for col in columns_to_remove:
        if col in numeric_columns:
            numeric_columns.remove(col)
    
    # --- Anomaly score (fit only on train, apply to both) ---
    print("Computing anomaly scores (fitted on train data only)...")
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(df_train[numeric_columns].fillna(0))

    # Apply to both datasets
    df_train['anomaly_score'] = iso.predict(df_train[numeric_columns].fillna(0))
    df_test['anomaly_score'] = iso.predict(df_test[numeric_columns].fillna(0))
    
    return df_train, df_test


def prepare_data(config, train_path=None, test_path=None, output_dir=None):
    """
    Main data preparation function using Hydra config.
    """
    print("Starting data preparation...")
    
    # Use config defaults if not provided
    if train_path is None:
        train_path = config.data_preparation.get('train_path', 'data/ml_ozon_train.csv')
    if test_path is None:
        test_path = config.data_preparation.get('test_path', 'data/ml_ozon_test.csv')
    if output_dir is None:
        output_dir = config.data_preparation.get('output_dir', 'data')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    print("Loading raw data...")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    df_train = pd.read_csv(train_path, index_col=0)
    df_test = pd.read_csv(test_path, index_col=0)
    
    # Store original source IDs
    print("Preserving original source IDs...")
    df_train['id'] = df_train.index
    df_test['id'] = df_test.index
    
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"Train source IDs range: {df_train['id'].min()} to {df_train['id'].max()}")
    print(f"Test source IDs range: {df_test['id'].min()} to {df_test['id'].max()}")
    print(f"Target distribution in train:")
    print(df_train['resolution'].value_counts())
    
    
    # Fill nulls
    cat_cols = ['brand_name', 'CommercialTypeName4']
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in ['resolution', 'id']]

    df_train[numeric_columns] = df_train[numeric_columns].fillna(0)
    df_test[numeric_columns] = df_test[numeric_columns].fillna(0)

    df_train[cat_cols] = df_train[cat_cols].fillna("unknown")
    df_test[cat_cols] = df_test[cat_cols].fillna("unknown")
    
        # Add engineered features
    df_train, df_test = add_engineered_features(df_train, df_test)
    
    # Clean and process text data using config
    df_train = clean_and_process_text(df_train, config)
    df_test = clean_and_process_text(df_test, config)
    
    # Extract image features if enabled
    if config.preprocessing.image.get('extract_features', False):
        print("Extracting image features...")
        
        # Get image paths from experiment config
        train_image_dir = config.experiment.data.get('train_images_path', 'data/train_images/train_images')
        test_image_dir = config.experiment.data.get('test_images_path', 'data/test_images/test_images')
        
        print(f"Training images path: {train_image_dir}")
        print(f"Test images path: {test_image_dir}")
        
        # Extract features for training data
        train_image_features, train_embeddings = extract_image_features(
            df_train, train_image_dir, config
        )
        
        # Extract features for test data  
        test_image_features, test_embeddings = extract_image_features(
            df_test, test_image_dir, config
        )
        
        # Merge image features with main datasets
        id_column = 'ItemID' if 'ItemID' in df_train.columns else 'item_id'
        df_train = df_train.merge(train_image_features, on=id_column, how='left')
        df_test = df_test.merge(test_image_features, on=id_column, how='left')
        
        # --- TRAIN/TEST SEPARATION FOR ANOMALY DETECTION ---
        print("Computing train-only statistics for anomaly detection...")
        
        # 1. Anomaly detection using only train statistics
        train_with_anomalies, train_stats = detect_anomalies_by_category(
            train_image_features, df_train, train_embeddings, is_train=True
        )
        
        # Apply train statistics to test data
        test_with_anomalies = detect_anomalies_by_category(
            test_image_features, df_test, test_embeddings, is_train=False, train_stats=train_stats
        )
        
        # 2. Duplicate detection with proper train/test separation
        train_duplicates = detect_duplicate_images(
            train_image_features, reference_df=None, is_train=True
        )
        
        test_duplicates = detect_duplicate_images(
            test_image_features, reference_df=train_image_features, is_train=False
        )
        
        # 3. Cross-seller analysis with train/test separation
        train_cross_seller = analyze_cross_seller_patterns(
            train_image_features, df_train, is_train=True
        )
        
        test_cross_seller = analyze_cross_seller_patterns(
            test_image_features, df_test, 
            reference_df=train_image_features, reference_original_df=df_train, 
            is_train=False
        )
        
        # 4. Consistency features with train/test separation
        train_consistency = add_consistency_features(
            train_image_features, df_train, train_embeddings, is_train=True
        )
        
        test_consistency = add_consistency_features(
            test_image_features, df_test, test_embeddings,
            reference_features_df=train_image_features, 
            reference_original_df=df_train,
            reference_embeddings_data=train_embeddings,
            is_train=False
        )
        
        # Merge all features back to main datasets
        df_train = df_train.merge(train_duplicates, on=id_column, how='left')
        df_train = df_train.merge(train_cross_seller, on=id_column, how='left')
        df_train = df_train.merge(train_with_anomalies[[id_column, 'price_anomaly_score']], on=id_column, how='left')
        df_train = df_train.merge(train_consistency[[id_column, 'category_consistency', 'price_anomaly_score']], on=id_column, how='left', suffixes=('', '_consistency'))
        
        df_test = df_test.merge(test_duplicates, on=id_column, how='left')
        df_test = df_test.merge(test_cross_seller, on=id_column, how='left')
        df_test = df_test.merge(test_with_anomalies[[id_column, 'price_anomaly_score']], on=id_column, how='left')
        df_test = df_test.merge(test_consistency[[id_column, 'category_consistency', 'price_anomaly_score']], on=id_column, how='left', suffixes=('', '_consistency'))
        
        # Save train statistics for future use
        train_stats_path = os.path.join(output_dir, 'train_image_stats.pkl')
        with open(train_stats_path, 'wb') as f:
            pickle.dump(train_stats, f)
        print(f"Saved train statistics to: {train_stats_path}")
        
        # Save embeddings separately for train and test
        train_embeddings_path = os.path.join(output_dir, 'train_image_features.pkl')
        test_embeddings_path = os.path.join(output_dir, 'test_image_features.pkl')
        
        with open(train_embeddings_path, 'wb') as f:
            pickle.dump(train_embeddings, f)
        with open(test_embeddings_path, 'wb') as f:
            pickle.dump(test_embeddings, f)
            
        print(f"Saved train embeddings to: {train_embeddings_path}")
        print(f"Saved test embeddings to: {test_embeddings_path}")

    
    # Save prepared datasets
    print("Saving prepared datasets...")

    # Restore original indices
    df_train = df_train.set_index('id')
    df_test = df_test.set_index('id')
    
    # Save full prepared datasets
    train_prepared_path = os.path.join(output_dir, 'train_prepared.csv')
    test_prepared_path = os.path.join(output_dir, 'test_prepared.csv')
    
    df_train.to_csv(train_prepared_path)
    df_test.to_csv(test_prepared_path)
    
    print(f"Saved prepared training data to: {train_prepared_path}")
    print(f"Saved prepared test data to: {test_prepared_path}")
    
    # Create train/validation split
    print("Creating train/validation split...")
    df_train_clean = df_train.drop(columns=['SellerID'], errors='ignore')
    df_test_clean = df_test.drop(columns=['SellerID'], errors='ignore')
    
    print(f"Final datasets contain id column: Train={('id' in df_train_clean.columns)}, Test={('id' in df_test_clean.columns)}")
    
    df_train_train, df_train_val = train_test_split(
        df_train_clean, test_size=0.2, random_state=42,
        stratify=df_train_clean['resolution']
    )
    
    # Save train/validation splits
    train_split_path = os.path.join(output_dir, 'train.csv')
    val_split_path = os.path.join(output_dir, 'val.csv')
    test_final_path = os.path.join(output_dir, 'test.csv')
    
    df_train_train.to_csv(train_split_path)
    df_train_val.to_csv(val_split_path)
    df_test_clean.to_csv(test_final_path)
    
    print(f"Saved training split to: {train_split_path}")
    print(f"Saved validation split to: {val_split_path}")
    print(f"Saved final test data to: {test_final_path}")
    
    print("Data preparation completed successfully!")
    
    return df_train, df_test


@hydra.main(version_base=None, config_path="../src/config", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main function using Hydra configuration.
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    try:
        prepare_data(config)
    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
