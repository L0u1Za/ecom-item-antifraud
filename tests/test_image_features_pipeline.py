#!/usr/bin/env python3
"""
Test script to verify the complete image features extraction and training pipeline.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import tempfile
import shutil
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hydra import initialize, compose
from dataset.processor import ImageProcessor, TextProcessor, TabularProcessor
from dataset.fraud_dataset import FraudDataset
from models.architecture import FraudDetectionModel
from torch.utils.data import DataLoader

def create_mock_images(image_dir, num_images=5):
    """Create mock images for testing"""
    os.makedirs(image_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create a simple colored image
        img = Image.new('RGB', (224, 224), color=(i*50 % 255, (i*30) % 255, (i*70) % 255))
        img.save(os.path.join(image_dir, f"{i}.jpg"))
    
    print(f"Created {num_images} mock images in {image_dir}")

def create_mock_dataset(num_samples=5):
    """Create a mock dataset CSV for testing"""
    data = {
        'ItemID': list(range(num_samples)),
        'name_rus': [f'Product {i}' for i in range(num_samples)],
        'description': [f'Description for product {i}' for i in range(num_samples)],
        'brand_name': [f'Brand {i % 3}' for i in range(num_samples)],
        'price': np.random.uniform(100, 1000, num_samples),
        'CategoryID': np.random.randint(1, 5, num_samples),
        'SellerID': np.random.randint(1, 3, num_samples),
        'resolution': np.random.randint(0, 2, num_samples)  # target
    }
    return pd.DataFrame(data)

def test_image_feature_extraction():
    """Test the image feature extraction process"""
    print("Testing image feature extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        image_dir = os.path.join(temp_dir, "images")
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        create_mock_images(image_dir, 5)
        mock_df = create_mock_dataset(5)
        
        # Save mock dataset
        dataset_path = os.path.join(data_dir, "test_data.csv")
        mock_df.to_csv(dataset_path, index=False)
        
        # Test image feature extraction
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        from prepare_data import ImageFeatureExtractor, extract_image_features
        
        # Extract features
        features_df, embeddings_data = extract_image_features(
            mock_df, image_dir, data_dir, None
        )
        
        # Verify features were extracted
        assert len(features_df) == 5, f"Expected 5 feature rows, got {len(features_df)}"
        assert len(embeddings_data) == 5, f"Expected 5 embedding entries, got {len(embeddings_data)}"
        
        # Check feature columns
        expected_cols = ['ItemID', 'image_exists', 'clip_text_similarity']
        for col in expected_cols:
            assert col in features_df.columns, f"Missing column: {col}"
        
        # Check embeddings structure
        for item_id, emb in embeddings_data.items():
            assert 'clip_embedding' in emb, f"Missing CLIP embedding for {item_id}"
            assert 'resnet_embedding' in emb, f"Missing ResNet embedding for {item_id}"
            assert len(emb['clip_embedding']) == 512, f"Wrong CLIP embedding size for {item_id}"
            assert len(emb['resnet_embedding']) == 2048, f"Wrong ResNet embedding size for {item_id}"
        
        print("✅ Image feature extraction test passed!")
        return True

def test_precomputed_features_pipeline():
    """Test the complete pipeline with precomputed features"""
    print("Testing precomputed features pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test data
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Create mock precomputed features
            mock_features = {}
            for i in range(5):
                mock_features[str(i)] = {
                    'clip_embedding': np.random.randn(512).astype(np.float32),
                    'resnet_embedding': np.random.randn(2048).astype(np.float32)
                }
            
            # Save mock features
            features_path = data_dir / "image_features.pkl"
            with open(features_path, 'wb') as f:
                pickle.dump(mock_features, f)
            
            # Create mock dataset
            mock_dataset = create_mock_dataset(5)
            dataset_path = data_dir / "test_dataset.csv"
            mock_dataset.to_csv(dataset_path, index=False)
            
            # Initialize Hydra config
            with initialize(version_base=None, config_path="../src/config"):
                cfg = compose(config_name="config")
                
                # Enable precomputed features
                cfg.preprocessing.image.use_precomputed_features = True
                cfg.preprocessing.image.precomputed_features_path = str(features_path)
                cfg.model.image.enabled = True
                cfg.model.image.use_precomputed_features = True
                cfg.model.image.precomputed_input_dim = 2560  # 512 + 2048
            
            # Initialize processors
            text_processor = TextProcessor()
            image_processor = ImageProcessor(cfg.preprocessing.image, training=False)
            
            # Create mock tabular processor
            tabular_processor = TabularProcessor()
            sample_data = mock_dataset.drop(['resolution'], axis=1)
            tabular_processor.fit(sample_data)
            
            # Create dataset
            dataset = FraudDataset(
                data_path=str(dataset_path),
                image_dir=None,  # Not needed for precomputed features
                text_processor=text_processor,
                image_processor=image_processor,
                tabular_processor=tabular_processor,
                model_config=cfg.model
            )
            
            # Test data loading
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(dataloader))
            processed_data, labels = batch
            
            # Verify batch structure
            assert processed_data['text'] is not None, "Text features should not be None"
            assert processed_data['images'] is not None, "Image features should not be None"
            assert processed_data['tabular'] is not None, "Tabular features should not be None"
            
            # Check image features shape (should be precomputed features)
            expected_img_shape = (2, 2560)  # batch_size=2, feature_dim=2560
            actual_img_shape = processed_data['images'].shape
            assert actual_img_shape == expected_img_shape, f"Expected image shape {expected_img_shape}, got {actual_img_shape}"
            
            # Test model forward pass
            model = FraudDetectionModel(cfg, training=False)
            model.eval()
            
            with torch.no_grad():
                outputs = model(processed_data)
                assert outputs.shape == (2, 1), f"Expected output shape (2, 1), got {outputs.shape}"
            
            print("✅ Precomputed features pipeline test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_feature_consistency():
    """Test that extracted features are consistent and meaningful"""
    print("Testing feature consistency...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images with different characteristics
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        # Create images with different properties
        # Bright image
        bright_img = Image.new('RGB', (224, 224), color=(200, 200, 200))
        bright_img.save(os.path.join(image_dir, "0.jpg"))
        
        # Dark image
        dark_img = Image.new('RGB', (224, 224), color=(50, 50, 50))
        dark_img.save(os.path.join(image_dir, "1.jpg"))
        
        # Create mock dataset
        mock_df = pd.DataFrame({
            'ItemID': [0, 1],
            'name_rus': ['Bright Product', 'Dark Product'],
            'description': ['A bright white product', 'A dark black product'],
            'brand_name': ['BrandA', 'BrandB'],
            'price': [100, 200],
            'CategoryID': [1, 2],
            'resolution': [0, 1]
        })
        
        # Extract features
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        from prepare_data import ImageFeatureExtractor
        
        extractor = ImageFeatureExtractor()
        
        # Extract features for both images
        bright_features = extractor.extract_quality_features(os.path.join(image_dir, "0.jpg"))
        dark_features = extractor.extract_quality_features(os.path.join(image_dir, "1.jpg"))
        
        # Verify brightness difference is captured
        assert bright_features['mean_brightness'] > dark_features['mean_brightness'], \
            "Bright image should have higher mean brightness"
        
        # Extract embeddings
        bright_emb = extractor.extract_visual_embeddings(os.path.join(image_dir, "0.jpg"))
        dark_emb = extractor.extract_visual_embeddings(os.path.join(image_dir, "1.jpg"))
        
        # Verify embeddings are different
        clip_similarity = np.dot(bright_emb['clip_embedding'], dark_emb['clip_embedding'])
        assert clip_similarity < 0.99, "Different images should have different CLIP embeddings"
        
        print("✅ Feature consistency test passed!")
        return True

def main():
    """Run all tests"""
    print("🚀 Testing Image Features Pipeline")
    print("=" * 50)
    
    tests = [
        test_image_feature_extraction,
        test_precomputed_features_pipeline,
        test_feature_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Image features pipeline is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
