#!/usr/bin/env python3
"""
Demo script showing modality dropout functionality for the fraud detection model.
This script demonstrates how the model handles missing modalities and applies dropout during training.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.architecture import FraudDetectionModel
from omegaconf import OmegaConf

def create_demo_config():
    """Create a demo configuration for testing."""
    config_dict = {
        'model': {
            'text': {
                'enabled': True,
                'name': 'bert-base-uncased',
                'projection_dim': 128,
                'dropout': 0.3
            },
            'image': {
                'enabled': True,
                'name': 'resnet18',
                'pretrained': True,
                'pool_type': 'mean',
                'projection_dim': 128,
                'dropout': 0.3
            },
            'tabular': {
                'enabled': True,
                'input_dim': 64,
                'hidden_dim': 128,
                'projection_dim': 128,
                'dropout': 0.3
            },
            'classifier': {
                'hidden_dim': 256,
                'dropout': 0.5
            },
            'fusion': {
                'input_dim': 384,  # 128 + 128 + 128
                'output_dim': 256
            },
            'modality_dropout': 0.0,
            'training_modality_dropout': 0.2
        }
    }
    return OmegaConf.create(config_dict)

def create_mock_towers(model):
    """Create mock towers for demonstration purposes."""
    
    # Mock text tower
    if hasattr(model, 'text_tower'):
        class MockTextTower:
            def __init__(self):
                self.text_proj = type('Mock', (), {'out_features': 128})()
            def forward(self, inputs):
                if inputs is None:
                    return None
                return torch.randn(2, 128)
        model.text_tower = MockTextTower()
    
    # Mock image tower
    if hasattr(model, 'image_tower'):
        class MockImageTower:
            def __init__(self):
                self.proj = type('Mock', (), {'out_features': 128})()
            def forward(self, inputs):
                if inputs is None:
                    return None
                return torch.randn(2, 128)
        model.image_tower = MockImageTower()
    
    # Mock tabular tower
    if hasattr(model, 'tabular_tower'):
        class MockTabularTower:
            def __init__(self):
                self.encoder = type('Mock', (), {'normalized_shape': [128]})()
            def forward(self, inputs):
                if inputs is None:
                    return None
                return torch.randn(2, 128)
        model.tabular_tower = MockTabularTower()

def create_mock_fusion():
    """Create a mock fusion module."""
    class MockFusion:
        def __init__(self):
            pass
        def forward(self, embeds):
            # Simple concatenation for demo
            return torch.cat(embeds, dim=1)
    
    return MockFusion()

def demo_scenario_1(model, scenario_name, batch):
    """Demo scenario 1: All modalities present."""
    print(f"\n=== {scenario_name} ===")
    print(f"Input batch keys: {list(batch.keys())}")
    
    with torch.no_grad():
        logits, outputs = model(batch)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of embeddings: {len(outputs['embeds'])} (for enabled modalities)")
    print(f"Fused features shape: {outputs['fused_features'].shape}")

def demo_scenario_2(model, scenario_name, batch):
    """Demo scenario 2: Missing text modality."""
    print(f"\n=== {scenario_name} ===")
    print(f"Input batch keys: {list(batch.keys())}")
    print("Note: Text modality is missing (None) - will be zero-filled")
    
    with torch.no_grad():
        logits, outputs = model(batch)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of embeddings: {len(outputs['embeds'])} (for enabled modalities)")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    # Check which embeddings are zero
    modality_names = []
    if hasattr(model, 'text_tower'):
        modality_names.append('Text')
    if hasattr(model, 'image_tower'):
        modality_names.append('Image')
    if hasattr(model, 'tabular_tower'):
        modality_names.append('Tabular')
    
    for i, embed in enumerate(outputs['embeds']):
        if torch.all(embed == 0).item():
            print(f"  {modality_names[i]} embedding: Zero-filled (missing modality)")
        else:
            print(f"  {modality_names[i]} embedding: Real data")

def demo_scenario_3(model, scenario_name, batch):
    """Demo scenario 3: Training mode with modality dropout."""
    print(f"\n=== {scenario_name} ===")
    print(f"Input batch keys: {list(batch.keys())}")
    print("Model is in training mode with modality dropout enabled")
    print("Note: Only enabled modalities can be randomly masked to zero")
    
    # Set model to training mode
    model.train()
    
    with torch.no_grad():
        logits, outputs = model(batch)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of embeddings: {len(outputs['embeds'])} (for enabled modalities)")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    # Check which embeddings might be zero due to dropout
    modality_names = []
    if hasattr(model, 'text_tower'):
        modality_names.append('Text')
    if hasattr(model, 'image_tower'):
        modality_names.append('Image')
    if hasattr(model, 'tabular_tower'):
        modality_names.append('Tabular')
    
    for i, embed in enumerate(outputs['embeds']):
        if torch.all(embed == 0).item():
            print(f"  {modality_names[i]} embedding: Zero (randomly masked during training)")
        else:
            print(f"  {modality_names[i]} embedding: Active (not masked)")

def demo_scenario_4(model, scenario_name, batch):
    """Demo scenario 4: Evaluation mode - no modality dropout."""
    print(f"\n=== {scenario_name} ===")
    print(f"Input batch keys: {list(batch.keys())}")
    print("Model is in evaluation mode - NO modality dropout applied")
    print("Note: All enabled modalities work normally during inference")
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        logits, outputs = model(batch)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of embeddings: {len(outputs['embeds'])} (for enabled modalities)")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    # All embeddings should be active in evaluation mode
    modality_names = []
    if hasattr(model, 'text_tower'):
        modality_names.append('Text')
    if hasattr(model, 'image_tower'):
        modality_names.append('Image')
    if hasattr(model, 'tabular_tower'):
        modality_names.append('Tabular')
    
    for i, embed in enumerate(outputs['embeds']):
        if torch.all(embed == 0).item():
            print(f"  {modality_names[i]} embedding: Zero (missing input modality)")
        else:
            print(f"  {modality_names[i]} embedding: Active (working normally)")

def demo_scenario_5(model, scenario_name, batch):
    """Demo scenario 5: Training mode with high modality dropout."""
    print(f"\n=== {scenario_name} ===")
    print(f"Input batch keys: {list(batch.keys())}")
    print("Model is in training mode with HIGH modality dropout (80%)")
    print("This will demonstrate random masking of enabled modalities")
    
    # Set high training modality dropout
    original_dropout = model.training_modality_dropout
    model.training_modality_dropout = 0.8
    
    # Set model to training mode
    model.train()
    
    # Run multiple times to show randomness
    print("Running multiple forward passes to show dropout randomness:")
    
    for run in range(3):
        with torch.no_grad():
            logits, outputs = model(batch)
        
        print(f"  Run {run + 1}: ", end="")
        for i, embed in enumerate(outputs['embeds']):
            modality_names = []
            if hasattr(model, 'text_tower'):
                modality_names.append('Text')
            if hasattr(model, 'image_tower'):
                modality_names.append('Image')
            if hasattr(model, 'tabular_tower'):
                modality_names.append('Tabular')
                
            if torch.all(embed == 0).item():
                print(f"{modality_names[i]}[0] ", end="")
            else:
                print(f"{modality_names[i]}[+] ", end="")
        print()
    
    # Restore original dropout rate
    model.training_modality_dropout = original_dropout

def main():
    """Main demo function."""
    print("🚀 Fraud Detection Model - Modality Dropout Demo")
    print("=" * 50)
    
    # Create configuration
    config = create_demo_config()
    print(f"Configuration created with training modality dropout: {config.model.training_modality_dropout}")
    
    # Create model
    model = FraudDetectionModel(config)
    print("Model created successfully")
    
    # Count enabled modalities
    enabled_count = 0
    if hasattr(model, 'text_tower'):
        enabled_count += 1
    if hasattr(model, 'image_tower'):
        enabled_count += 1
    if hasattr(model, 'tabular_tower'):
        enabled_count += 1
    
    print(f"Number of enabled modalities: {enabled_count}")
    
    # Create mock towers and fusion
    create_mock_towers(model)
    model.fusion = create_mock_fusion()
    print("Mock towers and fusion created")
    
    # Demo 1: All modalities present
    batch1 = {
        'text': {'input_ids': torch.randint(0, 1000, (2, 10))},
        'images': torch.randn(2, 3, 3, 224, 224),
        'tabular': torch.randn(2, 64)
    }
    demo_scenario_1(model, "All Modalities Present", batch1)
    
    # Demo 2: Missing text modality
    batch2 = {
        'text': None,  # Missing text
        'images': torch.randn(2, 3, 3, 224, 224),
        'tabular': torch.randn(2, 64)
    }
    demo_scenario_2(model, "Missing Text Modality", batch2)
    
    # Demo 3: Training mode with modality dropout
    batch3 = {
        'text': {'input_ids': torch.randint(0, 1000, (2, 10))},
        'images': torch.randn(2, 3, 3, 224, 224),
        'tabular': torch.randn(2, 64)
    }
    demo_scenario_3(model, "Training Mode with Modality Dropout", batch3)
    
    # Demo 4: Evaluation mode - no modality dropout
    batch4 = {
        'text': {'input_ids': torch.randint(0, 1000, (2, 10))},
        'images': torch.randn(2, 3, 3, 224, 224),
        'tabular': torch.randn(2, 64)
    }
    demo_scenario_4(model, "Evaluation Mode - No Modality Dropout", batch4)
    
    # Demo 5: High modality dropout demonstration
    batch5 = {
        'text': {'input_ids': torch.randint(0, 1000, (2, 10))},
        'images': torch.randn(2, 3, 3, 224, 224),
        'tabular': torch.randn(2, 64)
    }
    demo_scenario_5(model, "High Modality Dropout Demonstration", batch5)
    
    # Demo 6: Missing multiple modalities
    batch6 = {
        'text': None,  # Missing text
        'images': None,  # Missing images
        'tabular': torch.randn(2, 64)  # Only tabular present
    }
    demo_scenario_2(model, "Missing Text and Image Modalities", batch6)
    
    print("\n✅ Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print(f"- Provides {enabled_count} embeddings (one per enabled modality)")
    print("- Missing modalities are automatically zero-filled")
    print("- Modality dropout only applies to ENABLED modalities during training")
    print("- Random masking of enabled modalities during training")
    print("- NO modality dropout during inference/evaluation")
    print("- Robust fusion with flexible modality structure")

if __name__ == "__main__":
    main()
