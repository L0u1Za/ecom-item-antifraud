from torch.utils.data import DataLoader
from typing import Optional, Dict
import hydra
from omegaconf import DictConfig
from processor import ImageProcessor, TabularProcessor, TextProcessor
from fraud_dataset import FraudDataset
from collator import MultiModalCollator

class DataLoaderFactory:
    @staticmethod
    def create_dataloaders(
        config: DictConfig,
        text_processor: Optional[TextProcessor] = None,
        image_processor: Optional[ImageProcessor] = None,
        tabular_processor: Optional[TabularProcessor] = None,
        image_processor_test: Optional[ImageProcessor] = None
    ) -> Dict[str, DataLoader]:
        
        # Create datasets
        train_dataset = FraudDataset(
            text_processor=text_processor,
            image_processor=image_processor,
            tabular_processor=tabular_processor
        )
        
        val_dataset = FraudDataset(
            text_processor=text_processor,
            image_processor=image_processor_test,
            tabular_processor=tabular_processor
        )
        # Create collator
        collator = MultiModalCollator(config)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        return {
            'train': train_loader,
            'val': val_loader
        }