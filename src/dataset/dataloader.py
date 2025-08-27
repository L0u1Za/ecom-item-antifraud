from torch.utils.data import DataLoader
from typing import Optional, Dict
import hydra
from omegaconf import DictConfig
from dataset.processor import ImageProcessor, TabularProcessor, TextProcessor
from dataset.fraud_dataset import FraudDataset
from dataset.collator import MultiModalCollator

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
            data_path=config.experiment.data.train_path,
            image_dir=config.experiment.data.train_images_path,
            text_processor=text_processor,
            image_processor=image_processor,
            tabular_processor=tabular_processor,
            model_config=config.model.model
        )
        
        # Create collator
        collator = MultiModalCollator(config)
        
        # Create train dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.experiment.data.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        dataloaders = {'train': train_loader}
        
        # Conditionally create val dataloader if val_path is specified and not empty
        val_path = getattr(config.experiment.data, "val_path", None)
        if val_path:
            val_dataset = FraudDataset(
                data_path=val_path,
                image_dir=config.experiment.data.train_images_path,
                text_processor=text_processor,
                image_processor=image_processor_test,
                tabular_processor=tabular_processor,
                model_config=config.model.model
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.experiment.data.num_workers,
                pin_memory=True,
                collate_fn=collator
            )
            dataloaders['val'] = val_loader
        
        return dataloaders