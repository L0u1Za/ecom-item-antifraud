import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch

from dataset.processor import TextProcessor, ImageProcessor, TabularProcessor
from dataset.dataloader import DataLoaderFactory
from models.architecture import FraudDetectionModel
from training.trainer import Trainer



log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds
    torch.manual_seed(cfg.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.project.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Initialize processors
    processors = {
        'text': TextProcessor(**cfg.preprocessing.text),
        'image': ImageProcessor(**cfg.preprocessing.image),
        'tabular': TabularProcessor(**cfg.preprocessing.tabular)
    }

    # Create dataloaders using factory
    dataloaders = DataLoaderFactory.create_dataloaders(
        config=cfg,
        text_processor=processors['text'],
        image_processor=processors['image'],
        tabular_processor=processors['tabular']
    )

    # Initialize model
    model = FraudDetectionModel(cfg).to(device)

    # Setup criterion
    criterion = hydra.utils.instantiate(cfg.training.criterion)

    # Setup optimizer
    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer,
        params=model.parameters()
    )

    # Setup scheduler
    scheduler = hydra.utils.instantiate(
        cfg.training.scheduler,
        optimizer=optimizer
    ) if cfg.training.scheduler else None

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        criterion=criterion,
        epochs=cfg.training.epochs,
        scheduler=scheduler,
        device=device,
        config=cfg
    )

    try:
        # Train model
        trainer.train()

    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()