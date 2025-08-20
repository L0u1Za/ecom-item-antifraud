import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch

from dataset.processor import TextProcessor, ImageProcessor, TabularProcessor
from dataset.dataloader import DataLoaderFactory
from models.architecture import FraudDetectionModel
from training.trainer import Trainer

import warnings
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./config", config_name="config")
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
        'image': ImageProcessor(cfg),
        'tabular': TabularProcessor(
            categorical_cols=list(cfg.preprocessing.tabular.categorical_cols),
            numerical_cols=list(cfg.preprocessing.tabular.numerical_cols),
            scaling=cfg.preprocessing.tabular.get('scaling', 'standard')
        ),
        'image_test': ImageProcessor(cfg, training=False),
    }

    # Fit TabularProcessor on train data and update model.tabular config for FTTransformer
    from pathlib import Path
    import pandas as pd
    train_df = pd.read_csv(cfg.experiment.data.train_path)
    processors['tabular'].fit(train_df)
    # Inject learned cardinalities and continuous count into cfg for model init
    cfg.model.model.tabular.categories = processors['tabular'].categories_cardinalities
    extra_continuous = 0
    if cfg.preprocessing.image.get('compute_clip_similarity', False):
        extra_continuous += 1
    cfg.model.model.tabular.num_continuous = processors['tabular'].num_continuous + extra_continuous

    # Create dataloaders using factory
    dataloaders = DataLoaderFactory.create_dataloaders(
        config=cfg,
        text_processor=processors['text'],
        image_processor=processors['image'],
        tabular_processor=processors['tabular'],
        image_processor_test=processors['image_test']
    )

    # Initialize model
    model = FraudDetectionModel(cfg.model).to(device)

    # Setup criterion
    criterion_cfg = cfg.training.criterion
    if '_target_' in criterion_cfg and criterion_cfg._target_ == "torch.nn.BCEWithLogitsLoss":
        pos_weight = None
        if 'pos_weight' in criterion_cfg and criterion_cfg.pos_weight is not None:
            pos_weight = torch.tensor(float(criterion_cfg.pos_weight), device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        else:
            criterion = hydra.utils.instantiate(criterion_cfg)

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