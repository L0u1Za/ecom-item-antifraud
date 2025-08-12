import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from wandb_logger import WandBLogger
from training.validation import Validator
from utils.logger import setup_logger
from pathlib import Path
from omegaconf import DictConfig

class Trainer:
    def __init__(self, 
                 model, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 optimizer, 
                 criterion, 
                 epochs: int, 
                 device: str,
                 scheduler: _LRScheduler = None,
                 config: DictConfig = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler
        self.logger = setup_logger('fraud_detection', 'fraud_detection.log')
        
        # Initialize validator
        self.validator = Validator(
            model=model,
            dataloader=val_loader,
            device=device,
            threshold=config.training.validation.threshold  # Initial threshold, can be optimized during training
        )
        
        if config.experiment.wandb.enabled == True:
            project_name = config.experiment.wandb.project
            # Initialize wandb logger
            wandb_config = {
                "model_type": model.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "scheduler": scheduler.__class__.__name__ if scheduler else "None",
                "scheduler_params": scheduler.state_dict() if scheduler else None,
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                **(config or {})
            }
            self.wandb_logger = WandBLogger(project_name, wandb_config)
            self.wandb_logger.watch_model(model)

    def save_checkpoint(self, model_state, is_best: bool = False, epoch: int = None):
        """Save model checkpoint using config settings"""
        if not hasattr(self, 'config') or self.config is None:
            return
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'config': self.config
        }
        
        # Get checkpoint settings with defaults
        checkpoint_cfg = self.config.experiment.get('checkpointing', {})
        save_dir = Path(checkpoint_cfg.get('dir', 'checkpoints'))
        save_last = checkpoint_cfg.get('save_last', False)
        save_best = checkpoint_cfg.get('save_best', True)
        save_frequency = checkpoint_cfg.get('save_frequency', 0)
        save_frequency_enabled = checkpoint_cfg.get('save_frequency_enabled', False)
        experiment_name = self.config.experiment.get('name', 'default')
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save periodic checkpoint if enabled
        if save_frequency_enabled and epoch is not None and epoch % save_frequency == 0:
            periodic_path = save_dir / f"{experiment_name}_epoch_{epoch}.pt"
            torch.save(checkpoint, periodic_path)
            if hasattr(self, 'wandb_logger'):
                self.wandb_logger.save_model(str(periodic_path))
        
        # Save last checkpoint if enabled
        if save_last:
            last_path = save_dir / f"{experiment_name}_last.pt"
            torch.save(checkpoint, last_path)
            if hasattr(self, 'wandb_logger'):
                self.wandb_logger.save_model(str(last_path))
        
        # Save best checkpoint if enabled and is best
        if save_best and is_best:
            best_path = save_dir / f"{experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
            if hasattr(self, 'wandb_logger'):
                self.wandb_logger.save_model(str(best_path))

    def train(self):
        self.model.to(self.device)
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Log batch metrics
                if hasattr(self, 'wandb_logger') and batch_idx % 100 == 0:  # Log every 100 batches
                    self.wandb_logger.log_batch(
                        batch_idx=batch_idx,
                        epoch=epoch,
                        batch_size=len(self.train_loader),
                        loss=loss.item()
                    )

            avg_loss = total_loss / len(self.train_loader)
            self.logger.log_epoch(epoch, avg_loss)

            # Validation
            val_metrics = self.validator.validate()
            val_loss = val_metrics.pop('loss', float('inf'))
            self.logger.log_validation(epoch, val_loss, val_metrics['f1_score'])

            if hasattr(self, 'wandb_logger'):
                current_lr = self.optimizer.param_groups[0]['lr']
                # Log metrics
                metrics = {
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    **val_metrics,
                    "learning_rate": current_lr
                }
                self.wandb_logger.log_epoch(epoch, metrics)

             # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoints
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(
                model_state=self.model.state_dict(),
                is_best=is_best,
                epoch=epoch
            )
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.finish()
        return self.model

if __name__ == "__main__":
    # Example usage
    pass