import wandb
from typing import Dict, Any

class WandBLogger:
    def __init__(self, project_name: str, config: Dict[str, Any] = None):
        """Initialize WandB logger
        Args:
            project_name: Name of the project in WandB
            config: Configuration dictionary for the experiment
        """
        self.config = config or {}
        wandb.init(project=project_name, config=self.config)
    
    def watch_model(self, model):
        """Watch model gradients"""
        wandb.watch(model)
    
    def log_batch(self, batch_idx: int, epoch: int, batch_size: int, loss: float):
        """Log batch-level metrics"""
        wandb.log({
            "batch": batch_idx + epoch * batch_size,
            "batch_loss": loss
        })
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        metrics["epoch"] = epoch
        wandb.log(metrics)
    
    def save_model(self, path: str):
        """Save model artifact"""
        wandb.save(path)
    
    def finish(self):
        """Finish the WandB run"""
        wandb.finish()