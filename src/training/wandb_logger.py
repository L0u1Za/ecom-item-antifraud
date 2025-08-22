import wandb
from typing import Dict, Any, Tuple, List
import torch
from torch.utils.data import Dataset


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

    def log_predictions_table(self, epoch: int, probs: List[float], labels: List[int], dataset: Dataset, num_examples: int = 10):
        """Log a table of model predictions to wandb"""
        # Create a wandb.Table
        columns = ["item_id", "prediction", "ground_truth", "title", "description", "image"]
        table = wandb.Table(columns=columns)

        # Limit the number of examples to log
        num_examples = min(num_examples, len(probs))

        for i, (prob, label) in enumerate(zip(probs[:num_examples], labels[:num_examples])):
            # Fetch the original data item
            item = dataset.data.iloc[i]
            item_id = int(item['ItemID'])
            title = item.get('title', '')
            description = item.get('description', '')

            # Get image if available
            image_path = dataset.get_image_path(i)
            if image_path:
                image = wandb.Image(image_path)
            else:
                image = None

            # Add data to the table
            table.add_data(
                item_id,
                prob,
                label,
                title,
                description,
                image
            )

        # Log the table
        wandb.log({f"epoch_{epoch}_predictions": table})
    
    def save_model(self, path: str):
        """Save model artifact"""
        wandb.save(path)
    
    def finish(self):
        """Finish the WandB run"""
        wandb.finish()