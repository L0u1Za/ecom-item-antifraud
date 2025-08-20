import torch
from omegaconf import DictConfig

from models.architecture import FraudDetectionModel
import numpy as np

class Predictor:
    def __init__(self, cfg: DictConfig, model_path: str, threshold=0.5, device: str = "cpu"):
        self.device = device
        self.model = FraudDetectionModel(cfg, training=False)  # Set training=False
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = threshold

    def predict(self, batch):
        with torch.no_grad():
            batch = self._move_to_device(batch)
            logits, _ = self.model(batch)
            predictions = np.array(torch.sigmoid(logits.squeeze(-1)))
            predictions = (predictions >= self.threshold).astype(int)
            return predictions

    def _move_to_device(self, batch):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    if isinstance(batch[key][sub_key], torch.Tensor):
                        batch[key][sub_key] = batch[key][sub_key].to(self.device)
        return batch