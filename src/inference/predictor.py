import torch
from src.models.classifier import Classifier
from src.inference.cache_manager import CacheManager

class Predictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = Classifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.cache_manager = CacheManager()

    def predict(self, features):
        with torch.no_grad():
            features = self.cache_manager.get_cached_features(features)
            inputs = self.prepare_inputs(features)
            outputs = self.model(inputs)
            predictions = torch.sigmoid(outputs)
            return predictions

    def prepare_inputs(self, features):
        # Implement any necessary preprocessing for the inputs
        return features