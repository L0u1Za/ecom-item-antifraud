import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Validator:
    def __init__(self, model, dataloader, device, criterion, threshold=0.5):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold
        self.criterion = criterion

    def validate(self, threshold=None):
        """
        Validate model with specified threshold
        Args:
            threshold: Optional override for instance threshold
        Returns:
            Dictionary containing metrics including loss
        """
        if threshold is not None:
            self.threshold = threshold

        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for data in self.dataloader:
                inputs, labels = data
                # Move nested dict to device
                def move_to_device(x):
                    if isinstance(x, dict):
                        return {k: move_to_device(v) for k, v in x.items()}
                    if hasattr(x, 'to'):
                        return x.to(self.device)
                    return x
                inputs = move_to_device(inputs)
                labels = labels.to(self.device)

                outputs, _ = self.model(inputs)
                probabilities = torch.sigmoid(outputs.squeeze(-1))  # For binary classification

                # Calculate loss
                loss = self.criterion(outputs.squeeze(-1), labels)
                total_loss += loss.item()

                all_probs.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
         # Calculate average loss
        avg_loss = total_loss / len(self.dataloader)
        
        # Apply threshold to get predictions
        predictions = (all_probs >= self.threshold).astype(int)

        # Get metrics and add loss
        metrics = self.calculate_metrics(all_labels, predictions)
        metrics['loss'] = avg_loss

        return metrics

    def calculate_metrics(self, true_labels, predictions):
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='binary')
        recall = recall_score(true_labels, predictions, average='binary')
        f1 = f1_score(true_labels, predictions, average='binary')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': self.threshold
        }

    def find_best_threshold(self, thresholds=None):
        """
        Find the best threshold based on F1 score
        Args:
            thresholds: List of thresholds to try. Default: np.arange(0.1, 0.9, 0.1)
        Returns:
            best_threshold, best_metrics
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.1)

        best_f1 = 0
        best_threshold = 0.5
        best_metrics = None

        for threshold in thresholds:
            metrics = self.validate(threshold=threshold)
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics