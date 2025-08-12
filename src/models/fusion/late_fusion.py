import numpy as np

class LateFusion:
    def __init__(self, models):
        self.models = models

    def predict(self, inputs):
        predictions = [model.predict(inputs) for model in self.models]
        return self.combine_predictions(predictions)

    def combine_predictions(self, predictions):
        # Average the predictions from all models
        return np.mean(predictions, axis=0)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)