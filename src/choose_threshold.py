import hydra
import pandas as pd
import torch
import pickle
import json
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import BinaryPrecisionRecallCurve

from dataset.fraud_dataset import FraudDataset
from dataset.collator import MultiModalCollator
from dataset.processor import TextProcessor, ImageProcessor, TabularProcessor
from inference.predictor import Predictor
import numpy as np

@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Starting threshold selection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize processors
    print("Initializing processors...")
    text_processor = TextProcessor(**cfg.preprocessing.text)
    image_processor = ImageProcessor(cfg, training=False)
    
    processor_path = to_absolute_path(cfg.processors.tabular_processor_path)
    with open(processor_path, "rb") as f:
        tabular_processor = pickle.load(f)
    print(f"Loaded fitted TabularProcessor from {processor_path}")

    cfg.model.model.tabular.categories = tabular_processor.category_cardinalities
    extra_continuous = 1 if cfg.preprocessing.image.get('compute_clip_similarity', False) else 0
    cfg.model.model.tabular.num_continuous = tabular_processor.num_continuous_features + extra_continuous

    # 2. Create dataset and dataloader for validation set
    print("Creating dataset and dataloader for val.csv...")
    val_data_path = to_absolute_path(cfg.experiment.data.val_path)
    val_df = pd.read_csv(val_data_path)
    targets = torch.tensor(val_df['resolution'].values, dtype=torch.float32)

    val_dataset = FraudDataset(
        data_path=val_data_path,
        image_dir=cfg.experiment.data.train_images_path,
        text_processor=text_processor,
        image_processor=image_processor,
        tabular_processor=tabular_processor
    )

    collator = MultiModalCollator(cfg)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.experiment.data.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # 3. Initialize predictor
    print("Initializing predictor...")
    model_path = to_absolute_path(cfg.inference.model_path)
    predictor = Predictor(cfg=cfg.model, model_path=model_path, device=device)

    # 4. Get predictions for the validation set
    print("Generating predictions for validation set...")
    logits = []
    with torch.no_grad():
        for batch, label in tqdm(val_dataloader, desc="Generating validation predictions"):
            batch_preds = predictor.predict_proba(batch)
            logits.append(batch_preds)
    logits = torch.cat(logits)

    # Ensure logits and targets are on the same device and have the same shape
    logits = logits.squeeze().to(device)
    targets = targets.to(device)

    # 5. Calculate Precision-Recall curve and find best threshold
    print("Calculating Precision-Recall curve...")
    pr_curve = BinaryPrecisionRecallCurve().to(device)
    precision, recall, thresholds = pr_curve(logits, targets.int())

    # Move to CPU for numpy operations
    precision = precision.cpu().numpy()
    recall = recall.cpu().numpy()
    thresholds = thresholds.cpu().numpy()

    # Handle the case where thresholds might not include 0 and 1
    if len(thresholds) == len(precision) - 1:
        thresholds = np.concatenate(([0], thresholds))

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1_score = f1_scores[best_f1_idx]

    print(f"\nBest F1 Score: {best_f1_score:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")

    # 6. Save optimal threshold
    threshold_data = {
        "threshold": float(best_threshold),
        "f1_score": float(best_f1_score),
        "precision": float(precision[best_f1_idx]),
        "recall": float(recall[best_f1_idx])
    }
    
    threshold_path = to_absolute_path(cfg.threshold.optimal_threshold_path)
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Saved optimal threshold to {threshold_path}")

    # 7. Plot and save the PR curve
    print("Plotting PR curve...")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='PR Curve')
    plt.scatter(recall[best_f1_idx], precision[best_f1_idx], marker='o', color='red', label=f'Best F1 (Thresh={best_threshold:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    output_path = to_absolute_path(cfg.inference.pr_curve_path)
    plt.savefig(output_path)
    print(f"Saved PR curve plot to {output_path}")


if __name__ == "__main__":
    main()
