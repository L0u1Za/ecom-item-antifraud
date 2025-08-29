#!/usr/bin/env python3
"""
Temperature Scaling Calibration Script

This script calibrates a trained model using temperature scaling to improve
probability calibration. It loads a trained model, optimizes the temperature
parameter on a validation set, and saves the calibrated model.

Usage:
    python src/calibrate_temperature.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
import pandas as pd
import pickle
from hydra.utils import to_absolute_path

from models.temperature_scaled_model import TemperatureScaledModel, expected_calibration_error
from models.architecture import FraudDetectionModel
from dataset.fraud_dataset import FraudDataset
from dataset.collator import MultiModalCollator
from dataset.processor import TextProcessor, ImageProcessor, TabularProcessor
from utils.logger import setup_logger
from tqdm import tqdm

class TemperatureCalibrator:
    def __init__(self, model, device='cpu', logger=None, config=None):
        self.model = model
        self.device = device
        self.logger = logger or setup_logger('temperature_calibrator', 'calibration.log')
        self.config = config
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Start with reasonable initial value
        
    def collect_logits_and_labels(self, dataloader):
        """Collect logits and labels from validation set"""
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting validation data"):
                inputs, labels = batch
                
                # Move batch to device
                def move_to_device(x):
                    if isinstance(x, dict):
                        return {k: move_to_device(v) for k, v in x.items()}
                    if hasattr(x, 'to'):
                        return x.to(self.device)
                    return x
                
                inputs = move_to_device(inputs)
                labels = labels.to(self.device)
                
                logits, _ = self.model(inputs)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        return torch.cat(all_logits), torch.cat(all_labels)
    
    def optimize_temperature(self, logits, labels):
        """Optimize temperature parameter using NLL loss"""
        max_iter = self.config.calibration.optimization.max_iter
        lr = self.config.calibration.optimization.lr
        
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        
        def eval_loss():
            optimizer.zero_grad()
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits.squeeze(), labels.float()
            )
            loss.backward()
            return loss
        
        initial_loss = eval_loss().item()
        self.logger.info(f"Initial NLL loss: {initial_loss:.4f}")
        
        optimizer.step(eval_loss)
        
        final_loss = eval_loss().item()
        optimal_temp = self.temperature.item()
        
        self.logger.info(f"Optimal temperature: {optimal_temp:.4f}")
        self.logger.info(f"Final NLL loss: {final_loss:.4f}")
        self.logger.info(f"Loss improvement: {initial_loss - final_loss:.4f}")
        
        return optimal_temp
    
    def evaluate_calibration(self, logits, labels, temperature=1.0):
        """Evaluate calibration metrics before and after temperature scaling"""
        # Before temperature scaling
        probs_before = torch.sigmoid(logits).numpy()
        labels_np = labels.numpy()
        
        # After temperature scaling
        scaled_logits = logits / temperature
        probs_after = torch.sigmoid(scaled_logits).numpy()
        
        # Calculate ECE
        n_bins = self.config.calibration.evaluation.n_bins
        ece_before = expected_calibration_error(labels_np, probs_before, n_bins=n_bins)
        ece_after = expected_calibration_error(labels_np, probs_after, n_bins=n_bins)
        
        # Calculate NLL
        nll_before = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), labels.float()
        ).item()
        nll_after = nn.functional.binary_cross_entropy_with_logits(
            scaled_logits.squeeze(), labels.float()
        ).item()
        
        metrics = {
            'ece_before': ece_before,
            'ece_after': ece_after,
            'nll_before': nll_before,
            'nll_after': nll_after,
            'temperature': temperature
        }
        
        return metrics, probs_before, probs_after
    
    def plot_reliability_diagram(self, labels, probs_before, probs_after, save_path=None):
        """Plot reliability diagram comparing before and after calibration"""
        figsize = self.config.calibration.plotting.figsize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        def plot_reliability(ax, y_true, y_prob, title):
            n_bins = self.config.calibration.evaluation.n_bins
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            accuracies = []
            confidences = []
            counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    accuracies.append(accuracy_in_bin)
                    confidences.append(avg_confidence_in_bin)
                    counts.append(in_bin.sum())
                else:
                    accuracies.append(0)
                    confidences.append((bin_lower + bin_upper) / 2)
                    counts.append(0)
            
            # Plot reliability diagram
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            ax.bar(confidences, accuracies, width=0.1, alpha=0.7, 
                   edgecolor='black', label='Model')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plot_reliability(ax1, labels, probs_before, 'Before Temperature Scaling')
        plot_reliability(ax2, labels, probs_after, 'After Temperature Scaling')
        
        plt.tight_layout()
        
        if save_path:
            dpi = self.config.calibration.plotting.dpi
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            self.logger.info(f"Reliability diagram saved to {save_path}")
        
        return fig

def create_validation_dataloader(cfg):
    """Create validation dataloader with processors like choose_threshold.py"""
    # Initialize processors
    text_processor = TextProcessor(**cfg.preprocessing.text)
    image_processor = ImageProcessor(cfg, training=False)
    
    processor_path = to_absolute_path(cfg.processors.tabular_processor_path)
    with open(processor_path, "rb") as f:
        tabular_processor = pickle.load(f)
    
    # Update config with tabular processor info
    cfg.model.model.tabular.categories = tabular_processor.category_cardinalities
    extra_continuous = 1 if cfg.preprocessing.image.get('compute_clip_similarity', False) else 0
    cfg.model.model.tabular.num_continuous = tabular_processor.num_continuous_features + extra_continuous
    
    # Create validation dataset
    val_data_path = to_absolute_path(cfg.experiment.data.val_path)
    val_dataset = FraudDataset(
        data_path=val_data_path,
        image_dir=cfg.experiment.data.train_images_path,
        text_processor=text_processor,
        image_processor=image_processor,
        tabular_processor=tabular_processor
    )
    
    # Create dataloader
    collator = MultiModalCollator(cfg)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.experiment.data.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    return val_dataloader

def load_model_from_checkpoint(checkpoint_path, config, device):
    """Load model from checkpoint"""
    logger = setup_logger('model_loader', 'calibration.log')
    
    # Create model
    model = FraudDetectionModel(config.model, training=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device,  weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    return model

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup device
    if cfg.calibration.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.calibration.device
    
    logger = setup_logger('temperature_calibration', 'calibration.log')
    logger.info(f"Using device: {device}")
    
    # Create validation dataloader with processors
    logger.info("Creating validation dataloader with processors...")
    val_loader = create_validation_dataloader(cfg)
    
    # Load model using training config from unified config
    model = load_model_from_checkpoint(cfg.calibration.model_path, cfg, device)
    
    if val_loader is None:
        logger.error("Validation loader is required for temperature calibration")
        return
    
    # Initialize calibrator
    calibrator = TemperatureCalibrator(model, device, logger, cfg)
    
    # Collect validation logits and labels
    logger.info("Collecting validation data...")
    val_logits, val_labels = calibrator.collect_logits_and_labels(val_loader)
    
    # Optimize temperature
    logger.info("Optimizing temperature parameter...")
    optimal_temperature = calibrator.optimize_temperature(val_logits, val_labels)
    
    # Evaluate calibration
    logger.info("Evaluating calibration metrics...")
    metrics, probs_before, probs_after = calibrator.evaluate_calibration(
        val_logits, val_labels, optimal_temperature
    )
    
    # Log results
    logger.info("=== Calibration Results ===")
    logger.info(f"Optimal Temperature: {metrics['temperature']:.4f}")
    logger.info(f"ECE Before: {metrics['ece_before']:.4f}")
    logger.info(f"ECE After: {metrics['ece_after']:.4f}")
    logger.info(f"ECE Improvement: {metrics['ece_before'] - metrics['ece_after']:.4f}")
    logger.info(f"NLL Before: {metrics['nll_before']:.4f}")
    logger.info(f"NLL After: {metrics['nll_after']:.4f}")
    logger.info(f"NLL Improvement: {metrics['nll_before'] - metrics['nll_after']:.4f}")
    
    # Create output directory
    output_dir = Path(to_absolute_path(cfg.calibration.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot reliability diagram
    fig = calibrator.plot_reliability_diagram(
        val_labels.numpy(), probs_before, probs_after,
        save_path=output_dir / 'reliability_diagram.png'
    )
    
    # Create temperature scaled model
    temp_scaled_model = TemperatureScaledModel(model, optimal_temperature)
    
    # Save calibrated model
    calibrated_model_path = output_dir / 'temperature_scaled_model.pt'
    torch.save({
        'model_state_dict': temp_scaled_model.state_dict(),
        'temperature': optimal_temperature,
        'calibration_metrics': metrics,
        'config': cfg
    }, calibrated_model_path)
    
    logger.info(f"Calibrated model saved to {calibrated_model_path}")
    
    # Save metrics if enabled
    if cfg.calibration.logging.save_metrics:
        metrics_path = output_dir / 'calibration_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("Temperature Scaling Calibration Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Optimal Temperature: {metrics['temperature']:.4f}\n")
            f.write(f"ECE Before: {metrics['ece_before']:.4f}\n")
            f.write(f"ECE After: {metrics['ece_after']:.4f}\n")
            f.write(f"ECE Improvement: {metrics['ece_before'] - metrics['ece_after']:.4f}\n")
            f.write(f"NLL Before: {metrics['nll_before']:.4f}\n")
            f.write(f"NLL After: {metrics['nll_after']:.4f}\n")
            f.write(f"NLL Improvement: {metrics['nll_before'] - metrics['nll_after']:.4f}\n")
        
        logger.info(f"Calibration metrics saved to {metrics_path}")
    
    logger.info("Temperature scaling calibration completed successfully!")

if __name__ == "__main__":
    main()
