import hydra
import pandas as pd
import torch
import pickle
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.fraud_dataset import InferenceDataset
from dataset.collator import MultiModalCollatorTest
from dataset.processor import TextProcessor, ImageProcessor, TabularProcessor
from inference.predictor import Predictor


@hydra.main(config_path="../src/config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Starting inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize processors
    print("Initializing processors...")
    text_processor = TextProcessor(**cfg.preprocessing.text)
    image_processor = ImageProcessor(cfg, training=False)
    
    # Load the fitted tabular processor
    processor_path = to_absolute_path(cfg.processors.tabular_processor_path)
    with open(processor_path, "rb") as f:
        tabular_processor = pickle.load(f)
    print(f"Loaded fitted TabularProcessor from {processor_path}")

    # Inject learned cardinalities and continuous count into cfg for model init
    cfg.model.model.tabular.categories = tabular_processor.category_cardinalities
    extra_continuous = 1 if cfg.preprocessing.image.get('compute_clip_similarity', False) else 0
    cfg.model.model.tabular.num_continuous = tabular_processor.num_continuous_features + extra_continuous

    # 2. Create dataset and dataloader
    print("Creating dataset and dataloader...")
    test_dataset = InferenceDataset(
        data_path=to_absolute_path(cfg.experiment.data.test_path),
        text_processor=text_processor,
        image_processor=image_processor,
        tabular_processor=tabular_processor
    )

    collator = MultiModalCollatorTest(cfg)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size * 2,  # Use a larger batch size for inference
        shuffle=False,
        num_workers=cfg.experiment.data.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # 3. Initialize predictor
    # Note: You need to provide a path to your trained model checkpoint.
    # This can be done via config override: +inference.model_path="path/to/your/model.pth"
    print("Initializing predictor...")
    predictor = Predictor(cfg=cfg.model, model_path=to_absolute_path(cfg.inference.model_path), device=device)

    # 4. Get predictions
    print("Generating predictions...")
    predictions = []
    item_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating predictions"):
            batch_preds = predictor.predict(batch).cpu().numpy().flatten()
            predictions.extend(batch_preds)
            item_ids.extend(batch["item_id"]) # item_id is now a list/numpy array, not a tensor

    # 5. Create submission file
    submission_df = pd.DataFrame({
        'id': item_ids,
        'prediction': predictions
    })

    submission_df.to_csv(to_absolute_path(cfg.inference.output_path), index=False)
    print("Inference complete. Submission file created at submission.csv")


if __name__ == "__main__":
    main()
