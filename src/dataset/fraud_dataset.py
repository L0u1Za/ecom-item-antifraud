from torch.utils.data import Dataset
from typing import Dict, Any, Tuple
import torch

class FraudDataset(Dataset):
    def __init__(self, 
                 text_processor=None,
                 image_processor=None,
                 tabular_processor=None):
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.tabular_processor = tabular_processor
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        item = self.data[idx]
        
        processed = {
            'text': self.text_processor(item['text']) if self.text_processor else None,
            'image': self.image_processor(item['image']) if self.image_processor else None,
            'tabular': self.tabular_processor(item['tabular']) if self.tabular_processor else None
        }
        
        label = torch.tensor(item['label'], dtype=torch.float32)
        return processed, label