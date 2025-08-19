from torch.utils.data import Dataset
from typing import Dict, Any, Tuple
import torch
import pandas as pd

class FraudDataset(Dataset):
    def __init__(self, 
                 data_path,
                 text_processor=None,
                 image_processor=None,
                 tabular_processor=None):
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.tabular_processor = tabular_processor
        
        self.data = pd.read_csv(data_path, index_col=0)
        self.data['brand_name'] = self.data['brand_name'].astype('str')
        self.data['description'] = self.data['description'].astype('str')
        self.data['name_rus'] = self.data['name_rus'].astype('str')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        item = self.data.iloc[idx, ]
        
        text_obj = {'title': item['name_rus'], 'description': item['description'], 'brand_name': item['brand_name']}
        # Keep only relevant tabular columns: categorical + numerical from processor
        # Also exclude target label if present
        drop_cols = ['description', 'name_rus']
        if 'resolution' in item.index:
            drop_cols.append('label')
        tabular_obj = item.drop(drop_cols, errors='ignore').to_dict()
        images_obj = {
            "title": item['name_rus'],
            "images": None
        }
        
        processed = {
            'text': self.text_processor(text_obj) if self.text_processor else None,
            'images': self.image_processor(images_obj) if self.image_processor else None,
            'tabular': self.tabular_processor(tabular_obj) if self.tabular_processor else None
        }
        
        label = torch.tensor(item['resolution'], dtype=torch.float32)
        return processed, label