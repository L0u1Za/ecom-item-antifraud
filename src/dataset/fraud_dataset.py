import os
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple
import torch
import pandas as pd

class FraudDataset(Dataset):
    def __init__(self, 
                 data_path,
                 image_dir=None,
                 text_processor=None,
                 image_processor=None,
                 tabular_processor=None):
        self.image_dir = image_dir
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
        item_id = item['ItemID'] # Assuming ItemID is the index
        
        text_obj = {'title': item['name_rus'], 'description': item['description'], 'brand_name': item['brand_name']}
        # Keep only relevant tabular columns: categorical + numerical from processor
        # Also exclude target label if present
        drop_cols = ['description', 'name_rus']
        if 'resolution' in item.index:
            drop_cols.append('label')
        tabular_obj = item.drop(drop_cols, errors='ignore').to_dict()
        
        image_path = os.path.join(self.image_dir, f"{item_id}.png") if self.image_dir else None

        processed = {
            'text': self.text_processor(text_obj) if self.text_processor else None,
            'images': self.image_processor(image_path) if self.image_processor and image_path and os.path.exists(image_path) else self.image_processor.get_empty_image(),
            'tabular': self.tabular_processor(tabular_obj) if self.tabular_processor else None
        }
        
        label = torch.tensor(item['resolution'], dtype=torch.float32)
        return processed, label


class InferenceDataset(Dataset):
    def __init__(self, 
                 data_path,
                 image_dir=None,
                 text_processor=None,
                 image_processor=None,
                 tabular_processor=None):
        self.image_dir = image_dir
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.tabular_processor = tabular_processor
        
        self.data = pd.read_csv(data_path)
        # Ensure correct dtypes
        self.data['brand_name'] = self.data['brand_name'].astype('str')
        self.data['description'] = self.data['description'].astype('str')
        self.data['name_rus'] = self.data['name_rus'].astype('str')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data.iloc[idx, ]
        item_id = item['ItemID']
        
        text_obj = {'title': item['name_rus'], 'description': item['description'], 'brand_name': item['brand_name']}
        drop_cols = ['description', 'name_rus']
        tabular_obj = item.drop(drop_cols, errors='ignore').to_dict()

        image_path = os.path.join(self.image_dir, f"{item_id}.png") if self.image_dir else None

        processed = {
            'item_id': item['id'],
            'text': self.text_processor(text_obj) if self.text_processor else None,
            'images': self.image_processor(image_path) if self.image_processor and image_path and os.path.exists(image_path) else self.image_processor.get_empty_image(),
            'tabular': self.tabular_processor(tabular_obj) if self.tabular_processor else None
        }
        
        return processed