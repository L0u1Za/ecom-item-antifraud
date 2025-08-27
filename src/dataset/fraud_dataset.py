import os
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple
import torch
import pandas as pd
from hydra.utils import to_absolute_path

class FraudDataset(Dataset):
    def __init__(self, 
                 data_path,
                 image_dir=None,
                 text_processor=None,
                 image_processor=None,
                 tabular_processor=None,
                 model_config=None):
        self.image_dir = to_absolute_path(image_dir) if image_dir else None
        self.text_processor = text_processor if (model_config is None or model_config.text.enabled) else None
        self.image_processor = image_processor if (model_config is None or model_config.image.enabled) else None
        self.tabular_processor = tabular_processor if (model_config is None or model_config.tabular.enabled) else None
        
        self.data = pd.read_csv(data_path, index_col=0)
        self.data['brand_name'] = self.data['brand_name'].astype('str')
        self.data['description'] = self.data['description'].astype('str')
        self.data['name_rus'] = self.data['name_rus'].astype('str')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        item = self.data.iloc[idx, ]
        item_id = int(item['ItemID']) # Assuming ItemID is the index
        
        text_obj = {'title': item['name_rus'], 'description': item['description'], 'brand_name': item['brand_name']}
        # Keep only relevant tabular columns: categorical + numerical from processor
        # Also exclude target label if present
        drop_cols = ['description', 'name_rus']
        if 'resolution' in item.index:
            drop_cols.append('resolution')
        tabular_obj = item.drop(drop_cols, errors='ignore').to_dict()
        
        image_path = os.path.join(self.image_dir, f"{item_id}.png") if self.image_dir else None

        processed = {}
        processed['text'] = self.text_processor(text_obj) if self.text_processor else None
    
        processed['images'] = self.image_processor(image_path, text_obj['title']) if self.image_processor else None
    
        processed['tabular'] = self.tabular_processor(tabular_obj) if self.tabular_processor else None
        
        label = torch.tensor(item['resolution'], dtype=torch.float32)
        return processed, label

    def get_image_path(self, idx: int):
        if not self.image_dir:
            return None
        item = self.data.iloc[idx, ]
        item_id = int(item['ItemID'])
        image_path = os.path.join(self.image_dir, f"{item_id}.png")
        return image_path if os.path.exists(image_path) else None


class InferenceDataset(Dataset):
    def __init__(self, 
                 data_path,
                 image_dir=None,
                 text_processor=None,
                 image_processor=None,
                 tabular_processor=None,
                 model_config=None):
        self.image_dir = to_absolute_path(image_dir) if image_dir else None
        self.text_processor = text_processor if (model_config is None or model_config.text.enabled) else None
        self.image_processor = image_processor if (model_config is None or model_config.image.enabled) else None
        self.tabular_processor = tabular_processor if (model_config is None or model_config.tabular.enabled) else None
        
        self.data = pd.read_csv(data_path)
        # Ensure correct dtypes
        self.data['brand_name'] = self.data['brand_name'].astype('str')
        self.data['description'] = self.data['description'].astype('str')
        self.data['name_rus'] = self.data['name_rus'].astype('str')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data.iloc[idx, ]
        item_id = int(item[0])
        
        text_obj = {'title': item['name_rus'], 'description': item['description'], 'brand_name': item['brand_name']}
        drop_cols = ['description', 'name_rus']
        if 'resolution' in item.index:
            drop_cols.append('resolution')
        tabular_obj = item.drop(drop_cols, errors='ignore').to_dict()

        image_path = os.path.join(self.image_dir, f"{item_id}.png") if self.image_dir else None

        processed = {'item_id': item['id']}

        processed['text'] = self.text_processor(text_obj) if self.text_processor else None
    
        processed['images'] = self.image_processor(image_path, text_obj['title']) if self.image_processor else None
    
        processed['tabular'] = self.tabular_processor(tabular_obj) if self.tabular_processor else None
        
        return processed

    def get_image_path(self, idx: int):
        if not self.image_dir:
            return None
        item = self.data.iloc[idx, ]
        item_id = int(item['id'])
        image_path = os.path.join(self.image_dir, f"{item_id}.png")
        return image_path if os.path.exists(image_path) else None