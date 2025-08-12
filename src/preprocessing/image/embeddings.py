import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageEmbedder:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def generate_embedding(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(image)
        return embedding.squeeze().numpy()

def cache_embeddings(embeddings, cache_path):
    import numpy as np
    np.save(cache_path, embeddings)

def load_embeddings(cache_path):
    import numpy as np
    return np.load(cache_path)