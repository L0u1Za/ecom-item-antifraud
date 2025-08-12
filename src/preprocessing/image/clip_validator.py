import torch

class CLIPValidator:
    def __init__(self, clip_model, preprocess):
        self.clip_model = clip_model
        self.preprocess = preprocess

    def validate(self, image, text_description):
        image_input = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_description)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        cosine_similarity = (image_features @ text_features.T).squeeze().cpu().numpy()
        return cosine_similarity

    def is_valid(self, image_path, text_description, threshold=0.5):
        cosine_similarity = self.validate(image_path, text_description)
        return cosine_similarity >= threshold