from .text.cleaner import clean_text
from .text.normalizer import normalize_text
from .text.vectorizer import vectorize_text
from .text.business_rules import apply_business_rules
from .image.augmentations import augment_images
from .image.embeddings import generate_image_embeddings
from .image.clip_validator import validate_image_with_clip
from .tabular.encoders import encode_metadata
from .tabular.metadata_processor import process_metadata

class PreprocessingPipeline:
    def __init__(self):
        pass

    def preprocess_text(self, text):
        cleaned_text = clean_text(text)
        normalized_text = normalize_text(cleaned_text)
        vectorized_text = vectorize_text(normalized_text)
        return apply_business_rules(vectorized_text)

    def preprocess_image(self, image):
        augmented_image = augment_images(image)
        image_embedding = generate_image_embeddings(augmented_image)
        is_valid = validate_image_with_clip(augmented_image)
        return image_embedding, is_valid

    def preprocess_tabular(self, metadata):
        encoded_metadata = encode_metadata(metadata)
        processed_metadata = process_metadata(encoded_metadata)
        return processed_metadata

    def run_pipeline(self, text, image, metadata):
        processed_text = self.preprocess_text(text)
        processed_image, image_validity = self.preprocess_image(image)
        processed_metadata = self.preprocess_tabular(metadata)
        return processed_text, processed_image, processed_metadata, image_validity