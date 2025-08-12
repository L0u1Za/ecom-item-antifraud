import unittest
from src.preprocessing.pipeline import PreprocessingPipeline

class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = PreprocessingPipeline()

    def test_text_cleaning(self):
        raw_text = "<p>This is a test item description.</p>"
        cleaned_text = self.pipeline.clean_text(raw_text)
        self.assertEqual(cleaned_text, "This is a test item description.")

    def test_image_augmentation(self):
        image = "path/to/image.jpg"
        augmented_image = self.pipeline.augment_image(image)
        self.assertIsNotNone(augmented_image)

    def test_metadata_encoding(self):
        metadata = {"item_name": "Test Item", "category": "Test Category"}
        encoded_metadata = self.pipeline.encode_metadata(metadata)
        self.assertIn("encoded_item_name", encoded_metadata)

    def test_pipeline_integration(self):
        item_data = {
            "title": "<p>Test Item</p>",
            "description": "<p>This is a test item description.</p>",
            "image": "path/to/image.jpg",
            "metadata": {"item_name": "Test Item", "category": "Test Category"}
        }
        processed_data = self.pipeline.process(item_data)
        self.assertIn("cleaned_title", processed_data)
        self.assertIn("augmented_image", processed_data)
        self.assertIn("encoded_metadata", processed_data)

if __name__ == '__main__':
    unittest.main()