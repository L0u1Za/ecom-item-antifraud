import re
from typing import List, Optional
import unicodedata
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextCleaner:
    def __init__(self, nltk_data_dir):
        # Download required NLTK data
        nltk.download('punkt', nltk_data_dir)
        nltk.download('stopwords', nltk_data_dir)
        nltk.download('wordnet', nltk_data_dir)
        
        self.stop_words = set(stopwords.words('russian') + stopwords.words('english'))
        
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and hidden characters"""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove hidden characters
        text = ' '.join(text.split())
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to their closest ASCII representation"""
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    def clean_repeating_chars(self, text: str) -> str:
        """Clean repeating character sequences"""
        # Replace repeating punctuation with max 3 occurrences
        text = re.sub(r'([!?.])\1{2,}', r'\1' * 3, text)
        # Replace other repeating chars with max 2 occurrences
        text = re.sub(r'([^!?.])\1{2,}', r'\1\1', text)
        return text

    def truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length at word boundary"""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at last sentence
        truncated = text[:max_length]
        last_sentence = re.search(r'.*[.!?]', truncated)
        if last_sentence:
            return last_sentence.group(0)
        
        # If no sentence boundary, truncate at last word
        last_word = re.search(r'.*\s', truncated)
        if last_word:
            return last_word.group(0)
        
        return truncated
    
    def clean_text(self, 
                  text: str, 
                  remove_stopwords: bool = False) -> dict:
        """
        Clean text with all necessary preprocessing steps
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to apply lemmatization
            
        Returns:
            Dictionary containing cleaned text and extracted information
        """
        # Clean HTML and normalize spaces
        text = self.clean_html(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize Unicode characters
        # text = self.normalize_unicode(text)
        
        # Tokenize
        tokens = word_tokenize(text, language='russian')
        
        # Remove stop words if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join tokens back into text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text