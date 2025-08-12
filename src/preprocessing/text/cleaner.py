import re
from typing import List, Optional
import unicodedata
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
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
    
    def clean_text(self, 
                  text: str, 
                  remove_stopwords: bool = False,
                  lemmatize: bool = True,
                  extract_contacts: bool = True) -> dict:
        """
        Clean text with all necessary preprocessing steps
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to apply lemmatization
            extract_contacts: Whether to extract contact information
            
        Returns:
            Dictionary containing cleaned text and extracted information
        """
        # Store original text for contact extraction
        original_text = text
        
        # Clean HTML and normalize spaces
        text = self.clean_html(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize Unicode characters
        text = self.normalize_unicode(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if requested
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text