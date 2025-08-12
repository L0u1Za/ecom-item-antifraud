import re
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure that the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def normalize_text(text):
    """
    Normalize the input text by lowercasing, removing special characters,
    and lemmatizing the words.
    
    Args:
        text (str): The input text to normalize.
        
    Returns:
        str: The normalized text.
    """
    # Lowercase the text
    text = text.lower()
    
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Lemmatize each word
    normalized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a single string
    normalized_text = ' '.join(normalized_words)
    
    return normalized_text