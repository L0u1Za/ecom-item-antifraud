from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Union
import numpy as np

class TextVectorizer:
    def __init__(self,
                 max_features: int = 30000,          # Increased from 10000
                 min_df: Union[int, float] = 2,      # Minimum document frequency
                 max_df: Union[int, float] = 0.95,   # Maximum document frequency
                 ngram_range: tuple = (1, 3),        # Word n-grams
                 char_ngram_range: tuple = (2, 4),   # Character n-grams
                 max_char_features: int = 10000):    # Max character features
        
        # Word-level vectorizer
        self.word_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            analyzer='word',
            strip_accents='unicode'
        )
        
        # Character-level vectorizer
        self.char_vectorizer = TfidfVectorizer(
            max_features=max_char_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=char_ngram_range,
            analyzer='char',
            strip_accents='unicode'
        )
        
        self.is_fitted = False
        self._feature_names = None
    
    def fit(self, texts: List[str]):
        """Fit both word and character vectorizers"""
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(texts)
        self.is_fitted = True
        
        # Cache feature names
        word_features = self.word_vectorizer.get_feature_names_out()
        char_features = self.char_vectorizer.get_feature_names_out()
        self._feature_names = np.concatenate([word_features, char_features])
        
        return self
    
    def transform(self, texts: List[str]):
        """Transform texts using both vectorizers"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
            
        # Get both word and character features
        word_features = self.word_vectorizer.transform(texts)
        char_features = self.char_vectorizer.transform(texts)
        
        # Horizontally stack the features
        return np.hstack([word_features.toarray(), char_features.toarray()])
    
    def fit_transform(self, texts: List[str]):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> np.ndarray:
        """Get combined feature names from both vectorizers"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self._feature_names
    
    def get_important_features(self, 
                             texts: List[str], 
                             top_n: int = 10) -> List[dict]:
        """Get most important features for each text"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting important features")
            
        vectors = self.transform(texts)
        feature_names = self.get_feature_names()
        
        important_features = []
        for vector in vectors:
            # Get indices of top N features
            top_indices = np.argsort(vector)[-top_n:]
            
            # Create feature-value pairs
            features = {
                feature_names[idx]: float(vector[idx])
                for idx in top_indices
                if vector[idx] > 0
            }
            important_features.append(features)
            
        return important_features
    
    def save_vectorizer(self, filepath: str):
        """Save both vectorizers"""
        import joblib
        state = {
            'word_vectorizer': self.word_vectorizer,
            'char_vectorizer': self.char_vectorizer,
            'is_fitted': self.is_fitted,
            'feature_names': self._feature_names
        }
        joblib.dump(state, filepath)
    
    def load_vectorizer(self, filepath: str):
        """Load both vectorizers"""
        import joblib
        state = joblib.load(filepath)
        self.word_vectorizer = state['word_vectorizer']
        self.char_vectorizer = state['char_vectorizer']
        self.is_fitted = state['is_fitted']
        self._feature_names = state['feature_names']