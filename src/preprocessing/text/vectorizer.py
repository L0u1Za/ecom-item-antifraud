from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def save_vectorizer(self, filepath):
        import joblib
        joblib.dump(self.vectorizer, filepath)

    def load_vectorizer(self, filepath):
        import joblib
        self.vectorizer = joblib.load(filepath)