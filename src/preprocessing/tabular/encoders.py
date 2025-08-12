import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class MetadataEncoder:
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoders = {}

    def fit_label_encoder(self, column: str, data: pd.Series):
        le = LabelEncoder()
        self.label_encoders[column] = le.fit(data)

    def transform_label(self, column: str, data: pd.Series) -> pd.Series:
        if column in self.label_encoders:
            return self.label_encoders[column].transform(data)
        else:
            raise ValueError(f"Label encoder for {column} not fitted.")

    def fit_one_hot_encoder(self, column: str, data: pd.Series):
        ohe = OneHotEncoder(sparse=False)
        self.one_hot_encoders[column] = ohe.fit(data.values.reshape(-1, 1))

    def transform_one_hot(self, column: str, data: pd.Series) -> pd.DataFrame:
        if column in self.one_hot_encoders:
            return pd.DataFrame(self.one_hot_encoders[column].transform(data.values.reshape(-1, 1)),
                                columns=self.one_hot_encoders[column].get_feature_names_out([column]))
        else:
            raise ValueError(f"One-hot encoder for {column} not fitted.")

    def fit(self, df: pd.DataFrame):
        for column in df.select_dtypes(include=['object']).columns:
            self.fit_label_encoder(column, df[column])
            self.fit_one_hot_encoder(column, df[column])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        for column in df.select_dtypes(include=['object']).columns:
            transformed_df[column] = self.transform_label(column, df[column])
            one_hot_df = self.transform_one_hot(column, df[column])
            transformed_df = pd.concat([transformed_df, one_hot_df], axis=1).drop(column, axis=1)
        return transformed_df