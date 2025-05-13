# models/text_cleaner.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # allow list inputs
        if not hasattr(X, "str"):
            X = pd.Series(X)
        return (
            X.str.lower()
             .str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
             .str.replace(r"<.*?>", "", regex=True)
             .str.replace(r"[^a-z\s]", "", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
        )
