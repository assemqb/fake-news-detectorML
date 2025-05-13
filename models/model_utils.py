# models/model_utils.py

import joblib
from typing import Tuple

_PIPELINE = joblib.load("models/fake_news_pipeline.joblib")

def predict_text(text: str) -> Tuple[str, float]:
    prob_real = _PIPELINE.predict_proba([text])[0][1]
    label     = "REAL" if prob_real >= 0.5 else "FAKE"
    return label, float(prob_real)
