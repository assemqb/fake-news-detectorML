# scripts/train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

from models.text_cleaner import TextCleaner

def main():
    os.makedirs("models", exist_ok=True)
    fake = pd.read_csv("data/Fake.csv"); fake["label"] = 0
    real = pd.read_csv("data/True.csv"); real["label"] = 1
    df   = pd.concat([fake, real], ignore_index=True).sample(frac=1, random_state=42)

    X, y = df["text"], df["label"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf",   TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df=5)),
        ("clf",     XGBClassifier(eval_metric="logloss", random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    for name, Xs, ys in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        preds = pipeline.predict(Xs)
        probs = pipeline.predict_proba(Xs)[:,1]
        print(f"\n--- {name} Metrics ---")
        print(classification_report(ys, preds, target_names=["FAKE","REAL"]))
        print("ROC-AUC:", roc_auc_score(ys, probs))
        print("Confusion Matrix:\n", confusion_matrix(ys, preds))

    joblib.dump(pipeline, "models/fake_news_pipeline.joblib")
    print("\n>> Pipeline saved to models/fake_news_pipeline.joblib")

if __name__ == "__main__":
    main()
