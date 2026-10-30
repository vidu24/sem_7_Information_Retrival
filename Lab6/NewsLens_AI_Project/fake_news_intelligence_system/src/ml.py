import re
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils import shuffle
from .config import MODEL_PATH

REAL_LABELS = {"real", "true", "reliable", "1", "genuine", "mostly-real"}
FAKE_LABELS = {"fake", "false", "misleading", "0", "unreliable"}

def normalize_label(value):
    s = str(value).strip().lower()
    if s in REAL_LABELS: return "REAL"
    if s in FAKE_LABELS: return "FAKE"
    return None

def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000,
                                  min_df=1, max_df=0.98, strip_accents="unicode",
                                  sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1200, class_weight="balanced"))
    ])

def train_model(df, text_col="text", label_col="label"):
    data = df[[text_col, label_col]].dropna().copy()
    data["target"] = data[label_col].map(normalize_label)
    data = data[data["target"].notna()]
    data[text_col] = data[text_col].astype(str).str.strip()
    data = data[data[text_col].str.len() >= 20]
    if data["target"].nunique() != 2 or len(data) < 4:
        raise ValueError("Training requires at least 4 usable rows and both REAL and FAKE labels.")
    X, y = data[text_col].tolist(), data["target"].tolist()
    counts = data["target"].value_counts()
    can_test = len(data) >= 10 and counts.min() >= 2
    metrics = {"rows": len(data), "evaluation": "No holdout metrics: dataset too small for a reliable split."}
    if can_test:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = make_pipeline()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary", pos_label="FAKE", zero_division=0)
        metrics = {
            "rows": len(data), "test_rows": len(yte),
            "accuracy": float(accuracy_score(yte, pred)),
            "precision_fake": float(p), "recall_fake": float(r), "f1_fake": float(f),
            "confusion_matrix": confusion_matrix(yte, pred, labels=["REAL","FAKE"]).tolist(),
            "evaluation": "Stratified 80/20 holdout; metrics are dataset-dependent."
        }
    final_model = make_pipeline()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    return final_model, metrics

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def predict(model, text):
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    idx = int(np.argmax(probs))
    label = classes[idx]
    confidence = float(probs[idx])
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {str(c): float(p) for c,p in zip(classes, probs)},
        "caution": "This is a statistical text-pattern prediction, not a fact-check or proof of truth."
    }
