```python
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "messages_merged.csv"
MODEL_PATH = BASE / "models" / "message_classifier.joblib"

LABELS = {
    "legitimate": "Legitimate",
    "spam": "Spam",
    "phishing": "Phishing",
}

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.lower().str.strip()
df = df[
    df["text"].ne("")
    & df["label"].isin(LABELS)
].drop_duplicates(subset=["text"])

print("Dataset rows:", len(df))
print("\nClass counts:")
print(df["label"].value_counts().to_string())

class_counts = df["label"].value_counts()

for label in LABELS:
    if class_counts.get(label, 0) < 5:
        raise ValueError(
            f"Not enough examples for '{label}'. "
            "Please check the dataset before training."
        )

print("\nDataset validation passed.")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("classifier", LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        )),
    ])

evaluation_model = make_pipeline()
evaluation_model.fit(X_train, y_train)
predictions = evaluation_model.predict(X_test)



print("\n=== Phishing-specific evaluation ===")

phishing_true = (y_test == "phishing")
phishing_pred = (predictions == "phishing")

phishing_recall = recall_score(
    phishing_true,
    phishing_pred,
    zero_division=0,
)

phishing_precision = precision_score(
    phishing_true,
    phishing_pred,
    zero_division=0,
)

phishing_f1 = f1_score(
    phishing_true,
    phishing_pred,
    zero_division=0,
)

print(f"Phishing recall:    {phishing_recall:.1%}")
print(f"Phishing precision: {phishing_precision:.1%}")
print(f"Phishing F1-score:  {phishing_f1:.1%}")
print(f"Actual phishing messages in test set: {phishing_true.sum()}")

print("\n=== Held-out evaluation (20% test set) ===")
print(classification_report(
    y_test,
    predictions,
    labels=["legitimate", "spam", "phishing"],
    target_names=["Legitimate", "Spam", "Phishing"],
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted)")
print("Label order: legitimate, spam, phishing")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["legitimate", "spam", "phishing"],
))

final_model = make_pipeline()
final_model.fit(df["text"], df["label"])

bundle = {
    "pipeline": final_model,
    "labels": LABELS,
    "training_rows": int(len(df)),
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODEL_PATH)

print("\nFinal model retrained on all", len(df), "messages.")
print("Saved to:", MODEL_PATH)
print("Evaluation metrics above are from the held-out split.")
```