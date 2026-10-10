
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Dataset and output paths
DATA_PATH = Path(
    r"C:\Users\Deekshith c v\Downloads\signalshield_three_class.csv"
)

MODEL_PATH = Path("signalshield_improved_model.joblib")
ERROR_PATH = Path("signalshield_misclassified_messages.csv")

LABELS = ["legitimate", "spam", "phishing"]

# 1. Load and validate data
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"]).copy()

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()

df = df[df["text"] != ""]
df = df[df["label"].isin(LABELS)]

# 2. Remove exact duplicate messages.
# If one identical message has conflicting labels, exclude it.
df["normalized_text"] = (
    df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
)

label_counts = df.groupby("normalized_text")["label"].nunique()
conflicting_texts = label_counts[label_counts > 1].index

print("Conflicting messages excluded:", len(conflicting_texts))

df = df[~df["normalized_text"].isin(conflicting_texts)]
df = df.drop_duplicates(subset="normalized_text").copy()

print("\nDataset after cleaning:", len(df))
print(df["label"].value_counts().reindex(LABELS, fill_value=0))

if df["label"].nunique() != 3:
    raise ValueError("All three classes must be present.")

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# 4. Build model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        )
    )
])

# 5. Train
print("\nTraining improved model...")
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)

print("\nAccuracy: {:.2f}%".format(
    accuracy_score(y_test, predictions) * 100
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=LABELS,
    target_names=LABELS,
    zero_division=0
))

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=LABELS))
print("Class order:", LABELS)

# 7. Save errors for review
errors = pd.DataFrame({
    "text": X_test.to_numpy(),
    "actual_label": y_test.to_numpy(),
    "predicted_label": predictions
})

errors = errors[errors["actual_label"] != errors["predicted_label"]]
errors.to_csv(ERROR_PATH, index=False, encoding="utf-8-sig")

print("\nMisclassified messages saved to:")
print(ERROR_PATH.resolve())
print("Number of errors:", len(errors))

# 8. Save to a NEW model file
joblib.dump(model, MODEL_PATH)

print("\nImproved model saved to:")
print(MODEL_PATH.resolve())

# 9. Test messages
def check_message(message):
    prediction = model.predict([message])[0]
    scores = model.predict_proba([message])[0]

    print("\nMessage:", message)
    print("Predicted class:", prediction.upper())
    print("Model scores:")

    for label, score in sorted(
        zip(model.classes_, scores),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"  {label}: {score * 100:.2f}%")

    print("These scores are not calibrated risk probabilities.")

samples = [
    "Hello, are we still meeting for lunch tomorrow?",
    "Congratulations! You have won a prize. Claim it now!",
    "Your account will be suspended. Verify your password immediately.",
    "Your package is delayed. Please check the delivery status."
]

print("\nSample predictions:")
for sample in samples:
    check_message(sample)

# 10. Interactive testing
print("\nEnter a message to test, or type exit to stop.")

while True:
    message = input("\nMessage: ").strip()

    if message.lower() == "exit":
        break

    if message:
        check_message(message)
    else:
        print("Please enter a message.")
