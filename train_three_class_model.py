
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

# -----------------------------------------
# 1. File paths and settings
# -----------------------------------------

DATA_PATH = Path(
    r"C:\Users\Deekshith c v\Downloads\signalshield_three_class.csv"
)

MODEL_PATH = Path("signalshield_three_class_model.joblib")

LABELS = ["legitimate", "spam", "phishing"]

# -----------------------------------------
# 2. Load the dataset
# -----------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = {"text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "CSV must contain 'text' and 'label' columns."
    )

# Clean missing and empty records
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["text"] != ""]

# Keep only the expected classes
df = df[df["label"].isin(LABELS)].copy()

if df.empty:
    raise ValueError("No valid labeled messages found.")

if df["label"].nunique() != 3:
    raise ValueError(
        "All three classes must be present in the dataset."
    )

print("=" * 55)
print("SIGNALSHIELD: THREE-CLASS EMAIL DETECTOR")
print("=" * 55)

print("\nDataset loaded successfully!")
print("Total messages:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts().reindex(LABELS))

# -----------------------------------------
# 3. Split into training and testing data
# -----------------------------------------

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# -----------------------------------------
# 4. Build the machine learning pipeline
# -----------------------------------------

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
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# -----------------------------------------
# 5. Train the model
# -----------------------------------------

print("\nTraining model... Please wait.")

model.fit(X_train, y_train)

print("Training completed successfully!")

# -----------------------------------------
# 6. Evaluate the model
# -----------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0
    )
)

print("\nConfusion Matrix")
print("Class order:", LABELS)
print(confusion_matrix(y_test, y_pred, labels=LABELS))

# -----------------------------------------
# 7. Save the trained model
# -----------------------------------------

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Model location:", MODEL_PATH.resolve())

# -----------------------------------------
# 8. Test new messages
# -----------------------------------------

def predict_message(message):
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    print("\nMessage:", message)
    print("Predicted category:", prediction.upper())
    print("Class probability estimates:")

    for label, probability in sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"  {label}: {probability * 100:.2f}%")

    print(
        "\nNote: These are model estimates, "
        "not guaranteed risk probabilities."
    )


print("\n" + "=" * 55)
print("TEST SAMPLE MESSAGES")
print("=" * 55)

test_messages = [
    "Hello, are we still meeting for lunch tomorrow?",
    "Congratulations! You have won a prize. Claim it now!",
    "Your account will be suspended. Verify your password immediately."
]

for message in test_messages:
    predict_message(message)

# -----------------------------------------
# 9. Interactive prediction
# -----------------------------------------

print("\n" + "=" * 55)
print("TRY YOUR OWN MESSAGE")
print("Type 'exit' to stop.")
print("=" * 55)

while True:
    message = input("\nEnter an email or message: ").strip()

    if message.lower() == "exit":
        print("SignalShield stopped.")
        break

    if not message:
        print("Please enter a message.")
        continue

    predict_message(message)
