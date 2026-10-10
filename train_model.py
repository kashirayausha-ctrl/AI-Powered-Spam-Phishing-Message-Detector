"""Train the baseline three-class text classifier from data/messages.csv."""
from pathlib import Path
from model_utils import ensure_model

BASE = Path(__file__).resolve().parent
bundle = ensure_model(BASE / "models" / "message_classifier.joblib", BASE / "data" / "messages.csv")
print(f"Model trained/loaded. Rows: {bundle.get('training_rows', 'saved model')}")
print("Saved to models/message_classifier.joblib")
