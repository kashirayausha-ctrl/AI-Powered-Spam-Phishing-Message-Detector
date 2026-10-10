from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from model_utils import analyze_message, ensure_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "message_classifier.joblib"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024  # 32 KB request limit

_model_bundle = None


def get_model():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = ensure_model(MODEL_PATH, BASE_DIR / "data" / "messages.csv")
    return _model_bundle


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        get_model()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception:
        app.logger.exception("Model health check failed")
        return jsonify({"status": "error", "model_loaded": False}), 503


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object containing message and optional sender."}), 400

    message = str(payload.get("message", "")).strip()
    sender = str(payload.get("sender", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a message to analyze."}), 400
    if len(message) > 12000:
        return jsonify({"error": "Message is too long. Maximum length is 12,000 characters."}), 413
    if len(sender) > 320:
        return jsonify({"error": "Sender field is too long."}), 400

    try:
        result = analyze_message(message, sender, get_model())
        return jsonify(result)
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify({"error": "Analysis failed. Check the server logs and try again."}), 500


@app.post("/api/feedback")
def feedback():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid feedback payload."}), 400

    message = str(payload.get("message", "")).strip()
    predicted = str(payload.get("predicted_label", "")).strip()
    corrected = str(payload.get("corrected_label", "")).strip().lower()
    if not message or len(message) > 12000:
        return jsonify({"error": "A valid message is required."}), 400
    if predicted not in {"Legitimate", "Spam", "Phishing"}:
        return jsonify({"error": "Invalid predicted label."}), 400
    if corrected not in {"legitimate", "spam", "phishing"}:
        return jsonify({"error": "Choose Legitimate, Spam, or Phishing."}), 400

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp_utc", "message", "predicted_label", "corrected_label"
        ])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "predicted_label": predicted,
            "corrected_label": corrected,
        })
    return jsonify({"status": "saved", "message": "Thanks. Your feedback was saved locally."})


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Request too large."}), 413


if __name__ == "__main__":
    # Development server only. Use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
