from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


LABELS = {"legitimate": "Legitimate", "spam": "Spam", "phishing": "Phishing"}

URL_RE = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"']+""")
IP_URL_RE = re.compile(r"""(?i)\bhttps?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?""")
URGENT_TERMS = {
    "urgent", "immediately", "suspended", "suspension", "verify", "verification",
    "expire", "expired", "act now", "limited time", "final warning", "within 24 hours",
}
CREDENTIAL_TERMS = {
    "password", "otp", "one-time password", "login", "log in", "credentials",
    "bank details", "card number", "pin", "security code",
}
MONEY_TERMS = {
    "winner", "won", "prize", "claim your", "cash prize", "lottery", "free gift",
    "investment guaranteed", "earn money", "reward",
}
THREAT_TERMS = {
    "account will be closed", "account suspended", "legal action", "penalty",
    "unusual activity", "confirm your identity",
}


def _url_host(url: str) -> str:
    candidate = url if "://" in url else "http://" + url
    try:
        return (urlparse(candidate).hostname or "").lower()
    except ValueError:
        return ""


def extract_indicators(message: str, sender: str = "") -> list[dict]:
    text = message.lower()
    indicators = []

    urls = URL_RE.findall(message)
    if urls:
        indicators.append({
            "name": "Contains link(s)",
            "detail": f"Found {len(urls)} URL-like link(s). Links are not automatically malicious; inspect the destination carefully.",
            "severity": "medium",
        })

    if IP_URL_RE.search(message):
        indicators.append({
            "name": "Link uses an IP address",
            "detail": "A URL points directly to a numeric IP address, which can be suspicious in unsolicited messages.",
            "severity": "high",
        })

    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "ow.ly"}
    for url in urls:
        host = _url_host(url.rstrip(".,);]"))
        if host in shorteners:
            indicators.append({
                "name": "Shortened URL",
                "detail": f"The link uses {host}, which hides the final destination until expanded.",
                "severity": "medium",
            })
            break

    matched_urgent = sorted(term for term in URGENT_TERMS if term in text)
    if matched_urgent:
        indicators.append({
            "name": "Urgency or pressure language",
            "detail": "Matched: " + ", ".join(matched_urgent[:5]) + ". Pressure can be used to rush unsafe decisions.",
            "severity": "medium",
        })

    matched_credentials = sorted(term for term in CREDENTIAL_TERMS if term in text)
    if matched_credentials:
        indicators.append({
            "name": "Sensitive information request or reference",
            "detail": "Matched: " + ", ".join(matched_credentials[:5]) + ". Never share passwords, PINs, or OTPs through a message link.",
            "severity": "high",
        })

    matched_money = sorted(term for term in MONEY_TERMS if term in text)
    if matched_money:
        indicators.append({
            "name": "Prize, reward, or money bait",
            "detail": "Matched: " + ", ".join(matched_money[:5]) + ". Unexpected rewards and guaranteed returns deserve scrutiny.",
            "severity": "medium",
        })

    matched_threats = sorted(term for term in THREAT_TERMS if term in text)
    if matched_threats:
        indicators.append({
            "name": "Threat or account consequence",
            "detail": "Matched: " + ", ".join(matched_threats[:5]) + ". Threats can be used to provoke hurried action.",
            "severity": "medium",
        })

    sender_lower = sender.lower()
    if sender and ("@" not in sender or sender_lower.endswith(("@gmail.com", "@yahoo.com", "@outlook.com"))):
        if any(word in text for word in ("bank", "account", "security", "support", "delivery", "tax")):
            indicators.append({
                "name": "Sender identity may need verification",
                "detail": "The supplied sender is incomplete or a general mailbox while the message claims an organization-related purpose.",
                "severity": "low",
            })

    if message.count("!") >= 3 or message.isupper() and len(message) > 12:
        indicators.append({
            "name": "Unusual emphasis",
            "detail": "Repeated exclamation marks or all-caps wording can indicate pressure or promotional content.",
            "severity": "low",
        })

    return indicators


def ensure_model(model_path: Path, data_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            if "pipeline" in bundle and "labels" in bundle:
                return bundle
        except Exception:
            pass

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    frame = pd.read_csv(data_path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset must contain text and label columns.")
    frame = frame.dropna(subset=["text", "label"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.lower().str.strip()
    frame = frame[frame["label"].isin(LABELS)]
    frame = frame[frame["text"].str.len() > 0]
    if frame["label"].nunique() < 3:
        raise ValueError("Training data must include legitimate, spam, and phishing examples.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            max_features=25000, sublinear_tf=True, min_df=1
        )),
        ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42))
    ])
    pipeline.fit(frame["text"], frame["label"])
    bundle = {"pipeline": pipeline, "labels": LABELS, "training_rows": int(len(frame))}
    joblib.dump(bundle, model_path)
    return bundle


def analyze_message(message: str, sender: str, bundle: dict) -> dict:
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba([message])[0]
    classes = list(pipeline.classes_)
    prob_map = {LABELS.get(label, label.title()): float(probabilities[i]) for i, label in enumerate(classes)}
    predicted_raw = classes[int(probabilities.argmax())]
    predicted = LABELS.get(predicted_raw, predicted_raw.title())
    indicators = extract_indicators(message, sender)

    # Combine the model estimate with transparent indicators. This is a heuristic score,
    # not a calibrated probability of compromise.
    high = sum(1 for item in indicators if item["severity"] == "high")
    medium = sum(1 for item in indicators if item["severity"] == "medium")
    heuristic_points = min(35, high * 14 + medium * 7)
    model_points = round(prob_map.get("Phishing", 0) * 65 + prob_map.get("Spam", 0) * 38)
    risk_score = max(0, min(100, int(round(max(model_points, heuristic_points)))))
    if predicted == "Phishing" or (high >= 2) or (risk_score >= 68):
        risk_level = "High"
    elif predicted == "Spam" or risk_score >= 35 or indicators:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if not indicators:
        indicators.append({
            "name": "No obvious rule-based warning found",
            "detail": "No common warning pattern was matched. This does not prove the message is safe.",
            "severity": "low",
        })

    explanation = {
        "Legitimate": "The text model found patterns more similar to legitimate examples in its training set.",
        "Spam": "The text model found patterns more similar to unsolicited or promotional examples in its training set.",
        "Phishing": "The text model found patterns more similar to credential-theft or impersonation examples in its training set.",
    }[predicted]

    return {
        "label": predicted,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": round(max(probabilities) * 100, 1),
        "class_probabilities": {key: round(value * 100, 1) for key, value in prob_map.items()},
        "explanation": explanation,
        "indicators": indicators,
        "disclaimer": "This is an AI-assisted screening result, not a guarantee. Do not click unexpected links or share credentials; verify important messages through an independently obtained official channel.",
    }
