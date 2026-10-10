import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app


def test_health():
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["model_loaded"] is True


def test_empty_message_rejected():
    client = app.test_client()
    response = client.post("/api/analyze", json={"message": "   "})
    assert response.status_code == 400


def test_analysis_returns_required_fields():
    client = app.test_client()
    response = client.post("/api/analyze", json={
        "message": "URGENT verify your password immediately at http://example.test/login",
        "sender": "alerts@example.test"
    })
    assert response.status_code == 200
    body = response.json
    for key in ("label", "risk_score", "risk_level", "confidence",
                "class_probabilities", "explanation", "indicators", "disclaimer"):
        assert key in body
    assert 0 <= body["risk_score"] <= 100
    assert body["label"] in {"Legitimate", "Spam", "Phishing"}


def test_bad_payload_rejected():
    client = app.test_client()
    response = client.post("/api/analyze", data="not json", content_type="text/plain")
    assert response.status_code == 400
