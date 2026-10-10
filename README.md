# SignalShield — AI-Powered Spam & Phishing Message Detector

A complete academic prototype for Problem Statement 32. It uses a Python/Flask backend, a responsive HTML/CSS/JavaScript frontend, a three-class text classifier, explainable indicators, risk scoring, and human feedback.

## Features mapped to the problem statement

- Classifies messages as **Legitimate**, **Spam**, or **Phishing**.
- Analyzes text with TF-IDF word/phrase features and Logistic Regression.
- Checks multiple signals: URL presence, IP-address URLs, shortened URLs, urgency, credential requests, reward bait, threats, sender details, and unusual emphasis.
- Displays class probabilities, model confidence, a 0–100 screening score, risk level, and plain-language explanations.
- Supports difficult cases such as legitimate messages with links and phishing written in professional language. A link alone does not automatically mean phishing.
- Lets a human correct the label. Feedback is appended to `data/feedback.csv`.
- Includes example messages, API validation, size limits, and basic tests.

## Project structure

```text
.
├── app.py
├── model_utils.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── messages.csv
│   └── feedback.csv                 # created when feedback is submitted
├── models/
│   └── message_classifier.joblib    # generated when needed; ignored by Git
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
└── tests/
    └── test_api.py
```

The repository root is the application directory. Development and analysis scripts or artifacts may also be present in the root.

## Requirements

- Python 3.10+ recommended
- pip
- Internet is **not required for backend analysis**. The stylesheet includes an optional Google Fonts import; system fonts remain available if offline.

## Run on Windows

Open PowerShell or the VS Code terminal in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```

If PowerShell blocks activation, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.\.venv\Scripts\python.exe app.py`.

Open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl+C`.

## Run on macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open http://127.0.0.1:5000.

## API

### `POST /api/analyze`

Request JSON:

```json
{
  "message": "Urgent: verify your password at http://example.test",
  "sender": "security@example.test"
}
```

Response fields: `label`, `risk_score`, `risk_level`, `confidence`, `class_probabilities`, `explanation`, `indicators`, and `disclaimer`.

### `POST /api/feedback`

Request JSON:

```json
{
  "message": "Example message",
  "predicted_label": "Phishing",
  "corrected_label": "legitimate"
}
```

Feedback is stored locally as CSV. In a real deployment, add access control, retention limits, consent, encryption, and a protected database. Messages may contain personal information; this prototype deliberately does not transmit them to a third-party service.

## Tests

```bash
python -m pytest -q
```

## Important dataset and evaluation note

`data/messages.csv` is a **small, hand-curated demonstration dataset created for this prototype**, not a validated production corpus. It is useful to prove the end-to-end workflow, but its sample count and wording are too limited to claim real-world accuracy. The reported confidence is the model's maximum class probability, not guaranteed correctness. The risk score is a transparent heuristic and is **not** a calibrated probability of compromise.

For a credible evaluation:
1. Replace or augment the seed data with a licensed, documented dataset containing legitimate, spam, and phishing examples.
2. Remove duplicates and redact personal information.
3. Use a stratified train/test split and preferably cross-validation; keep near-duplicate messages out of both train and test sets.
4. Report per-class precision, recall, F1, confusion matrix, and false-positive rate, especially for legitimate messages.
5. Tune thresholds on a validation set, then evaluate once on a held-out test set.
6. Version the dataset and model. Do not automatically retrain on user feedback without review.

## Limitations and safety

- Text-only analysis cannot confirm the true destination or reputation of a live URL.
- Sender spoofing, image-only messages, attachments, QR codes, and domain reputation are not fully evaluated here.
- A professional-sounding phishing message may evade the baseline classifier; an ordinary message may contain a suspicious-looking link.
- Do not click a link to test it. Verify a sender using contact details obtained independently.
- Use only synthetic or authorized messages during demos. Never paste passwords, OTPs, private customer data, or confidential mail.
- This is an educational screening prototype, not an enterprise anti-phishing product.

## Demo plan (3–4 minutes)

1. Open the local app and explain the three labels.
2. Run the **Legitimate** example; discuss why a link alone is not enough to mark phishing.
3. Run the **Spam** example; show promotional and reward language indicators.
4. Run the **Phishing** example; show urgency, credential request, and suspicious URL indicators.
5. Correct a result using the feedback buttons; show `data/feedback.csv`.
6. Explain model probabilities versus the separate heuristic risk score and mention the seed-data limitation.
