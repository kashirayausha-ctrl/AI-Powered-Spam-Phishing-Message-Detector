
from pathlib import Path
import pandas as pd
import re

DOWNLOADS = Path(r"C:\Users\Deekshith c v\Downloads")

# Change this to the actual verified SpamAssassin source CSV.
SPAMASSASSIN_FILE = DOWNLOADS / "SpamAssassin.csv"

OUTPUT_FILE = DOWNLOADS / "signalshield_three_class_validated.csv"
AUDIT_FILE = DOWNLOADS / "signalshield_dataset_audit.csv"

# These mappings must match the documentation for the exact source files.
# For Enron/Ling, use only label 0 as legitimate.
# Do NOT map their label 1 to spam or phishing.
ENRON_SAFE_LABEL = 0
LING_SAFE_LABEL = 0

# Configure these only after checking SpamAssassin's own label documentation.
SPAM_TEXT_COLUMN = "body"
SPAM_LABEL_COLUMN = "label"

# Example mapping ONLY if the file's documentation confirms it:
# SPAMASSASSIN_LABEL_MAP = {"ham": "legitimate", "spam": "spam"}
# For numeric labels, use the documented values, not assumptions.
SPAMASSASSIN_LABEL_MAP = None


def normalize(text):
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    return pd.read_csv(path, encoding_errors="replace")


def get_text(df):
    # Support known layouts without inventing missing content.
    if "subject" in df.columns and "body" in df.columns:
        return (
            df["subject"].fillna("").astype(str)
            + "\n"
            + df["body"].fillna("").astype(str)
        ).str.strip()
    if "body" in df.columns:
        return df["body"].fillna("").astype(str).str.strip()
    if "text" in df.columns:
        return df["text"].fillna("").astype(str).str.strip()
    raise ValueError(f"No recognized text columns: {list(df.columns)}")


records = []

# 1. Enron: only documented safe class
enron_path = DOWNLOADS / "Enron.csv"
if enron_path.exists():
    enron = load_csv(enron_path)
    if not {"label", "body"}.issubset(enron.columns):
        raise ValueError("Enron.csv must contain body and label columns.")

    safe = enron[enron["label"].astype(str) == str(ENRON_SAFE_LABEL)].copy()
    for text in get_text(safe):
        if text.strip():
            records.append({
                "text": text,
                "label": "legitimate",
                "source": "Enron",
                "source_label": str(ENRON_SAFE_LABEL)
            })

# 2. Ling: only documented safe class
ling_path = DOWNLOADS / "Ling.csv"
if ling_path.exists():
    ling = load_csv(ling_path)
    if not {"label", "body"}.issubset(ling.columns):
        raise ValueError("Ling.csv must contain body and label columns.")

    safe = ling[ling["label"].astype(str) == str(LING_SAFE_LABEL)].copy()
    for text in get_text(safe):
        if text.strip():
            records.append({
                "text": text,
                "label": "legitimate",
                "source": "Ling",
                "source_label": str(LING_SAFE_LABEL)
            })

# 3. Nazario: phishing-only, if every row's provenance is verified
nazario_path = DOWNLOADS / "Nazario.csv"
if nazario_path.exists():
    nazario = load_csv(nazario_path)
    for text in get_text(nazario):
        if text.strip():
            records.append({
                "text": text,
                "label": "phishing",
                "source": "Nazario",
                "source_label": "verified_phishing_corpus"
            })

# 4. SpamAssassin: stop unless its labels are verified
if SPAMASSASSIN_FILE.exists():
    spamassassin = load_csv(SPAMASSASSIN_FILE)
    print("\nSpamAssassin columns:", list(spamassassin.columns))
    if SPAMASSASSIN_LABEL_MAP is None:
        raise ValueError(
            "Verify SpamAssassin's label encoding first. "
            "Inspect the source documentation, then set "
            "SPAMASSASSIN_LABEL_MAP."
        )

    if SPAM_TEXT_COLUMN not in spamassassin.columns:
        raise ValueError(f"Missing text column: {SPAM_TEXT_COLUMN}")
    if SPAM_LABEL_COLUMN not in spamassassin.columns:
        raise ValueError(f"Missing label column: {SPAM_LABEL_COLUMN}")

    for _, row in spamassassin.iterrows():
        raw_label = str(row[SPAM_LABEL_COLUMN]).strip().lower()
        if raw_label not in SPAMASSASSIN_LABEL_MAP:
            continue

        label = SPAMASSASSIN_LABEL_MAP[raw_label]
        if label not in {"legitimate", "spam"}:
            raise ValueError(f"Unexpected SpamAssassin label: {label}")

        text = str(row[SPAM_TEXT_COLUMN]).strip()
        if text and text.lower() != "nan":
            records.append({
                "text": text,
                "label": label,
                "source": "SpamAssassin",
                "source_label": raw_label
            })
else:
    print(
        "\nSpamAssassin source CSV is missing. "
        "No spam labels will be fabricated."
    )

if not records:
    raise ValueError("No usable source records found.")

df = pd.DataFrame(records)
df["normalized_text"] = df["text"].map(normalize)
df = df[df["normalized_text"] != ""].copy()

# If identical text has different proposed classes, quarantine it.
label_counts = df.groupby("normalized_text")["label"].nunique()
conflicts = set(label_counts[label_counts > 1].index)

audit = df[df["normalized_text"].isin(conflicts)].copy()
audit["audit_status"] = "conflicting_labels_excluded"
audit.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

df = df[~df["normalized_text"].isin(conflicts)].copy()

# Deduplicate identical messages, preserving the first source record.
df = df.drop_duplicates(subset="normalized_text", keep="first")

df = df[["text", "label", "source", "source_label"]].reset_index(drop=True)

print("\nValidated candidate class counts:")
print(df["label"].value_counts())
print("\nCounts by source and class:")
print(pd.crosstab(df["source"], df["label"]))

present = set(df["label"])
required = {"legitimate", "spam", "phishing"}

if present != required:
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    raise ValueError(
        f"Dataset saved for inspection, but classes are incomplete. "
        f"Found: {sorted(present)}. Required: {sorted(required)}. "
        f"Do not train a three-class model yet."
    )

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nDataset saved:", OUTPUT_FILE)
print("Conflict audit saved:", AUDIT_FILE)
print("\nAll three classes are present. Review source provenance")
print("and class counts before training.")
