
import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\Deekshith c v\Downloads")

files = [
    "Enron.csv",
    "Ling.csv",
    "Merged_Dataset.csv",
]

targets = [
    "cell phone",
    "cell phones",
    "ii jornadas de linguistica aborigen",
    "get a $25 certificate",
    "thank you for selecting quickenloans",
]

for filename in files:
    path = folder / filename

    if not path.exists():
        print(f"\nSkipping missing file: {filename}")
        continue

    print(f"\n{'=' * 60}")
    print("SOURCE:", filename)

    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="cp1252",
            on_bad_lines="skip"
        )

    if "body" not in df.columns:
        print("No 'body' column. Available:", list(df.columns))
        continue

    for target in targets:
        matches = df[
            df["body"].astype(str).str.contains(
                target,
                case=False,
                regex=False,
                na=False
            )
        ]

        if not matches.empty:
            print(f"\nMATCH: {target}")
            columns = [
                c for c in ["subject", "body", "label"]
                if c in matches.columns
            ]
            print(
                matches[columns]
                .head(2)
                .to_string(index=False, max_colwidth=180)
            )
