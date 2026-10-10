
from pathlib import Path
import pandas as pd

folder = Path(r"C:\Users\Deekshith c v\Downloads")

files = [
    "Enron.csv",
    "Ling.csv",
    "Merged_Dataset.csv",
    "Nazario (1).csv",
    "Nazario.csv",
    "PhishingEmailData.csv"
]

for filename in files:
    path = folder / filename

    print("\n" + "=" * 50)
    print("FILE:", filename)

    if not path.exists():
        print("NOT FOUND")
        continue

    try:
        df = pd.read_csv(path, nrows=100000, low_memory=False)
        print("COLUMNS:", df.columns.tolist())
        print("ROWS INSPECTED:", len(df))

        for col in df.columns:
            if any(word in col.lower() for word in
                   ["label", "class", "type", "category"]):
                print("\nLABEL COUNTS:", col)
                print(df[col].value_counts(dropna=False).head(10))

        for col in df.columns:
            if col.lower() in ["body", "text", "message"]:
                print("\nSAMPLE TEXT:")
                for item in df[col].dropna().head(2):
                    print(str(item)[:100].replace("\n", " "))
                break

    except Exception as e:
        print("ERROR:", e)
