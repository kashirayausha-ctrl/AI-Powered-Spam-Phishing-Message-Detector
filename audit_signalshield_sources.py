
from pathlib import Path
import pandas as pd

folder = Path(r"C:\Users\Deekshith c v\Downloads")

files = [
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nazario (1).csv",
]

for name in files:
    path = folder / name
    if not path.exists():
        print(f"\nMissing: {name}")
        continue

    df = pd.read_csv(path, low_memory=False)
    print(f"\n--- {name} ---")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "label" in df.columns:
        print("Labels:")
        print(df["label"].value_counts(dropna=False).to_dict())

# Compare Nazario copies
a = folder / "Nazario.csv"
b = folder / "Nazario (1).csv"

if a.exists() and b.exists():
    first = pd.read_csv(a, low_memory=False)
    second = pd.read_csv(b, low_memory=False)

    print("\n--- Nazario file comparison ---")
    print("Same columns:", list(first.columns) == list(second.columns))
    print("Same row count:", len(first) == len(second))
    print("Identical data:", first.equals(second))

# Check whether the two small collections overlap
for name in ["Enron.csv", "Ling.csv"]:
    path = folder / name
    if path.exists():
        df = pd.read_csv(path, usecols=["body", "label"])
        print(f"\n{name}: label counts")
        print(df["label"].value_counts(dropna=False).to_dict())
