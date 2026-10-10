
import pandas as pd

path = r"C:\Users\Deekshith c v\Downloads\PhishingEmailData.csv"

df = pd.read_csv(path, encoding="cp1252")

print("Total records:", len(df))
print("Duplicate email contents:",
      df["Email_Content"].duplicated().sum())

print("\nEmail subject counts:")
print(df["Email_Subject"].value_counts().head(10))

print("\nSample subjects and content:")
for _, row in df[
    ["Email_Subject", "Email_Content"]
].head(10).iterrows():
    print("\nSubject:", str(row["Email_Subject"])[:120])
    print("Content:", str(row["Email_Content"])[:250])

print("\nPossible source labels or categories:")
for col in df.columns:
    if any(word in col.lower() for word in
           ["label", "class", "category", "type"]):
        print(col, df[col].value_counts(dropna=False))
