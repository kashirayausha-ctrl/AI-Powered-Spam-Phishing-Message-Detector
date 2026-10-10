
import pandas as pd

file_path = r"C:\Users\Deekshith c v\Downloads\PhishingEmailData.csv"

for encoding in ["utf-8", "cp1252", "latin1"]:
    try:
        df = pd.read_csv(file_path, encoding=encoding)

        print("Encoding used:", encoding)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("\nFirst five rows:")
        print(df.head().to_string(max_cols=5, max_colwidth=100))

        for col in df.columns:
            if "label" in col.lower() or "class" in col.lower():
                print("\nLabel counts:")
                print(df[col].value_counts(dropna=False))

        break

    except UnicodeDecodeError:
        continue
else:
    print("Could not decode the file using tested encodings.")
