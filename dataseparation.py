import pandas as pd
from pathlib import Path

# Load all sheets as a dictionary
all_sheets = pd.read_excel("bigdataset.xlsx", sheet_name=None)

out_dir = Path("curve_pairs")
out_dir.mkdir(exist_ok=True)

curve_id = 1  # To number the output files

for sheet_name, df in all_sheets.items():

    # Drop columns where the first 10 data rows are all empty
    df_cleaned = df.loc[:, df.iloc[:10].notna().any()]

    # Convert to numeric (non-numeric becomes NaN)
    df_numeric = df_cleaned.apply(pd.to_numeric, errors="coerce")
    cols = df_numeric.columns

    for i in range(0, len(cols), 2):
        pair = df_numeric.loc[:, cols[i:i + 2]].copy()

        # Ensure we have exactly 2 columns
        if pair.shape[1] != 2:
            continue

        # Replace empty strings or whitespace-only with NaN
        pair.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

        # Drop rows with any NaN (including those that were blank strings)
        pair = pair.dropna()
        print(pair.shape)
        # Skip if fewer than 5 complete rows
        if pair.shape[0] < 7:
            print("Skipping")
            continue

        # Rename columns
        pair.columns = ["Strain", "Stress"]

        # Save cleaned, valid dataset
        pair.to_excel(out_dir / f"{curve_id}.xlsx", index=False)
        curve_id += 1

