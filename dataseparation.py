import pandas as pd
from pathlib import Path

# Load all sheets as a dictionary
all_sheets = pd.read_excel("dataseparationtest.xlsx", sheet_name=None)

out_dir = Path("curve_pairs")
out_dir.mkdir(exist_ok=True)

sheet_num = 1  # To keep track of which sheet we're on
curve_id = 1   # To number the output files

for sheet_name, df in all_sheets.items():
    mask = df.apply(pd.to_numeric, errors="coerce").notna().any()
    df_numeric = df.loc[:, mask]
    cols = df_numeric.columns

    for i in range(0, len(cols), 2):
        pair = df_numeric.loc[:, cols[i:i+2]].copy()
        if pair.shape[1] < 2:
            continue  # Skip incomplete pairs
        pair.columns = ["Strain", "Stress"]
        pair.to_excel(out_dir / f"{curve_id}.xlsx", index=False)
        curve_id += 1

    sheet_num += 1