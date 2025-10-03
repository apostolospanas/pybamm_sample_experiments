import pandas as pd
import matplotlib.pyplot as plt
import glob
import balthazar as blt
from pathlib import Path

# Find all result CSVs in current working dir
csv_files = glob.glob("*.csv")

dfs, labels = [], []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

    # Infer C-rate from filename (assumes naming convention like ..._2C_...csv)
    if "2.5C" in file: labels.append("2.5C")
    elif "2C" in file: labels.append("2C")
    else: labels.append(file)

# Overlay Voltage vs Capacity
plt.figure(figsize=(10,6))
for df, label in zip(dfs, labels):
    plt.plot(df["Discharge capacity [A.h]"], df["Voltage [V]"], label=label)

plt.xlabel("Discharge Capacity [Ah]")
plt.ylabel("Voltage [V]")
plt.title("CCD Comparison Across Runs")
plt.legend(); plt.grid(True)

out_file = Path("compare_ccd_overlay.png")
plt.savefig(out_file, dpi=150)

blt.output["ccd_overlay"] = str(out_file)
