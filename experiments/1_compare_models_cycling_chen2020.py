import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Root Balthazar Runner data directory
DATA_DIR = Path(r"C:\Users\apost\AppData\Local\Balthazar Labs\Balthazar Runner\data")

# Search recursively for CSVs that match your keywords
csv_files = glob.glob(str(DATA_DIR / "**" / "*DFN_Ai2020*.csv"), recursive=True)

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(" -", f)

# Overlay Voltage vs Capacity from all matching runs
plt.figure(figsize=(10, 6))
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if "Discharge capacity [A.h]" in df.columns and "Voltage [V]" in df.columns:
            plt.plot(df["Discharge capacity [A.h]"], df["Voltage [V]"], label=Path(file).parent.name)
    except Exception as e:
        print(f"Error reading {file}: {e}")

plt.xlabel("Discharge Capacity [Ah]")
plt.ylabel("Terminal Voltage [V]")
plt.title("CCD Overlay: DFN + Ai2020")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
