import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Path where Balthazar stores all run folders
BASE_DIR = r"C:\Users\apost\AppData\Local\Balthazar Labs\Balthazar Runner\data"

# Find all result CSVs
csv_files = glob.glob(os.path.join(BASE_DIR, "**", "results_DFN_Ai2020.csv"), recursive=True)

runs = {}
for f in csv_files:
    # Derive run label (e.g. parent folder might include "2C" or "2.5C")
    parent = os.path.basename(os.path.dirname(f))
    label = f"DFN_{parent}"   # customise: parse "2C"/"2.5C" if in folder name
    
    try:
        df = pd.read_csv(f)
        runs[label] = df
    except Exception as e:
        print(f"Skipping {f}: {e}")

# ---------------------------
# PLOTTING
# ---------------------------

# 1) Voltage vs Time
plt.figure(figsize=(10,6))
for label, df in runs.items():
    plt.plot(df["Time [h]"], df["Voltage [V]"], label=label)
plt.xlabel("Time [h]"); plt.ylabel("Voltage [V]")
plt.title("Voltage vs Time across C-rates")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("compare_voltage_vs_time.png", dpi=150)
plt.show()

# 2) Voltage vs Discharge Capacity (Cycle 1 only)
plt.figure(figsize=(10,6))
for label, df in runs.items():
    npts = len(df) // 5  # crude: divide by n_cycles (5)
    first_cycle = df.iloc[:npts]
    plt.plot(first_cycle["Discharge capacity [A.h]"], first_cycle["Voltage [V]"], label=label)
plt.xlabel("Discharge Capacity [Ah]"); plt.ylabel("Voltage [V]")
plt.title("CCD Curve (Cycle 1) across C-rates")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("compare_voltage_vs_capacity.png", dpi=150)
plt.show()

# 3) Capacity Retention vs Cycle
plt.figure(figsize=(8,5))
for label, df in runs.items():
    n_cycles = 5
    npts = len(df) // n_cycles
    cycle_ends = [df["Discharge capacity [A.h]"].iloc[min((i+1)*npts - 1, len(df)-1)]
                  for i in range(n_cycles)]
    plt.plot(range(1, n_cycles+1), cycle_ends, marker='o', label=label)
plt.xlabel("Cycle"); plt.ylabel("End-of-Discharge Capacity [Ah]")
plt.title("Capacity Retention vs Cycle across C-rates")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("compare_capacity_vs_cycle.png", dpi=150)
plt.show()

# 4) Energy delivered per run
plt.figure(figsize=(8,5))
for label, df in runs.items():
    plt.plot(df["Time [h]"], df["Discharge energy [W.h]"], label=label)
plt.xlabel("Time [h]"); plt.ylabel("Discharge Energy [Wh]")
plt.title("Discharge Energy vs Time across C-rates")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("compare_energy_vs_time.png", dpi=150)
plt.show()
