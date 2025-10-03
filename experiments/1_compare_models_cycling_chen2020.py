import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import balthazar as blt

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = r"C:\Users\apost\AppData\Local\Balthazar Labs\Balthazar Runner\data"
MODEL_NAME = "DFN"
PARAM_SET = "Ai2020"
N_CYCLES = 5
OUT_DIR = os.getcwd()
PDF_FILE = os.path.join(OUT_DIR, f"compare_{MODEL_NAME}_{PARAM_SET}.pdf")
CSV_FILE = os.path.join(OUT_DIR, f"compare_{MODEL_NAME}_{PARAM_SET}_summary.csv")

# ---------------------------
# LOAD DATA FROM MULTIPLE RUNS
# ---------------------------
csv_files = glob.glob(os.path.join(BASE_DIR, "**", f"results_{MODEL_NAME}_{PARAM_SET}.csv"), recursive=True)
runs = {}

for f in csv_files:
    parent = os.path.basename(os.path.dirname(f))
    # Try to find C-rate from path string, else fallback
    label = parent
    if "2C" in parent:
        label = "2C"
    elif "2.5C" in parent:
        label = "2.5C"
    runs[label] = pd.read_csv(f)

print(f"Found {len(runs)} runs: {list(runs.keys())}")

# ---------------------------
# PLOTTING OVERLAYS
# ---------------------------
pdf = PdfPages(PDF_FILE)
summary_rows = []

# 1) Voltage vs Time
fig1, ax1 = plt.subplots(figsize=(10,6))
for label, df in runs.items():
    ax1.plot(df["Time [h]"], df["Voltage [V]"], label=label)
    summary_rows.append([label, df["Discharge capacity [A.h]"].max(),
                         df["Voltage [V]"].iloc[-1], df["Discharge energy [W.h]"].iloc[-1]])
ax1.set_xlabel("Time [h]"); ax1.set_ylabel("Voltage [V]")
ax1.set_title("Voltage vs Time across runs")
ax1.legend(); ax1.grid(True); fig1.tight_layout()
pdf.savefig(fig1); plt.close(fig1)

# 2) CCD (Voltage vs Capacity, Cycle 1)
fig2, ax2 = plt.subplots(figsize=(10,6))
for label, df in runs.items():
    npts = len(df) // N_CYCLES
    first_cycle = df.iloc[:npts]
    ax2.plot(first_cycle["Discharge capacity [A.h]"], first_cycle["Voltage [V]"], label=label)
ax2.set_xlabel("Discharge Capacity [Ah]"); ax2.set_ylabel("Voltage [V]")
ax2.set_title("CCD Curve (Cycle 1)")
ax2.legend(); ax2.grid(True); fig2.tight_layout()
pdf.savefig(fig2); plt.close(fig2)

# 3) Capacity Retention vs Cycle
fig3, ax3 = plt.subplots(figsize=(8,5))
for label, df in runs.items():
    npts = len(df) // N_CYCLES
    cycle_ends = [df["Discharge capacity [A.h]"].iloc[min((i+1)*npts - 1, len(df)-1)]
                  for i in range(N_CYCLES)]
    ax3.plot(range(1, N_CYCLES+1), cycle_ends, marker='o', label=label)
ax3.set_xlabel("Cycle"); ax3.set_ylabel("End-of-Discharge Capacity [Ah]")
ax3.set_title("Capacity Retention vs Cycle")
ax3.legend(); ax3.grid(True); fig3.tight_layout()
pdf.savefig(fig3); plt.close(fig3)

# 4) Energy vs Time
fig4, ax4 = plt.subplots(figsize=(8,5))
for label, df in runs.items():
    ax4.plot(df["Time [h]"], df["Discharge energy [W.h]"], label=label)
ax4.set_xlabel("Time [h]"); ax4.set_ylabel("Discharge Energy [Wh]")
ax4.set_title("Energy vs Time across runs")
ax4.legend(); ax4.grid(True); fig4.tight_layout()
pdf.savefig(fig4); plt.close(fig4)

# Close PDF
pdf.close()

# ---------------------------
# SAVE SUMMARY TABLE
# ---------------------------
summary_df = pd.DataFrame(summary_rows, columns=["Run Label", "Max Capacity [Ah]", "Final Voltage [V]", "Final Energy [Wh]"])
summary_df.to_csv(CSV_FILE, index=False)

# Push to Balthazar outputs
blt.output["comparison_pdf"] = PDF_FILE
blt.output["comparison_csv"] = CSV_FILE
blt.output["summary_table"] = summary_df

print(f"Saved report: {PDF_FILE}")
print(f"Saved summary: {CSV_FILE}")
