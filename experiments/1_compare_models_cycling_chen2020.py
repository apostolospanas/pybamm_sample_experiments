import pandas as pd
import matplotlib.pyplot as plt
import balthazar as blt
from pathlib import Path

# ---------------------------
# 1) Query all relevant runs by tags
# ---------------------------
runs = blt.query_runs(tags=["Ai2020", "DFN", "Pouch Cell", "CCD"])

dfs, labels = [], []

for run in runs:
    # Look for the CSV artifact in each run
    csv_artifacts = [a for a in run.artifacts if a.endswith("results_DFN_Ai2020.csv")]
    if not csv_artifacts:
        continue

    df = pd.read_csv(csv_artifacts[0])
    dfs.append(df)

    # Extract C-rate tag from run tags for labeling
    c_rate = next((t for t in run.tags if "C" in t), run.id)
    labels.append(c_rate)

# ---------------------------
# 2) Plot Voltage vs Capacity
# ---------------------------
plt.figure(figsize=(10,6))
for df, label in zip(dfs, labels):
    plt.plot(df["Discharge capacity [A.h]"], df["Voltage [V]"], label=label)

plt.xlabel("Discharge Capacity [Ah]")
plt.ylabel("Voltage [V]")
plt.title("CCD Comparison (DFN, Ai2020, Pouch Cell, varying C-rates)")
plt.legend()
plt.grid(True)
plt.tight_layout()

out_file = Path("compare_ccd_c-rates.png")
plt.savefig(out_file, dpi=150)

# ---------------------------
# 3) Push overlay as output
# ---------------------------
blt.output["ccd_overlay"] = str(out_file)
