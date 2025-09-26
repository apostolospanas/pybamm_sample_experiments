import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
PARAM_SET = "Ai2020"
MODEL_NAME = "DFN"

# Cycling protocol (same as before)
n_cycles = 3
charge_rate_C = 1.0
discharge_rate_C = 1.0
v_charge_cutoff = 4.20
v_discharge_cutoff = 3.00
cv_tail_cutoff = 0.05
rest_between = "10 minutes"
period_resolution = "1 minute"

# DFN model options
model_options = {"thermal": "lumped", "current collector": "uniform"}

# Sweep the **cathode** (positive electrode) thickness in micrometres
pos_thickness_um_list = [40, 60, 80, 100, 120]  # edit as you wish

# Timestamp & output dir
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(__file__).resolve().parent / f"results_pos_thickness_sweep_{stamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def out(name: str) -> str:
    return str(OUTPUT_DIR / name)

# ---------------------------
# MODEL & BASE PARAMETERS
# ---------------------------
if MODEL_NAME == "SPM":
    base_model = pybamm.lithium_ion.DFN(options=model_options)
else:
    base_model = pybamm.lithium_ion.SPMe(options=model_options)

base_param = pybamm.ParameterValues(PARAM_SET)
base_param.update({
    "Upper voltage cut-off [V]": v_charge_cutoff,
    "Lower voltage cut-off [V]": v_discharge_cutoff,
    "Ambient temperature [K]": 298.15,
})

# Report defaults
default_pos_thickness_m = float(base_param["Positive electrode thickness [m]"])
print(f"Default Ai2020 positive electrode thickness: {default_pos_thickness_m*1e6:.1f} µm")

# Experiment
experiment = pybamm.Experiment(
    [
        f"Discharge at {discharge_rate_C}C until {v_discharge_cutoff} V",
        f"Rest for {rest_between}",
        f"Charge at {charge_rate_C}C until {v_charge_cutoff} V",
        f"Hold at {v_charge_cutoff} V until {cv_tail_cutoff}C",
        f"Rest for {rest_between}",
    ] * n_cycles,
    period=period_resolution
)

def make_solver():
    try:
        return pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6)
    except Exception:
        return pybamm.ScipySolver(atol=1e-6, rtol=1e-6)

solver = make_solver()

# ---------------------------
# RUN SWEEP
# ---------------------------
summaries = []
dfs = {}

for pos_um in pos_thickness_um_list:
    pos_m = pos_um * 1e-6
    tag = f"{MODEL_NAME}_pos{pos_um:.0f}um"
    csv_file = out(f"results_{tag}.csv")

    if os.path.exists(csv_file):
        print(f"{csv_file} exists, skipping simulation for {tag}.")
        df = pd.read_csv(csv_file)
    else:
        print(f"\n=== Running {tag} with {PARAM_SET} ===")

        # Copy base params and ONLY change the cathode thickness
        p = base_param.copy()
        p.update({"Positive electrode thickness [m]": pos_m})

        # Fresh simulation with updated parameters
        sim = pybamm.Simulation(
            base_model,
            parameter_values=p,
            experiment=experiment,
            solver=solver,
        )
        sol = sim.solve()

        out_dict = {
            "Time [h]": sol["Time [h]"].entries,
            "Voltage [V]": sol["Terminal voltage [V]"].entries,
            "Current [A]": sol["Current [A]"].entries,
            "Discharge capacity [A.h]": sol["Discharge capacity [A.h]"].entries,
            "Discharge energy [W.h]": sol["Discharge energy [W.h]"].entries,
            "Power [W]": sol["Power [W]"].entries,
        }
        df = pd.DataFrame(out_dict)
        df.to_csv(csv_file, index=False)
        print(f"Saved {tag} results to {csv_file}")

    dfs[tag] = df

    # Simple metrics
    metrics = {
        "Positive thickness [um]": pos_um,
        "Max discharge capacity [Ah]": float(df["Discharge capacity [A.h]"].max()),
        "End voltage [V]": float(df["Voltage [V]"].iloc[-1]),
        "Final discharge energy [Wh]": float(df["Discharge energy [W.h]"].iloc[-1]),
    }
    summaries.append(metrics)

# ---------------------------
# VISUALISATION
# ---------------------------
# Voltage vs Time
plt.figure(figsize=(10, 6))
for tag, df in dfs.items():
    plt.plot(df["Time [h]"], df["Voltage [V]"], label=tag)
plt.xlabel("Time [h]"); plt.ylabel("Terminal Voltage [V]")
plt.title(f"Voltage vs Time — Positive electrode thickness sweep ({PARAM_SET})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Discharge Energy per Cycle (approx using equal segmentation)
def points_per_cycle(n, c): return n // c if c > 0 else n

plt.figure(figsize=(8, 5))
for tag, df in dfs.items():
    npts = points_per_cycle(len(df), n_cycles)
    energies = [df["Discharge energy [W.h]"].iloc[min((i+1)*npts - 1, len(df)-1)]
                for i in range(n_cycles)]
    plt.plot(range(1, n_cycles+1), energies, marker='o', label=tag)
plt.xlabel("Cycle"); plt.ylabel("Discharge Energy [Wh]")
plt.title(f"Discharge Energy vs Cycle — Positive thickness sweep ({PARAM_SET})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ---------------------------
# SAVE SUMMARY
# ---------------------------
summary_df = pd.DataFrame(summaries).sort_values("Positive thickness [um]")
summary_file = out("positive_thickness_sweep_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"\nSaved summary to {summary_file}")
print(summary_df)
