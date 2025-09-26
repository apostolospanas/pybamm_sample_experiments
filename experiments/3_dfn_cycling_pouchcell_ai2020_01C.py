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
MODEL_NAME = "DFN"        # or "SPMe" for faster testing
PARAM_SET = "Ai2020"      # pouch-cell parameter set

# Cycling protocol
n_cycles = 5
charge_rate_C = 0.1
discharge_rate_C = 0.1
v_charge_cutoff = 4.20
v_discharge_cutoff = 3.00
cv_tail_cutoff = 0.05
rest_between = "10 minutes"
period_resolution = "1 minute"

# Model options (pass to model constructor, not Simulation)
model_options = {
    "thermal": "lumped",
    "current collector": "uniform",
}

# Timestamp for outputs
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent / "results_ai2020"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def out(name: str) -> str:
    return str(OUTPUT_DIR / name)

# ---------------------------
# MODEL SETUP
# ---------------------------
if MODEL_NAME == "DFN":
    model = pybamm.lithium_ion.DFN(options=model_options)
elif MODEL_NAME == "SPMe":
    model = pybamm.lithium_ion.SPMe(options=model_options)
else:
    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

models = {MODEL_NAME: model}

param = pybamm.ParameterValues(PARAM_SET)
param.update({
    "Upper voltage cut-off [V]": v_charge_cutoff,
    "Lower voltage cut-off [V]": v_discharge_cutoff,
    "Ambient temperature [K]": 298.15,
})

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
# 1) SIMULATE & EXPORT DATA
# ---------------------------
solutions = {}
metrics = {}

for name, model in models.items():
    csv_file = out(f"results_{name}_{PARAM_SET}_{stamp}.csv")
    if os.path.exists(csv_file):
        print(f"{csv_file} exists, skipping simulation for {name}.")
        df = pd.read_csv(csv_file)
    else:
        print(f"\n=== Running {name} with {PARAM_SET} ===")
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            experiment=experiment,
            solver=solver
        )
        sol = sim.solve()
        solutions[name] = sol

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
        print(f"Saved {name} results to {csv_file}")

    metrics[name] = {
        "Max discharge capacity [Ah]": float(df["Discharge capacity [A.h]"].max()),
        "End voltage [V]": float(df["Voltage [V]"].iloc[-1]),
    }

# ---------------------------
# 2) LOAD DATA
# ---------------------------
dfs = {name: pd.read_csv(out(f"results_{name}_{PARAM_SET}_{stamp}.csv"))
       for name in models.keys()}

def points_per_cycle(df_len, n_cycles_):
    return df_len // n_cycles_ if n_cycles_ > 0 else df_len

# ---------------------------
# 3) PLOTS
# ---------------------------
plt.figure(figsize=(10, 6))
for name, df in dfs.items():
    plt.plot(df["Time [h]"], df["Voltage [V]"], label=name)
plt.xlabel("Time [h]"); plt.ylabel("Terminal Voltage [V]")
plt.title(f"Voltage vs Time ({PARAM_SET})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 6))
for name, df in dfs.items():
    npts = points_per_cycle(len(df), n_cycles)
    first_cycle = df.iloc[:npts]
    plt.plot(first_cycle["Discharge capacity [A.h]"], first_cycle["Voltage [V]"], label=name)
plt.xlabel("Discharge Capacity [Ah]"); plt.ylabel("Voltage [V]")
plt.title(f"Voltage vs Discharge Capacity (Cycle 1, {PARAM_SET})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
for name, df in dfs.items():
    npts = points_per_cycle(len(df), n_cycles)
    cycle_ends = [df["Discharge capacity [A.h]"].iloc[min((i+1)*npts - 1, len(df)-1)]
                  for i in range(n_cycles)]
    plt.plot(range(1, n_cycles+1), cycle_ends, marker='o', label=name)
plt.xlabel("Cycle"); plt.ylabel("End-of-Discharge Capacity [Ah]")
plt.title(f"Capacity Retention vs Cycle ({PARAM_SET})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ---------------------------
# 4) SAVE SUMMARY
# ---------------------------
summary = pd.DataFrame(metrics).T
summary_file = out(f"battery_model_summary_{PARAM_SET}_{stamp}.csv")
summary.to_csv(summary_file)
print(f"\nSaved summary metrics to {summary_file}")
print(summary)
