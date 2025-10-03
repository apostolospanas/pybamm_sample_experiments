import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import balthazar as blt

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

# Model options
model_options = {"thermal": "lumped", "current collector": "uniform"}

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(os.getcwd()) / f"results_ai2020_{stamp}"
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
# 1) SIMULATE
# ---------------------------
csv_file = out(f"results_{MODEL_NAME}_{PARAM_SET}.csv")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    print(f"\n=== Running {MODEL_NAME} with {PARAM_SET} ===")
    sim = pybamm.Simulation(model, parameter_values=param,
                            experiment=experiment, solver=solver)
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

# ---------------------------
# 2) METRICS
# ---------------------------
metrics = {
    "capacity_Ah": float(df["Discharge capacity [A.h]"].max()),
    "end_voltage_V": float(df["Voltage [V]"].iloc[-1]),
    "energy_Wh": float(df["Discharge energy [W.h]"].iloc[-1]),
}
# Push metrics into Balthazar outputs
for k, v in metrics.items():
    blt.output[k] = f"{v:.4f}"

# ---------------------------
# 3) PLOTS
# ---------------------------
def points_per_cycle(df_len, n_cycles_):
    return df_len // n_cycles_ if n_cycles_ > 0 else df_len

# Voltage vs Time
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df["Time [h]"], df["Voltage [V]"], label=MODEL_NAME)
ax1.set_xlabel("Time [h]"); ax1.set_ylabel("Terminal Voltage [V]")
ax1.set_title(f"Voltage vs Time ({PARAM_SET})")
ax1.legend(); ax1.grid(True); fig1.tight_layout()
fig1.savefig(out("voltage_vs_time.png"), dpi=150)
plt.show()

# Voltage vs Discharge Capacity (Cycle 1)
fig2, ax2 = plt.subplots(figsize=(10, 6))
npts = points_per_cycle(len(df), n_cycles)
first_cycle = df.iloc[:npts]
ax2.plot(first_cycle["Discharge capacity [A.h]"], first_cycle["Voltage [V]"], label=MODEL_NAME)
ax2.set_xlabel("Discharge Capacity [Ah]"); ax2.set_ylabel("Voltage [V]")
ax2.set_title(f"Voltage vs Discharge Capacity (Cycle 1, {PARAM_SET})")
ax2.legend(); ax2.grid(True); fig2.tight_layout()
fig2.savefig(out("voltage_vs_capacity.png"), dpi=150)
plt.show()

# Capacity Retention vs Cycle
fig3, ax3 = plt.subplots(figsize=(8, 5))
npts = points_per_cycle(len(df), n_cycles)
cycle_ends = [df["Discharge capacity [A.h]"].iloc[min((i+1)*npts - 1, len(df)-1)]
              for i in range(n_cycles)]
ax3.plot(range(1, n_cycles+1), cycle_ends, marker='o', label=MODEL_NAME)
ax3.set_xlabel("Cycle"); ax3.set_ylabel("End-of-Discharge Capacity [Ah]")
ax3.set_title(f"Capacity Retention vs Cycle ({PARAM_SET})")
ax3.legend(); ax3.grid(True); fig3.tight_layout()
fig3.savefig(out("capacity_vs_cycle.png"), dpi=150)
plt.show()

# ---------------------------
# 4) SAVE SUMMARY
# ---------------------------
summary = pd.DataFrame([metrics])
summary_file = out(f"battery_model_summary_{PARAM_SET}.csv")
summary.to_csv(summary_file, index=False)

blt.output["summary_csv"] = str(summary_file)
blt.output["summary_table"] = summary
