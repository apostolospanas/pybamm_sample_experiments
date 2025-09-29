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
PARAM_SET = "Ai2020"
MODEL_NAME = "SPMe"

n_cycles = 3
charge_rate_C = 1.0
discharge_rate_C = 1.0
v_charge_cutoff = 4.20
v_discharge_cutoff = 3.00
cv_tail_cutoff = 0.05
rest_between = "10 minutes"
period_resolution = "1 minute"

model_options = {"thermal": "lumped", "current collector": "uniform"}

pos_thickness_um_list = [40, 60, 80, 100, 120]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / f"results_pos_thickness_sweep_{stamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def out(name: str) -> str:
    return str(OUTPUT_DIR / name)


# ---------------------------
# MODEL & BASE PARAMETERS
# ---------------------------
if MODEL_NAME == "DFN":
    base_model = pybamm.lithium_ion.DFN(options=model_options)
else:
    base_model = pybamm.lithium_ion.SPMe(options=model_options)
base_param = pybamm.ParameterValues(PARAM_SET)
base_param.update({
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
# RUN SWEEP
# ---------------------------
summaries = []
dfs = {}

for pos_um in pos_thickness_um_list:
    pos_m = pos_um * 1e-6
    tag = f"{MODEL_NAME}_pos{pos_um:.0f}um"
    csv_file = out(f"results_{tag}.csv")

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        p = base_param.copy()
        p.update({"Positive electrode thickness [m]": pos_m})

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

    dfs[tag] = df

    metrics = {
        "Positive thickness [um]": pos_um,
        "Max discharge capacity [Ah]": float(df["Discharge capacity [A.h]"].max()),
        "End voltage [V]": float(df["Voltage [V]"].iloc[-1]),
        "Final discharge energy [Wh]": float(df["Discharge energy [W.h]"].iloc[-1]),
    }
    summaries.append(metrics)

    # Expose metrics to Balthazar
    for k, v in metrics.items():
        blt.output[f"{tag} - {k}"] = v
# ---------------------------
# VISUALISATION
# ---------------------------
import matplotlib
print("Matplotlib backend:", matplotlib.get_backend())  # shows in run log

def points_per_cycle(n, c): 
    return n // c if c > 0 else n

# 1) Voltage vs Time
fig1, ax1 = plt.subplots(figsize=(10, 6))
for tag, df in dfs.items():
    ax1.plot(df["Time [h]"], df["Voltage [V]"], label=tag)
ax1.set_xlabel("Time [h]"); ax1.set_ylabel("Terminal Voltage [V]")
ax1.set_title(f"Voltage vs Time — Positive electrode thickness sweep ({PARAM_SET})")
ax1.legend(); ax1.grid(True); fig1.tight_layout()

fig1_path = out("voltage_vs_time.png")
fig1.savefig(fig1_path, dpi=150)
blt.output["Voltage vs Time (png)"] = str(fig1_path)

plt.show(block=False)   # <-- show for Workbench preview
plt.pause(0.75)         # <-- give the UI time to capture

# 2) Discharge Energy vs Cycle
fig2, ax2 = plt.subplots(figsize=(8, 5))
for tag, df in dfs.items():
    npts = points_per_cycle(len(df), n_cycles)
    energies = [df["Discharge energy [W.h]"].iloc[min((i+1)*npts - 1, len(df)-1)]
                for i in range(n_cycles)]
    ax2.plot(range(1, n_cycles+1), energies, marker='o', label=tag)
ax2.set_xlabel("Cycle"); ax2.set_ylabel("Discharge Energy [Wh]")
ax2.set_title(f"Discharge Energy vs Cycle — Positive thickness sweep ({PARAM_SET})")
ax2.legend(); ax2.grid(True); fig2.tight_layout()

fig2_path = out("discharge_energy_vs_cycle.png")
fig2.savefig(fig2_path, dpi=150)
blt.output["Discharge Energy vs Cycle (png)"] = str(fig2_path)

plt.show(block=False)
plt.pause(0.75)

# (Optional) don't close figures; let the runner tear them down
# plt.close(fig1); plt.close(fig2)

# ---------------------------
# SAVE SUMMARY
# ---------------------------
summary_df = pd.DataFrame(summaries).sort_values("Positive thickness [um]")
summary_file = out("positive_thickness_sweep_summary.csv")
summary_df.to_csv(summary_file, index=False)

# Expose summary to Balthazar
blt.output["Summary CSV"] = str(summary_file)
blt.output["Summary Table"] = summary_df


