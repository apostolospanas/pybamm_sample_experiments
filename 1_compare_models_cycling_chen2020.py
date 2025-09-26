import pybamm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIG ---
models = {
    "SPM": pybamm.lithium_ion.SPM(),
    "SPMe": pybamm.lithium_ion.SPMe(),
    "DFN": pybamm.lithium_ion.DFN(),
}
param = pybamm.ParameterValues("Chen2020")
n_cycles = 5
charge_rate = 1
discharge_rate = 1
charge_voltage = 4.2
discharge_voltage = 3.0

experiment = pybamm.Experiment([
    f"Discharge at {discharge_rate}C until {discharge_voltage}V",
    "Rest for 10 minutes",
    f"Charge at {charge_rate}C until {charge_voltage}V",
    "Rest for 10 minutes",
] * n_cycles, period="1 minute")

# --- 1. SIMULATE & EXPORT DATA ---
solutions = {}
metrics = {}

for name, model in models.items():
    csv_file = f"results_{name}_cycles.csv"
    if os.path.exists(csv_file):
        print(f"{csv_file} exists, skipping simulation for {name}.")
        df = pd.read_csv(csv_file)
    else:
        print(f"\n=== Running {name} ===")
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sol = sim.solve()
        solutions[name] = sol

        # Collect main variables
        df = pd.DataFrame({
            "Time [h]": sol["Time [h]"].entries,
            "Voltage [V]": sol["Terminal voltage [V]"].entries,
            "Current [A]": sol["Current [A]"].entries,
            "Discharge capacity [A.h]": sol["Discharge capacity [A.h]"].entries,
            "Discharge energy [W.h]": sol["Discharge energy [W.h]"].entries,
            "Power [W]": sol["Power [W]"].entries,
        })
        df.to_csv(csv_file, index=False)
        print(f"Saved {name} results to {csv_file}")

    # Store summary metrics
    try:
        metrics[name] = {
            "Initial capacity [Ah]": df["Discharge capacity [A.h]"].max(),
            "Final capacity [Ah]": df["Discharge capacity [A.h]"].iloc[-1],
            "Initial voltage [V]": df["Voltage [V]"].iloc[0],
            "Final voltage [V]": df["Voltage [V]"].iloc[-1],
        }
    except Exception as e:
        print(f"Warning: error collecting summary for {name}: {e}")
        metrics[name] = {}

# --- 2. LOAD ALL DATA FOR VISUALISATION ---
dfs = {name: pd.read_csv(f"results_{name}_cycles.csv") for name in models.keys()}

# --- 3. VOLTAGE vs TIME ---
plt.figure(figsize=(10, 6))
for name, df in dfs.items():
    plt.plot(df["Time [h]"], df["Voltage [V]"], label=name)
plt.xlabel("Time [h]")
plt.ylabel("Terminal Voltage [V]")
plt.title("Voltage vs Time")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# --- 4. VOLTAGE vs DISCHARGE CAPACITY (Cycle 1) ---
plt.figure(figsize=(10, 6))
for name, df in dfs.items():
    n_points = len(df) // n_cycles
    plt.plot(df["Discharge capacity [A.h]"].iloc[:n_points], df["Voltage [V]"].iloc[:n_points], label=name)
plt.xlabel("Discharge Capacity [Ah]")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs Discharge Capacity (Cycle 1)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# --- 5. CAPACITY RETENTION vs CYCLE ---
plt.figure(figsize=(8, 5))
for name, df in dfs.items():
    n_points = len(df) // n_cycles
    cycle_ends = [df["Discharge capacity [A.h]"].iloc[(i+1)*n_points - 1] for i in range(n_cycles)]
    plt.plot(range(1, n_cycles+1), cycle_ends, marker='o', label=name)
plt.xlabel("Cycle")
plt.ylabel("End-of-Discharge Capacity [Ah]")
plt.title("Capacity Retention vs Cycle")
plt.legend()
plt.tight_layout()
plt.show()

# --- 6. COULOMBIC EFFICIENCY per CYCLE ---
plt.figure(figsize=(8, 5))
for name, df in dfs.items():
    n_points = len(df) // n_cycles
    efficiency = []
    for i in range(n_cycles):
        cycle_df = df.iloc[i*n_points:(i+1)*n_points]
        discharge = cycle_df["Discharge capacity [A.h]"].max()
        charge = -cycle_df["Discharge capacity [A.h]"].min()  # crude, adjust if you have separate charge
        if charge > 0:
            eff = discharge / charge
            efficiency.append(eff)
    plt.plot(range(1, len(efficiency)+1), efficiency, marker='o', label=name)
plt.xlabel("Cycle")
plt.ylabel("Coulombic Efficiency")
plt.title("Coulombic Efficiency per Cycle")
plt.legend()
plt.tight_layout()
plt.show()

# --- 7. VOLTAGE HYSTERESIS: Charge/Discharge Overlay (Cycle 1) ---
plt.figure(figsize=(10, 6))
for name, df in dfs.items():
    n_points = len(df) // n_cycles
    first_cycle = df.iloc[:n_points]
    discharge_mask = first_cycle["Current [A]"] < 0
    charge_mask = first_cycle["Current [A]"] > 0
    plt.plot(first_cycle["Discharge capacity [A.h]"][discharge_mask], first_cycle["Voltage [V]"][discharge_mask], label=f"{name} Discharge")
    plt.plot(first_cycle["Discharge capacity [A.h]"][charge_mask], first_cycle["Voltage [V]"][charge_mask], label=f"{name} Charge", linestyle="--")
plt.xlabel("Capacity [Ah]")
plt.ylabel("Voltage [V]")
plt.title("Voltage Hysteresis: Charge (dashed) vs Discharge (solid) (Cycle 1)")
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. dQ/dV DIAGNOSTIC (Cycle 1, Discharge) ---
plt.figure(figsize=(10, 5))
for name, df in dfs.items():
    n_points = len(df) // n_cycles
    first_cycle = df.iloc[:n_points]
    discharge_mask = first_cycle["Current [A]"] < 0
    cap = first_cycle["Discharge capacity [A.h]"][discharge_mask].values
    volt = first_cycle["Voltage [V]"][discharge_mask].values
    if len(cap) > 2:
        dv = np.diff(volt)
        dq = np.diff(cap)
        dq_dv = dq / dv
        plt.plot(volt[1:], dq_dv, label=name)
plt.xlabel("Voltage [V]")
plt.ylabel("dQ/dV [Ah/V]")
plt.title("dQ/dV Diagnostic (Cycle 1, Discharge)")
plt.legend()
plt.tight_layout()
plt.show()

# --- 9. ENERGY DELIVERED per CYCLE ---
plt.figure(figsize=(8, 5))
for name, df in dfs.items():
    n_points = len(df) // n_cycles
    energies = [df["Discharge energy [W.h]"].iloc[(i+1)*n_points - 1] for i in range(n_cycles)]
    plt.plot(range(1, n_cycles+1), energies, marker='o', label=name)
plt.xlabel("Cycle")
plt.ylabel("Discharge Energy [Wh]")
plt.title("Discharge Energy per Cycle")
plt.legend()
plt.tight_layout()
plt.show()

# --- 10. SAVE SUMMARY METRICS ---
summary = pd.DataFrame(metrics).T
summary.to_csv("battery_model_comparison_summary.csv")
print("\nSaved summary metrics to battery_model_comparison_summary.csv")
print(summary)
