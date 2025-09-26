# PyBaMM Sample Experiments

PyBaMM scripts for **cycling experiments**, **parameter sweeps**, and **model comparisons**.  
This repository demonstrates how to use PyBaMM with different models and parameters (Chen2020, Ai2020) to study battery behaviour.

---

##  Repository structure
\\\
experiments/   # individual cycling experiments
sweeps/        # parameter sweeps
requirements.txt
.gitignore
README.md
\\\

---

## ⚡ Getting started

### 1. Clone the repo
\\\ ash
git clone https://github.com/apostolospanas/pybamm_sample_experiments.git
cd pybamm_sample_experiments
\\\

### 2. Create a virtual environment
\\\ ash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Mac/Linux
\\\

### 3. Install dependencies
\\\ ash
pip install -r requirements.txt
\\\

---

## 🧪 Running experiments

### Run a cycling experiment (DFN, Ai2020, 1C)
\\\ash
python experiments/2_dfn_cycling_pouchcell_ai2020_1C.py
\\\

### Run a separator thickness sweep
\\\ash
python sweeps/separator_thickness_sweep_pouchcell_dfn_ai2020.py
\\\

### Run a cathode thickness sweep
\\\ash
python sweeps/cathode_thickness_sweep_pouchcell_SPM_Ai2020.py
\\\

Each script will:
- Run a PyBaMM simulation (cycling or sweep)
- Save results as .csv in a results folder
- Generate diagnostic plots (voltage curves, capacity fade, dQ/dV, etc.)

---

##  Outputs
- **CSV results**: stored under results_*/
- **Plots**: displayed during execution (voltage vs time, capacity retention, dQ/dV, etc.)
- **Summary files**: with capacity/energy metrics per run or per sweep

---

##  Models & Parameter Sets
- **Chen2020**: widely used for NMC/graphite cell validation
- **Ai2020**: pouch-cell parameter set
- Models: SPM, SPMe, DFN (single particle, reduced-order, Doyle–Fuller–Newman)

---

##  License
MIT License — free to use, modify, and share.
