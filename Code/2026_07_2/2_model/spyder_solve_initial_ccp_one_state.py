"""Manual Spyder entry point for one structural solve from packaged initial CCPs.

Restart the Spyder kernel before running this file.  Edit ``LOCAL_MODEL_ROOT``
if the downloaded ZIP was extracted elsewhere.  Set breakpoints inside
``model_solution_em.py`` and start Spyder's debugger on this file.
"""

#%% Configure the isolated local model root before importing model modules.
import os
from pathlib import Path


LOCAL_MODEL_ROOT = Path(r"C:\tmp\spyder_initial_ccp_debug\Model")
TYPE_ID = 1
X1_INDEX = 48

os.environ["MODEL_ROOT"] = str(LOCAL_MODEL_ROOT)


#%% Import the model only after MODEL_ROOT has been set.
import numpy as np

from config import ENSURE_DEFAULT_TREE, EST
import model_solution_em as model


#%% Validate the packaged inputs and display the selected task.
required = [
    LOCAL_MODEL_ROOT / "Estimates" / "auxiliary_em_results.npz",
    LOCAL_MODEL_ROOT / "Estimates" / "param_g.npy",
    LOCAL_MODEL_ROOT / "Estimates" / "budgetshock_params.npy",
    LOCAL_MODEL_ROOT / "Output" / "cache" / "interp_dict.joblib",
]
required.extend(
    LOCAL_MODEL_ROOT
    / "Output"
    / "ccp"
    / str(period)
    / f"ccp_t{period}_[{model.invariant_states[X1_INDEX]}]_em{TYPE_ID}.npz"
    for period in range(1, model.T)
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing packaged inputs:\n" + "\n".join(missing))

ENSURE_DEFAULT_TREE(T=model.T)
model.reload_budgetshock_params()

x1 = model.invariant_states[X1_INDEX]
parameter_vector = np.asarray(np.load(EST("param_g.npy")), dtype=float)
utility_parameters = model.build_param_g(TYPE_ID, parameter_vector)

print("MODEL_ROOT:", LOCAL_MODEL_ROOT)
print("type ID:", TYPE_ID)
print("x1 index/value:", X1_INDEX, x1)
print("risk aversion:", model.bs.risk_aversion(model.budget_params, x1))


#%% Solve iteration zero using the packaged auxiliary initial CCPs.
# Important: ccp_real=0 loads the packaged initial CCP guess.
ccp_real = 0
solution_mode = 0
counterfactual = 0
maxdebt = True
models = 0

model.get_all_evt(
    X1_INDEX,
    model.invariant_states,
    model.debt_range,
    model.debt_range,
    ccp_real,
    utility_parameters,
    models,
    solution_mode,
    counterfactual,
    TYPE_ID,
    maxdebt,
)

print("Completed the one-state structural solve.")
