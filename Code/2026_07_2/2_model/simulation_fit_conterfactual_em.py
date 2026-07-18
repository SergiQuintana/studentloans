# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:50:26 2024

@author: Sergi
"""

import numpy as np
import scipy
import os
import time
#import numba
from numba import njit,jit,prange
import multiprocessing 
import multiprocessing as mp
from scipy.optimize import minimize
from os import environ
#print(environ['MKL_NUM_THREADS'])
import warnings

from numba.core.errors import NumbaPendingDeprecationWarning,NumbaDeprecationWarning

warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)

#mkl.set_num_threads(1)

#os.chdir(r"C:/Users/Sergi/Dropbox/PhD/Real Model")
mu = 0
gamma = 0.57721
beta = 0.98
r = 0.05
T = 10

# --------------------------------------------------------------------------- #
# Import the model components
# --------------------------------------------------------------------------- #
import model_solution_em as ms
import model_simulation_em as msim
#import model_solution_nodebt as msnodebt
#import model_simulation_nodebt as msimnodebt


#--------------------------#

from pathlib import Path
from config import DIR, OUT, INP, FUN, RDATA, CONT, EST, LIK
from latent_types import TYPE_IDS, load_em_posteriors

def _mk(p): Path(p).mkdir(parents=True, exist_ok=True)

# ensure runtime dirs exist (safe if they already exist)
for p in [
    # per-period VJT/EVT trees we read/write during sim:
    *[OUT("vjt", str(t)) for t in range(0, 11)],
    *[OUT("vjt_nog", str(t)) for t in range(0, 11)],
    *[OUT("vjt_conter", str(t)) for t in range(0, 11)],
    *[OUT("evt", str(t)) for t in range(0, 11)],
    *[OUT("evt_nog", str(t)) for t in range(0, 11)],
    *[OUT("evt_conter", str(t)) for t in range(0, 11)],
    # simulation outputs:
    OUT("choice"), OUT("state"), OUT("epsilon"), OUT("welfare"), OUT("grad_prob"), OUT("types"),
    # estimates, function coeffs & real data:
    DIR["MODEL_ESTIMATES"], DIR["MODEL_FUNCOEF"], DIR["MODEL_REALDATA"],
]:
    _mk(p)

# short aliases (mimic your earlier pattern)
pathfunctions   = DIR["MODEL_FUNCOEF"]
path_realdata   = DIR["MODEL_REALDATA"]
path_estimates  = DIR["MODEL_ESTIMATES"]

# helpers for vjt/evt container dirs
def _vjt_dir(period: int, conter: int = 0, maxdebt: bool = False) -> Path:
    if conter == 0:  # real
        return Path(OUT("vjt", str(period)))
    elif conter == 1:  # conter with/without maxdebt in filename
        return Path(OUT("vjt_conter", str(period)))
    else:  # conter_not
        return Path(OUT("vjt_conter_not", str(period)))

def _evt_dir(period: int, conter: int = 0, maxdebt: bool = False) -> Path:
    if conter == 0:
        return Path(OUT("evt", str(period)))
    elif conter == 1:
        return Path(OUT("evt_conter", str(period)))
    else:
        return Path(OUT("evt_nog", str(period)))  # you used evt_nog for “not” elsewhere

# single-place output roots
def _out_choice():  return Path(OUT("choice"))
def _out_state():   return Path(OUT("state"))
def _out_eps():     return Path(OUT("epsilon"))
def _out_welfare(): return Path(OUT("welfare"))
def _out_grad():    return Path(OUT("grad_prob"))


def solve_simulation_values(conterfactual, maxdebt, uparams, debt_range):
    """Solve every invariant state for all permanent joint types."""
    solution_mode = 1
    ccp_real = []
    models = []
    args = [
        (
            i, ms.invariant_states, debt_range, debt_range, ccp_real,
            uparams[type_id - 1], models, solution_mode, conterfactual,
            type_id, maxdebt,
        )
        for type_id in TYPE_IDS
        for i in range(np.shape(ms.invariant_states)[0])
    ]
    with multiprocessing.Pool(
        60, initializer=ms.reload_budgetshock_params
    ) as pool_obj:
        pool_obj.starmap(ms.get_all_evt, args, chunksize=1)


def simulate_cohorts(conterfactual, maxdebt, q, uparams, samples=30):
    # ``sigma_u`` remains in the legacy public interface.  The simulation now
    # reloads parental-income/type risk aversion from the canonical budget file.
    sigma_u = 1.4
    args = [
        (cohort, sigma_u, conterfactual, q, maxdebt, uparams)
        for cohort in range(1, samples + 1)
    ]
    with multiprocessing.Pool(samples) as pool_obj:
        pool_obj.starmap(msim.simulate_choices, args, chunksize=1)


if __name__ == '__main__':
    debt_range = ms.get_debt_range()
    uparams = [ms.load_param_g(type_id, real=1) for type_id in TYPE_IDS]
    q = load_em_posteriors(EST("auxiliary_em_results.npz"))

    print("Solving and simulating the estimated baseline model...")
    solve_simulation_values(conterfactual=0, maxdebt=True,
                            uparams=uparams, debt_range=debt_range)
    simulate_cohorts(conterfactual=0, maxdebt=True, q=q, uparams=uparams)

    print("Solving and simulating the income-driven-repayment counterfactual...")
    solve_simulation_values(conterfactual=1, maxdebt=True,
                            uparams=uparams, debt_range=debt_range)
    simulate_cohorts(conterfactual=1, maxdebt=True, q=q, uparams=uparams)

    # The separate no-debt solver/simulation still uses the legacy two-type
    # interface.  It is intentionally not run from this sixteen-type driver.
